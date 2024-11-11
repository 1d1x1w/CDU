import copy
import os
import os.path as osp
import sys
import datetime
import time
from itertools import chain

import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from openTSNE import TSNE
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint, MetricMeter, AverageMeter
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tqdm import tqdm


from clip.model import convert_weights

from trainers.baseda import Base_PromptLearner
from utils.clip_part import TextEncoder, ImageEncoder_Trans, load_clip_to_cpu, ImageEncoder_Conv

_tokenizer = _Tokenizer()


class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )

    def forward(self, input_feat):
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))

        return final_feat.squeeze(-1).squeeze(-1)


def load_clip_to_cpu_teacher(cfg):
    backbone_name = cfg.TRAINER.CDUTARGET.TEACHER_NAME

    if backbone_name == "ViT-L/14":
        model_path = "assets/ViT-L-14.pt"
    elif backbone_name == "ViT-B/16":
        model_path = "assets/ViT-B-16.pt"
    else:
        print("teaher model name is false")
        sys.exit()

    print(f"CLIP Teacher name is {backbone_name}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # We default use PromptSRC to pretrain our teacher model

    model = clip.build_model(state_dict or model.state_dict())
    return model




class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CDUTARGET.N_CTX
        ctx_init = cfg.TRAINER.CDUTARGET.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # text encoder hidden size(512)
        self.dim = clip_model.text_projection.shape[1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.tp = cfg.TRAINER.CDUTARGET.TP
        self.vp = cfg.TRAINER.CDUTARGET.VP
        self.t_deep = cfg.TRAINER.CDUTARGET.T_DEEP
        self.v_deep = cfg.TRAINER.CDUTARGET.V_DEEP
        self.num_tokens = cfg.TRAINER.CDUTARGET.NUM_TOKENS  # number of prompted tokens
        self.deep_layer = cfg.TRAINER.CDUTARGET.DEEP_LAYERS  # num of layer has cdu ([1,3]: 1~3 layer has)
        self.location = cfg.TRAINER.CDUTARGET.LOCATION
        self.prompt_dropout = nn.Dropout(cfg.TRAINER.CDUTARGET.DROPOUT)
        self.num_layer = cfg.MODEL.NUM_LAYER
        self.hidden_size = clip_model.visual.conv1.weight.shape[0]  # visual encoder hiden size(768)

        self.ctx = None

        if ctx_init and n_ctx <= 4:  # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            self.ctx = nn.Parameter(ctx_vectors)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)

        vctx_vectors = torch.empty(n_ctx, self.hidden_size, dtype=dtype)
        nn.init.normal_(vctx_vectors, std=0.02)
        self.vctx = nn.Parameter(vctx_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of target model context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)

        self.device = torch.device("cuda:{}".format(cfg.GPU))
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

        self.dim = clip_model.text_projection.shape[1]

    def forward(self):
        vctx = self.vctx

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [65, 16, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = self.construct_prompts(ctx, prefix, suffix)


        return prompts, vctx


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        if cfg.TRAINER.CDUTARGET.TEACHER_NAME == "ViT-L/14":
            if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
                self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
                self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 768)
            else:  # RN50, RN101
                self.image_encoder = ImageEncoder_Conv(cfg, clip_model)
                if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'RN101' :
                    self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 768)
                else :
                    self.VPT_image_trans = Feature_Trans_Module_two_layer(1024, 768)
        else:
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
            self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 512)
            
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)


        self.cfg = cfg
        self.device = torch.device("cuda:{}".format(cfg.GPU))
        self.VPT_image_trans = self.VPT_image_trans.to(self.device)
        convert_weights(self.VPT_image_trans)

    def forward(self, image):
        _, vctx = self.prompt_learner()

        image_features = self.image_encoder(image.type(self.dtype), vctx)       # [8, 1024]
        image_features = self.VPT_image_trans(image_features)  # [B, 768]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        return image_features, logit_scale


class CustomCLIP_teacher(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
        self.device = torch.device("cuda:{}".format(cfg.GPU))
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner).to(self.device)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image=None):
        prompts, vctx  = self.prompt_learner()

        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image.type(self.dtype), vctx)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute the prompted logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # [B, C]

        return image_features, text_features, logits


@TRAINER_REGISTRY.register()
class CDUTARGET(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.CDUTARGET.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg

        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)

        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL

        output_dir = cfg.OUTPUT_DIR
        path_parts = output_dir.split('/')
        self.results_file ='/'.join(path_parts[:7])+ '/' + cfg.DATASET.NAME + ".csv"
        self.t_sne_path = '/'.join(path_parts[:7])

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        clip_model_teacher = load_clip_to_cpu_teacher(cfg)

        if cfg.TRAINER.CDUTARGET.PREC == "fp32" or cfg.TRAINER.CDUTARGET.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()


        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        self.model_teacher = CustomCLIP_teacher(cfg, classnames, clip_model_teacher)
        self.model_teacher.to(self.device)

        dataset_name = cfg.DATASET.NAME.lower()

        if cfg.TRAINER.CDUTARGET.TEACHER_NAME == "ViT-B/16":
            model_path = "/Workpalce_sdc/dxw/CDU/output/cdusource/CDUSOURCE/" + dataset_name + "/b32_ep20_" + dataset_name + "/ViT-B16/deepFalse_middle/" + cfg.DOMAINS + "_ntok4/PromptLearner/model-best.pth.tar"
        else:
            if dataset_name == "visda17":
                model_path = "/Workpalce_sdc/dxw/CDU/output/cdusource/CDUSOURCE/"+ dataset_name + "/b32_ep10_" + dataset_name[:-2] + "/ViT-L14/deepFalse_middle_img_w=25/"+ cfg.DOMAINS+"_ntok4/PromptLearner/model-best.pth.tar"
            elif dataset_name == "domainnet":
                model_path = "/Workpalce_sdc/dxw/CDU/output/cdusource/CDUSOURCE/" + dataset_name + "/b32_ep10_" + dataset_name + "/ViT-L14/deepFalse_middle/" + cfg.DOMAINS + "_ntok4/PromptLearner/model-best.pth.tar"
            else:
                model_path = "/Workpalce_sdc/dxw/CDU/output/cdusource/CDUSOURCE/" + dataset_name + "/b32_ep20_" + dataset_name + "/ViT-L14/deepFalse_middle/" + cfg.DOMAINS + "_ntok4/PromptLearner/model-best.pth.tar"
        print(model_path)
        # checkpoint = load_checkpoint(model_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["state_dict"]

        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]
        if "prompt_learner.token_prefix2" in state_dict:
            del state_dict["prompt_learner.token_prefix2"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]
        if "prompt_learner.token_suffix2" in state_dict:
            del state_dict["prompt_learner.token_suffix2"]

        self.model_teacher.load_state_dict(state_dict, strict=False)
        self.model_teacher.eval()

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if name_to_update in name:
                param.requires_grad_(True)
            if "VPT" in name:
                param.requires_grad_(True)

        Sum_Memory = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                Sum_Memory += param.numel() * param.element_size() / (1024 ** 2)
                print(str(name) + " " + str(param.requires_grad) + " " + str(
                    (param.numel() * param.element_size()) / (1024 ** 2)) + "MB")
        print("Total Memory : " + str(Sum_Memory) + "MB")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer

        self.trainable_list = nn.ModuleList([])
        self.trainable_list.append(self.model)

        self.optim = build_optimizer(self.trainable_list, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("PromptLearner", self.model, self.optim, self.sched)

        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1

        self.scaler = GradScaler() if cfg.TRAINER.CDUTARGET.PREC == "amp" else None
        self.temperature = cfg.TRAINER.CDUTARGET.TEMPERATURE

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)

        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError('Training batch name is wrong!')

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 \
                    or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch -
                              1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                    self.epoch + 1,
                    self.max_epoch,
                    self.batch_idx + 1,
                    self.num_batches,
                    batch_time=batch_time,
                    data_time=data_time,
                    eta=eta,
                    losses=losses,
                    lr=self.get_current_lr(),
                ))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def forward_backward(self, batch_x, batch_u):
        image, label = self.parse_batch_train(batch_u)

        with torch.no_grad():
            tea_image_features, tea_text_features, tea_logits = self.model_teacher(image)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.CDUTARGET.PREC
        if prec == "amp":
            with autocast():
                loss = model(image)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            image_ft, logit_scale = model(image)

            stu_logits = logit_scale * image_ft @ tea_text_features.t().detach()

            L_ukd = F.kl_div(
                F.log_softmax(stu_logits / self.temperature, dim=1),
                F.softmax(tea_logits / self.temperature, dim=1),
                reduction='sum',
            ) * (self.temperature * self.temperature) / stu_logits.numel()  # 求平均

            loss = self.cfg.TRAINER.CDUTARGET.KD_WEIGHT * L_ukd
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def after_epoch(self):
        start_test = time.time()
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
                                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            curr_result = self.test()
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                if self.save:
                    self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")
            self.set_model_mode("train")

        if self.save and (meet_checkpoint_freq or last_epoch):
            self.save_model(self.epoch, self.output_dir)

        end_test = time.time()
        test_time = end_test - start_test
        print(f"Model inference time: {test_time} seconds")


    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_prefix2" in state_dict:
                del state_dict["prompt_learner.token_prefix2"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            if "prompt_learner.token_suffix2" in state_dict:
                del state_dict["prompt_learner.token_suffix2"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def T_SNE_combined(self):
        self.set_model_mode("eval")
        #
        all_embeddings = []
        all_labels = []

        combined_loader = chain(self.train_loader_x, self.train_loader_u)

        for batch_idx, batch in enumerate(combined_loader):
            input, label = self.parse_batch_test(batch)

            prompts, vctx = self.model.prompt_learner()
            image_features = self.model.image_encoder(input.type(self.model.dtype), vctx)
            #     image_features = self.model.VPT_image_trans(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(image_features.cpu().numpy())
            if batch_idx < len(self.train_loader_x):
                all_labels.extend([0] * len(label))
            else:
                all_labels.extend([1] * len(label))

        all_embeddings = np.vstack(all_embeddings)

        tsne = TSNE(perplexity=50, metric="euclidean", random_state=42)
        embeddings = tsne.fit(all_embeddings)

        print(numpy.array(all_embeddings))

        source_mask = np.array(all_labels) == 0
        target_mask = np.array(all_labels) == 1

        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings[source_mask, 0], embeddings[source_mask, 1],
                    color='blue', marker='o', s=96, label='Source domain', alpha=0.5)
        plt.scatter(embeddings[target_mask, 0], embeddings[target_mask, 1],
                    color='red', marker='o', s=96, label='Target domain', alpha=0.5)

        plt.xticks(())
        plt.yticks(())

        out_dir = str(self.output_dir)
        last_three_chars = out_dir[-9:-6]
        last_three_chars_upper = last_three_chars.upper()
        print(self.t_sne_path + '/CDU' + ' (' + last_three_chars_upper + ')' + '.pdf')
        plt.title('CDU' + ' (' + last_three_chars_upper + ')', fontdict={"family": "Times New Roman", "size": 64})

        plt.savefig(self.t_sne_path + '/CDU' + ' (' + last_three_chars_upper + ')' + '.pdf')

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        # self.T_SNE_combined()
        # sys.exit()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train":
            data_loader = self.train_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")


        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)

            with torch.no_grad():
                # start_time1 = time.time()
                tea_image_features, tea_text_features, tea_logits = self.model_teacher(image)
            #     end_time1 = time.time()
            #     inference_time1 = end_time1 - start_time1
            #     print(f"Model inference time: {inference_time1} seconds")
            #
            # start_time = time.time()
            image_ft, logit_scale = self.model(image)

            output = logit_scale * image_ft @ tea_text_features.t()
            # end_time = time.time()
            # inference_time = end_time - start_time
            # print(f"Model inference time: {inference_time} seconds")
            # sys.exit()

            self.evaluator.process(output, label)


        if self.cfg.DATASET.NAME == "VisDA17":
            results, accs = self.evaluator.evaluate()
        else:
            results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        # 检查文件是否存在
        file_exists = os.path.isfile(self.results_file)

        if self.cfg.DATASET.NAME == "VisDA17":

            columns = ['epoch'] + ['acc_{}'.format(i + 1) for i in range(len(accs))] + ['avg']  # 10个accuracy列

            # 初始化DataFrame
            if not file_exists:
                df = pd.DataFrame(columns=columns)
            else:
                df = pd.read_csv(self.results_file)

            # 将epoch和accuracy_list合并成一个字典
            row_data = {'epoch': self.epoch + 1}  # epoch从1开始计数
            for i, acc in enumerate(accs):
                row_data['acc_{}'.format(i + 1)] = acc

            row_data['avg'] = results["perclass_accuracy"]
            df = df.append(row_data, ignore_index=True)

            # 保存DataFrame到CSV文件时不保存索引
            df.to_csv(self.results_file, index=False)

            return results["perclass_accuracy"]

        # 初始化DataFrame
        if not file_exists:
            initial_data = {'epoch': list(range(1, self.cfg.OPTIM.MAX_EPOCH + 1))}
            df = pd.DataFrame(initial_data)
        else:
            df = pd.read_csv(self.results_file)

        # 确保DataFrame有epoch行
        if len(df) < self.cfg.OPTIM.MAX_EPOCH:
            df = df.reindex(range(self.cfg.OPTIM.MAX_EPOCH))
            df['epoch'] = list(range(1, self.cfg.OPTIM.MAX_EPOCH + 1))

        # 确保新列存在
        if self.domains not in df.columns:
            df[self.domains] = ''

        # 迭代并添加新数据
        # 更新DataFrame中的特定行
        df.at[self.epoch, self.domains] = results["accuracy"]

        # 保存DataFrame到CSV文件时不保存索引
        df.to_csv(self.results_file, index=False)


        return list(results.values())[0]

