import argparse
import sys
import time

import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.data.datasets import OfficeHome, VisDA17, Office31

# custom
from trainers import *


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.model_dir:
        cfg.MODEL_DIR = args.model_dir
        if args.trainer == 'CLIP_ZS' or args.trainer == 'CLIP_LR':
            cfg.MODEL_DIR = None
        
    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.gpu:   
        cfg.GPU = args.gpu
    
    if args.save:
        cfg.SAVE_MODLE = args.save

    if args.kd:
        cfg.TRAINER.CDUTARGET.KD_WEIGHT= args.kd


        
    if args.domains:
        cfg.DOMAINS = args.domains
        cfg.WARM_UP = 0
        if cfg.DATASET.NAME == "OfficeHome":
            DOMAINS = {'a': "art", 'c':"clipart", 'p':"product", 'r':"real_world"}
        elif cfg.DATASET.NAME == "VisDA17":
            DOMAINS = {'s': "synthetic", 'r':"real"}
        elif cfg.DATASET.NAME == "Office31":
            DOMAINS = {'a': "amazon", 'w': "webcam", 'd': "dslr"}
        elif cfg.DATASET.NAME == "DomainNet":
            DOMAINS = {'c': "clipart", 'i': "infograph", 'p': "painting", 'q': "quickdraw", 'r': "real", 's': "sketch"}
        source_domain, target_domain = args.domains.split('-')[0], args.domains.split('-')[1]
        cfg.DATASET.SOURCE_DOMAINS = [DOMAINS[source_domain]]
        cfg.DATASET.TARGET_DOMAINS = [DOMAINS[target_domain]]


def extend_cfg(cfg, args):
    """
    Add new config variables for your method.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.MODEL.BACKBONE.PATH = "./assets"    # path of pretrained model
    cfg.MODEL.PATCH_SIZE = 16
    cfg.MODEL.HIDDEN_SIZE = 768     # as model change, this param need to be changed
    cfg.MODEL.NUM_LAYER = 12        # as model change, this param need to be changed
    cfg.DATASET.NUM_SHOTS = None    # optional
    cfg.SAVE_MODEL = True
    cfg.TEST.FINAL_MODEL == "best_val"
    
    if args.trainer == 'CLIP_ZS' or args.trainer == 'CLIP_LR' or args.trainer == 'CLIP_FC' or args.trainer == 'CLIP_FT':
        cfg.TRAINER.CLIP = CN()
        cfg.TRAINER.CLIP.PREC = "fp16"  # fp16, fp32, amp   

    elif args.trainer == 'CDUSOURCE':
        cfg.TRAINER.CDUSOURCE = CN()
        cfg.TRAINER.CDUSOURCE.PREC = "fp16"
        cfg.TRAINER.CDUSOURCE.DROPOUT = 0.0
        cfg.TRAINER.CDUSOURCE.DEEP_LAYERS = None
        cfg.TRAINER.CDUSOURCE.SHARE_LAYER = cfg.TRAINER.CDUSOURCE.DEEP_LAYERS

        cfg.TRAINER.CDUSOURCE.TP = True
        cfg.TRAINER.CDUSOURCE.T_DEEP = True
        cfg.TRAINER.CDUSOURCE.CSC = False
        cfg.TRAINER.CDUSOURCE.N_CTX = 2  # number of text context vectors
        cfg.TRAINER.CDUSOURCE.CTX_INIT = "a photo of a"
        cfg.TRAINER.CDUSOURCE.CLASS_TOKEN_POSITION = "end"

        cfg.TRAINER.CDUSOURCE.VP = True
        cfg.TRAINER.CDUSOURCE.V_DEEP = cfg.TRAINER.CDUSOURCE.T_DEEP
        cfg.TRAINER.CDUSOURCE.NUM_TOKENS = cfg.TRAINER.CDUSOURCE.N_CTX  # number of visual context vectors
        cfg.TRAINER.CDUSOURCE.LOCATION = "middle"
        cfg.TRAINER.CDUSOURCE.TEXT_LOSS_WEIGHT = 25
        cfg.TRAINER.CDUSOURCE.IMAGE_LOSS_WEIGHT = 10
        cfg.TRAINER.CDUSOURCE.GPA_MEAN = 15
        cfg.TRAINER.CDUSOURCE.GPA_STD = 1


    elif args.trainer == 'CDUTARGET':
        cfg.TRAINER.CDUTARGET = CN()
        cfg.TRAINER.CDUTARGET.PREC = "fp16"
        cfg.TRAINER.CDUTARGET.DROPOUT = 0.0
        cfg.TRAINER.CDUTARGET.DEEP_LAYERS = None
        cfg.TRAINER.CDUTARGET.SHARE_LAYER = cfg.TRAINER.CDUTARGET.DEEP_LAYERS

        cfg.TRAINER.CDUTARGET.TP = True
        cfg.TRAINER.CDUTARGET.T_DEEP = True
        cfg.TRAINER.CDUTARGET.CSC = False
        cfg.TRAINER.CDUTARGET.N_CTX = 2  # number of text context vectors
        cfg.TRAINER.CDUTARGET.CTX_INIT = "a photo of a"
        cfg.TRAINER.CDUTARGET.CLASS_TOKEN_POSITION = "end"

        cfg.TRAINER.CDUTARGET.VP = True
        cfg.TRAINER.CDUTARGET.V_DEEP = cfg.TRAINER.CDUTARGET.T_DEEP
        cfg.TRAINER.CDUTARGET.NUM_TOKENS = cfg.TRAINER.CDUTARGET.N_CTX  # number of visual context vectors
        cfg.TRAINER.CDUTARGET.LOCATION = "middle"
        cfg.TRAINER.CDUTARGET.PROJECT_LAYER = 2
        cfg.TRAINER.CDUTARGET.CE_WEIGHT = 0.0
        cfg.TRAINER.CDUTARGET.KD_WEIGHT= 100.0
        cfg.TRAINER.CDUTARGET.TEMPERATURE = 1.0
        cfg.TRAINER.CDUTARGET.TEACHER_NAME = "ViT/L-14"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)
    print(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    setup_logger(cfg.OUTPUT_DIR)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(cfg.MODEL_DIR, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="./results", help="output directory")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="",
                        help="path to config file for dataset setup")
    parser.add_argument("--model-dir", type=str, default="",
                        help="load model from this directory for eval-only mode")
    
    parser.add_argument("--domains", type=str, help="domains for DA/DG")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")

    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    
    parser.add_argument("--resume", type=str, default="",
                        help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--load-epoch", type=int,
                        help="load model weights at this epoch for evaluation")

    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    
    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")    # if you use this hyperpameter, you need modify the source code of dassl library.
                                                                                    # i.e., in dassl.engine.trainer line 314: self.device = torch.device("cuda:{}".format(cfg.GPU))
    parser.add_argument("--kd", type=int, default=0, help="kd_weight")
    parser.add_argument("--seed", type=int, default=2,
                        help="only positive value enables a fixed seed")
    parser.add_argument("--save", type=str, default=False, help="need to save model")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")

    args = parser.parse_args()
    
    main(args)
