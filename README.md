# Cross-domain Distillation for Unsupervised Domain Adaptation with Large Vision-language Models



------

## Highlights

![Architecture](https://github.com/1d1x1w/CDU/blob/main/Architecture.png)

> **Abstract:** Large vision-language models (VLMs), incorporating the prompt learning mechanism, have achieved promising results in cross-domain tasks. However, leveraging VLMs to transfer the knowledge from the source domain to the target domain remains a challenging task for unsupervised domain adaptation (UDA). To this end, we propose \textbf{\underline{C}}ross-domain \textbf{\underline{D}}istillation for \textbf{\underline{U}}DA with LVMs (termed as CDU). Firstly, CDU trains a source model by embedding the knowledge of the source domain (including both each sample and its corresponding class category) into VLMs in a lightweight manner. Secondly, CDU makes full use of the image and text semantics from the source model to guide the target model learning, thereby achieving domain alignment to yield semantically consistent representations across domains. We conduct extensive experiments on 4 popular UDA datasets including Office-31, Office-Home, Mini-DomainNet and VisDA-2017. Experimental results verify our method consistently surpasses the state-of-the-art (SOTA) UDA methods by a large margin with higher performance and lower model complexity on various UDA benchmarks. Take Office-Home as an example, the average accuracy of CDU exceeds existing methods by at least 3\%, yet the number of learnable parameters only accounts for 17.9\% and the inference time only takes up 4.3\% compared to those of others. The anonymous code of this paper is available at GitHub: https://github.com/1d1x1w/CDU.

## Main Contributions



- **New perspective.** To the best of our knowledge, this is the first attempt that leverages both the visual and textual semantic information of VLMs to transfer knowledge from the source domain to the target domain for UDA.

- **Novel method：** We introduce a novel UDA approach CDU to implement lightweight cross-domain distillation that makes full use of both the image and text semantics of the source domain, generated by VLMs, to simultaneously guide the image features generation and text label prediction for the target domain.
  
- **High Performance：** We conduct extensive experiments on 4 popular UDA datasets including Office-31, Office-Home, Mini-DomainNet and VisDA-2017. The experimental results validate the superiority of our CDU which achieves higher performance with lower model complexity compared with the state-of-the-art (SOTA) UDA methods on various cross-domain tasks.

------

## Results



### PMCC in comparison with existing prompt tuning methods



Results reported below show accuracy across 4 UDA datasets with ViT-B/16 backbone. Our PMCC method adopts the paradigm of multi-modal prompt tuning.

| Name                                         | Office-Home Acc. | Office-31 Acc. | VisDA-2017 Acc. |
| -------------------------------------------- | ---------------- | -------------- | --------------- |
| [CLIP](https://arxiv.org/abs/2103.00020)     | 82.1             | 77.5           | 88.9            |
| [CoOp](https://arxiv.org/abs/2109.01134)     | 83.9             | 89.4           | 82.7            |
| [CoCoOp](https://arxiv.org/abs/2203.05557)   | 84.1             | 88.9           | 84.2            |
| [VPT-deep](https://arxiv.org/abs/2203.17274) | 83.9             | 89.4           | 86.2            |
| [MaPLe](https://arxiv.org/abs/2210.03117)    | 84.2             | 89.6           | 83.5            |
| [DAPL](https://arxiv.org/abs/2202.06687)     | 84.4             | 81.2           | 89.5            |
| [PDA](https://arxiv.org/abs/2312.09553)      | 85.7             | 91.2           | 89.6            |
| **CDU(Ours)**                                | **90.1**         | **94.0**       | **89.8**        |

| Name                                                     | Mini-DomainNet Acc. |
| -------------------------------------------------------- | ------------------- |
| [DeiT](https://arxiv.org/abs/2012.12877)                 | 55.1                |
| [ViT](https://arxiv.org/abs/2010.11929)                  | 57.5                |
| [CLIP](https://arxiv.org/abs/2103.00020)                 | 69.3                |
| [SSRT](https://arxiv.org/abs/2204.07683)                 | 65.4                |
| [CDTrans](https://arxiv.org/abs/2109.06165)              | 63.2                |
| [DAPL](https://arxiv.org/abs/2202.06687)                 | 73.6                |
| [PMTrans](https://arxiv.org/abs/2303.13434)              | 69.6                |
| [PADCLIP](https://ieeexplore.ieee.org/document/10377727) | 74.7                |
| [UniMoS](https://ieeexplore.ieee.org/document/10656339/) | 76.0                |
| **CDU(Ours)**                                            | **78.0**            |



## Installation



For installation and other package requirements, please follow the instructions as follows. This codebase is tested on Ubuntu 22.04 LTS with python 3.7. Follow the below steps to create environment and install dependencies.

- Setup conda environment.

```
# Create a conda environment
conda create -y -n cdu python=3.7

# Activate the environment
conda activate cdu

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/get-started/previous-versions/ if your cuda version is different
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```



- Install dassl library.

```
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```



- Clone PMCC code repository and install requirements.

```
# Clone CDU code base
git clone https://github.com/1d1x1w/CDU.git
cd CDU

# Install requirements
pip install -r requirements.txt
```



## Data preparation



Please follow the instructions as follows to prepare all datasets. Datasets list:

- [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?pli=1&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)
- [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code)
- [VisDA-2017](http://ai.bu.edu/visda-2017/#download)
- [DomainNet](http://csr.bu.edu/ftp/visda/2019/multi-source)

------

## Training and Evaluation



Please follow the instructions for training, evaluating and reproducing the results. Firstly, you need to **modify the directory of data by yourself**.

### Source Model Training

```
# Example: trains on Office-Home dataset, and the source domian is art and the target domain is clipart (a-c)
bash scripts/cdu/main_cdusource.sh officehome b32_ep20_officehome CDUSOURCE ViT-L/14 4 a-c 0
```



### **Target Model Training**

```
# Example: trains on Office-Home dataset, and the source domian is art and the target domain is clipart (a-c)
bash scripts/cdu/main_cdutarget.sh officehome b32_ep20_officehome CDUTARGET ViT-B/16 4 a-c 0
```



### Evaluation

```
# evaluates on Office-Home dataset, and the source domian is art and the target domain is clipart (a-c)
bash scripts/cdu/eval_cdutarget.sh officehome b32_ep20_officehome CDUTARGET ViT-B/16 4 a-c 0
```



The details are at each method folder in [scripts folder](https://github.com/246dxw/CDU/tree/main/scripts).



## Acknowledgements



Our style of reademe refers to [PDA](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment). And our code is based on [CoOp and CoCoOp](https://github.com/KaiyangZhou/CoOp), [DAPL](https://github.com/LeapLabTHU/DAPrompt/tree/main) ，[MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning)  and [PDA](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment) etc. repository. We thank the authors for releasing their code. If you use their model and code, please consider citing these works as well. Supported methods are as follows:

| Method       | Paper                                          | Code                                                         |
| ------------ | ---------------------------------------------- | ------------------------------------------------------------ |
| CoOp         | [IJCV 2022](https://arxiv.org/abs/2109.01134)  | [link](https://github.com/KaiyangZhou/CoOp)                  |
| CoCoOp       | [CVPR 2022](https://arxiv.org/abs/2203.05557)  | [link](https://github.com/KaiyangZhou/CoOp)                  |
| VPT          | [ECCV 2022](https://arxiv.org/abs/2203.17274)  | [link](https://github.com/KMnP/vpt)                          |
| IVLP & MaPLe | [CVPR 2023](https://arxiv.org/abs/2210.03117)  | [link](https://github.com/muzairkhattak/multimodal-prompt-learning) |
| DAPL         | [TNNLS 2023](https://arxiv.org/abs/2202.06687) | [link](https://github.com/LeapLabTHU/DAPrompt)               |
| PDA          | [AAAI 2024](https://arxiv.org/abs/2312.09553)  | [link](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment) |

