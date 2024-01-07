# OOKD: Knowledge Distillation under Distribution Shift

This repo:

"Revisiting Knowledge Distillation under Distribution Shift" . [Paper](https://arxiv.org/abs/2312.16242)



![head](https://github.com/ZZhangsm/OOKD/blob/main/scripts/head.png)

**benchmarks 14 knowledge distillation methods in PyTorch, including:**

- (KD) - Distilling the Knowledge in a Neural Network

- (FitNet) - Fitnets: hints for thin deep nets

- (AT) - Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer

- (FT) - Paraphrasing Complex Network: Network Compression via Factor Transfer

- (AB) - Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons

- (NST) - Like what you like: knowledge distill via neuron selectivity transfer

- (SP) - Similarity-Preserving Knowledge Distillation

- (PKT) - Probabilistic Knowledge Transfer for deep representation learning

- (VID) - Variational Information Distillation for Knowledge Transfer

- (RKD) - Relational Knowledge Distillation

- (CC) - Correlation Congruence for Knowledge Distillation

- (CRD) - Contrastive Representation Distillation

- (ReviewKD) - [Distilling knowledge via knowledge review](https://scholar.google.com/scholar_url?url=http://openaccess.thecvf.com/content/CVPR2021/html/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.html&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=13522722160846228296&ei=t1OaZY_2HILYyQSdhKf4Ag&scisig=AFWwaearhD51ud-GKHC_oAhoJtxi)

- (DKD) - [Decoupled knowledge distillation](https://scholar.google.com/scholar_url?url=http://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=6183306406495914013&ei=olOaZa3cO4iyyASFsIeACQ&scisig=AFWwaeZliyVWHQO4uDafgIE3EFBD)

- (DiffKD) - [Knowledge Diffusion for Distillation](https://scholar.google.com/scholar_url?url=https://arxiv.org/abs/2305.15712&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=4615443208731882220&ei=klOaZbXcLbSx6rQP-4Of0A4&scisig=AFWwaeYndQMZIYIxgN7PoDwmZAxc)

  

**benchmarks 12 data manipulation methods in PyTorch, including:**

- Data Augmentation
  - (ImageNet baseline)
  - (AutoAugment) - [Autoaugment: Learning augmentation strategies from data](https://scholar.google.com/scholar_url?url=http://openaccess.thecvf.com/content_CVPR_2019/html/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.html&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=8586846733647481474&ei=RywIZeD9FquJ6rQP87mMqAI&scisig=AFWwaeZv1QAKL2BSZndnFSNoCgUf)
  - (RandAugment) - [Randaugment: Practical automated data augmentation with a reduced search space](https://scholar.google.com/scholar_url?url=http://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=10903590128857500448&ei=USwIZaSLDoG7yATum7PwBg&scisig=AFWwaeb1D0Tlcr-nn5gQIyBKVU2S)
  - (Mixup) - [**mixup**: Beyond empirical risk minimization](https://scholar.google.com/scholar_url?url=https://arxiv.org/abs/1710.09412&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=12669856454801555406&ei=NCwIZfv1IoLyyATpwqGYDA&scisig=AFWwaeYYx7R2aCkkVvdTRUjqhK-w)
  - (CutMix) - [**Cutmix**: Regularization strategy to train strong classifiers with localizable features](https://scholar.google.com/scholar_url?url=http://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=15618183235315733915&ei=WiwIZYulE432yATflamIBA&scisig=AFWwaea00JaGyqOMGlJHw_hBa3m3)
  - (DomainMix) - [**Domainmix**: Learning generalizable person re-identification without human annotations](https://scholar.google.com/scholar_url?url=https://arxiv.org/abs/2011.11953&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=7576833949867176269&ei=YiwIZcb-FpHHywTmq5JA&scisig=AFWwaeZPR4eUNfB7oJhS-92kVfyb)
  - (MixStyle) - [Domain generalization with **mixstyle**](https://scholar.google.com/scholar_url?url=https://arxiv.org/abs/2104.02008&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=4489212027125038279&ei=aywIZbypA5eI6rQPlKy8mAo&scisig=AFWwaeZPbmJyvOo4_vQ1tIhZ11vl)
  - Gaussian noise
- Data Pruning
  - (Random Prune) - pruning the data with random selection
  - (EL2N) - [Deep learning on a data diet: Finding important examples early in training](https://scholar.google.com/scholar_url?url=https://proceedings.neurips.cc/paper_files/paper/2021/hash/ac56f8fe9eea3e4a365f29f0f1957c55-Abstract.html&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=6692350500928309521&ei=eCwIZaepMKyR6rQPk7mgiAI&scisig=AFWwaeZ2fu3keoPPQe2umMGfqtKH)
  - (GraNd) - [Deep learning on a data diet: Finding important examples early in training](https://scholar.google.com/scholar_url?url=https://proceedings.neurips.cc/paper_files/paper/2021/hash/ac56f8fe9eea3e4a365f29f0f1957c55-Abstract.html&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=6692350500928309521&ei=eCwIZaepMKyR6rQPk7mgiAI&scisig=AFWwaeZ2fu3keoPPQe2umMGfqtKH)

#  Datasets

Our code supports the following dataset:

- Office-Home
- PACS
- DomainNet
- ColorMNIST
- CelebA-Blond

# Running

## Fetch the pretrained teacher models by:

```python
# For example: Dataset PACS
dataset="PACS"
algorithm="ERM"
random_seed=0
shared_args="--max_epoch 100 --net resnet18  --checkpoint_freq 1 --task img_dg --dataset $dataset --algorithm $algorithm --aug_policy default --batch_size 32 --seed $random_seed"
python train_teacher.py --lr 1e-3 --test_envs 0  --output output_dir $shared_args
```



## Run distillation for out-of-distribution datasets.

1. An example of running Decoupled Knowledge Distillation (DKD) is given by:

```python
dataset="PACS"
shared_args="--max_epoch 100  --dataset $dataset --gpu_id 1 --optim sgd --s_net resnet10\
--batch_size 16 --lr 2e-3 --kd_T 4 --alpha 0.4"
python train_student.py --test_envs 0  $shared_args --seed 0 --distill DKD --path_t my/dir/teacher_path
```

where the flags are explained as:

- `--path_t`: specify the path of the teacher model
- `--s_net`: specify the student model, see 'model' to check the available model types.
- `--distill`: specify the distillation method
- `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
- `-a`: the weight of the KD loss, default: `None`
- `-b`: the weight of other distillation losses, default: `None`
- `--test_envs`: the test environments for out-of-distribution dataset



2. An example of running Augmentation is given by in scripts/kd_data.sh

 Note: the default setting is for a single-GPU training. If you would like to play this repo with multiple GPUs, you might need to tune the learning rate, which empirically needs to be scaled up linearly with the batch size, see [this paper](https://arxiv.org/abs/1706.02677)

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@misc{zhang2023revisiting,
      title={Revisiting Knowledge Distillation under Distribution Shift}, 
      author={Songming Zhang and Ziyu Lyu and Xiaofeng Chen},
      year={2023},
      eprint={2312.16242},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

For any questions, please contact Songming Zhang (sm.zhang1@siat.ac.cn)

# Acknowledgment

Great thanks to [DomainBed](https://github.com/facebookresearch/DomainBed)，[RepDistiller](https://github.com/HobbitLong/RepDistiller)，[Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)，**[transferlearning/DeepDG](https://github.com/jindongwang/transferlearning/tree/60d89070549701c4a75fe5b1ac625264820d5ca8/code/DeepDG)**. 



