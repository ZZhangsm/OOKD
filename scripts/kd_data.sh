dataset="PACS"
shared_args="--max_epoch 100  --dataset $dataset --gpu_id 0 --optim sgd \
--batch_size 16 --lr 2e-3 --kd_T 4 --alpha 0.4 --distill kd"

python train_student.py --test_envs 0  $shared_args --seed 0 --aug_policy standard --output_dir aug_standard
python train_student.py --test_envs 1  $shared_args --seed 0 --aug_policy standard --output_dir aug_standard
python train_student.py --test_envs 2  $shared_args --seed 0 --aug_policy standard --output_dir aug_standard
python train_student.py --test_envs 3  $shared_args --seed 0 --aug_policy standard --output_dir aug_standard


python train_student.py --test_envs 0  $shared_args --seed 0 --aug_policy standard --output_dir aug_standard
python train_student.py --test_envs 0  $shared_args --seed 0 --aug_policy randaugment --output_dir aug_rand
python train_student.py --test_envs 0  $shared_args --seed 0 --aug_policy autoaugment --output_dir aug_auto
python train_student.py --test_envs 0  $shared_args --seed 0 --algorithm Mixup --output_dir alg_mixup
python train_student.py --test_envs 0  $shared_args --seed 0 --algorithm CutMix --output_dir alg_cutmix
python train_student.py --test_envs 0  $shared_args --seed 1 --algorithm DomainMix --output_dir alg_domainmix
python train_student.py --test_envs 0  $shared_args --seed 0 --s_net resnet18_ms_l123 --output_dir alg_mixstyle
python train_student.py --test_envs 0  $shared_args --seed 0 --prune random --prune_ratio 0.75 --output_dir prune_rand
python train_student.py --test_envs 0  $shared_args --seed 0 --prune el2n --prune_ratio 0.75 --output_dir prune_el2n
python train_student.py --test_envs 0  $shared_args --seed 0 --prune grand --prune_ratio 0.75 --output_dir prune_grand
python DeepDG/train_student.py --test_envs 2  $shared_args --seed 0 --noise 1 --output_dir noise

