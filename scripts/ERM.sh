dataset="PACS"
algorithm="ERM"
random_seed=0
shared_args="--max_epoch 100 --net resnet18  --checkpoint_freq 1 --task img_dg --dataset $dataset \
         --algorithm $algorithm  --aug_policy default --batch_size 32 --seed $random_seed"
python train.py --lr 1e-3 --test_envs 0  --output n2-1 $shared_args
python train.py --lr 1e-3 --test_envs 1  --output n2-3 $shared_args
python train.py --lr 1e-3 --test_envs 2  --output n2-5 $shared_args
python train.py --lr 1e-3 --test_envs 3  --output n2-7 $shared_args

