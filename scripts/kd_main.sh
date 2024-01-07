dataset="PACS"
#dataset="office-home"
shared_args="--max_epoch 100  --dataset $dataset --gpu_id 7 --optim sgd \
--batch_size 16 --lr 2e-3 --kd_T 4 --alpha 0.4"
python train_student.py --test_envs 0  $shared_args --seed 0 --distill kd
python train_student.py --test_envs 1  $shared_args --seed 0 --distill kd
python train_student.py --test_envs 2  $shared_args --seed 0 --distill kd
python train_student.py --test_envs 3  $shared_args --seed 0 --distill kd