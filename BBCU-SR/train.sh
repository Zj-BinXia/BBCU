CUDA_VISIBLE_DEVICES=3 bash ./scripts/dist_train.sh 1 ./options/train/bbcu/train_BBCUL_x4.yml --auto_resume
CUDA_VISIBLE_DEVICES=4 bash ./scripts/dist_train.sh 1 ./options/train/bbcu/train_BBCUL_x2.yml --auto_resume
