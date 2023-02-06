#denoising
CUDA_VISIBLE_DEVICES=0 python3 main_train_dncnn_denoising_b.py -opt ./options/train_cdncnn_denoising_b.json

#deblocking
#CUDA_VISIBLE_DEVICES=0 python3 main_train_dncnn3_deblocking_b.py -opt ./options/train_dncnn3_deblocking_b.json 