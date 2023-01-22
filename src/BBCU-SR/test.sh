#x2
CUDA_VISIBLE_DEVICES=4 python3 bbcu/test.py -opt options/test/bbcu/test_BBCUL_x2.yml

CUDA_VISIBLE_DEVICES=4 python3 bbcu/test.py -opt options/test/bbcu/test_BBCUM_x2.yml

#x4

CUDA_VISIBLE_DEVICES=4 python3 bbcu/test.py -opt options/test/bbcu/test_BBCUL_x4.yml

CUDA_VISIBLE_DEVICES=4 python3 bbcu/test.py -opt options/test/bbcu/test_BBCUM_x4.yml