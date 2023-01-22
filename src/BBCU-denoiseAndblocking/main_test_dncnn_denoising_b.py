import os.path
import logging
import argparse

import numpy as np
from datetime import datetime
from collections import OrderedDict
# from scipy.io import loadmat

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util


'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/DnCNN

@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)

by Kai Zhang (12/Dec./2019)
'''

"""
# --------------------------------------------
|--model_zoo          # model_zoo
   |--dncnn_15        # model_name
   |--dncnn_25
   |--dncnn_50
   |--dncnn_gray_blind
   |--dncnn_color_blind
   |--dncnn3
|--testset            # testsets
   |--set12           # testset_name
   |--bsd68
   |--cbsd68
|--results            # results
   |--set12_dncnn_15  # result_name = testset_name + '_' + model_name
   |--set12_dncnn_25
   |--bsd68_dncnn_15
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dncnn_50', help='dncnn_15, dncnn_25, dncnn_50, dncnn_gray_blind, dncnn_color_blind, dncnn3')
    parser.add_argument('--testset_name', type=str, default='bsd68', help='test set, bsd68 | set12| urban100')
    parser.add_argument('--noise_level_img', type=int, default=50, help='noise level: 15, 25, 50')
    parser.add_argument('--x8', type=bool, default=False, help='x8 to boost performance')
    parser.add_argument('--show_img', type=bool, default=False, help='show the image')
    parser.add_argument('--model_pool', type=str, default='model_zoo', help='path of model_zoo')
    parser.add_argument('--testsets', type=str, default='testsets', help='path of testing folder')
    parser.add_argument('--results', type=str, default='results', help='path of results')
    parser.add_argument('--need_degradation', type=bool, default=True, help='add noise or not')
    parser.add_argument('--task_current', type=str, default='dn', help='dn for denoising, fixed!')
    parser.add_argument('--sf', type=int, default=1, help='unused for denoising')
    args = parser.parse_args()

    n_channels = 3        # fixed, 1 for grayscale image, 3 for color image
    nb = 20               # fixed

    urban100_root = "/root/datasets/benchmark/Urban100/HR"
    kodak24_root = "/root/datasets/testsets/Kodak24"
    cbsd68_root = "/root/datasets/testsets/CBSD68"
    dataset = [urban100_root, cbsd68_root, kodak24_root]
    model_list = ["./checkpoints/BBCU-denoise.pth"]
    model_name_list = ['dncnn_color_blind']
    testset_name_list = ["urban100","cbsd68","kodak24"]
    noise_level_list = [15,25,50]
    np.random.seed(0)
    for i in range(len(dataset)):
        for j in range(len(model_list)):
            for k in noise_level_list:
                args.testset_name=testset_name_list[i]
                args.noise_level_img=k
                args.model_name = model_name_list[j]
                L_path =  dataset[i]
                model_path = model_list[j]
                result_name = args.testset_name + "_%02d"%(k)+'_' + args.model_name     # fixed
                border = args.sf if args.task_current == 'sr' else 0        # shave boader to calculate PSNR and SSIM

                # ----------------------------------------
                # L_path, E_path, H_path
                # ----------------------------------------

                H_path = L_path                               # H_path, for High-quality images
                E_path = os.path.join(args.results, result_name)   # E_path, for Estimated images
                util.mkdir(E_path)

                if H_path == L_path:
                    args.need_degradation = True
                logger_name = result_name
                utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
                logger = logging.getLogger(logger_name)

                need_H = True if H_path is not None else False
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # ----------------------------------------
                # load model
                # ----------------------------------------

                from models.network_dncnn_bin import DnCNN as net
                model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R',k=130.)
                model.load_state_dict(torch.load(model_path), strict=True)
                model.eval()
                for k, v in model.named_parameters():
                    v.requires_grad = False
                model = model.to(device)
                logger.info('Model path: {:s}'.format(model_path))
                number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
                logger.info('Params number: {}'.format(number_parameters))

                test_results = OrderedDict()
                test_results['psnr'] = []
                test_results['ssim'] = []

                logger.info('model_name:{}, image sigma:{}'.format(args.model_name, args.noise_level_img))
                logger.info(L_path)
                L_paths = util.get_image_paths(L_path)
                H_paths = util.get_image_paths(H_path) if need_H else None

                for idx, img in enumerate(L_paths):

                    # ------------------------------------
                    # (1) img_L
                    # ------------------------------------

                    img_name, ext = os.path.splitext(os.path.basename(img))
                    # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
                    img_L = util.imread_uint(img, n_channels=n_channels)
                    img_L = util.uint2single(img_L)

                    if args.need_degradation:  # degradation process
                        np.random.seed(seed=0)  # for reproducibility
                        img_L += np.random.normal(0, args.noise_level_img/255., img_L.shape)

                    util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(args.noise_level_img)) if args.show_img else None

                    img_L = util.single2tensor4(img_L)
                    img_L = img_L.to(device)

                    # ------------------------------------
                    # (2) img_E
                    # ------------------------------------

                    if not args.x8:
                        img_E = model(img_L)
                    else:
                        img_E = utils_model.test_mode(model, img_L, mode=3)

                    img_E = util.tensor2uint(img_E)

                    if need_H:

                        # --------------------------------
                        # (3) img_H
                        # --------------------------------

                        img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
                        img_H = img_H.squeeze()

                        # --------------------------------
                        # PSNR and SSIM
                        # --------------------------------

                        psnr = util.calculate_psnr(img_E, img_H, border=border)
                        ssim = util.calculate_ssim(img_E, img_H, border=border)
                        test_results['psnr'].append(psnr)
                        test_results['ssim'].append(ssim)
                        logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
                        util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if args.show_img else None

                    # ------------------------------------
                    # save results
                    # ------------------------------------

                    util.imsave(img_E, os.path.join(E_path, img_name+ext))

                if need_H:
                    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
                    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
                    logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

if __name__ == '__main__':

    main()
