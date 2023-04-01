# BBCU-SR

This project is the official implementation of 'Basic Binary Convolution Unit for Binarized Image Restoration Network', ICLR2023
> **Basic Binary Convolution Unit for Binarized Image Restoration Network [[Paper](https://arxiv.org/pdf/2210.00405.pdf)] [[Project](https://github.com/Zj-BinXia/BBCU)]**

This is code for BBCU-denoiseAndblocking (for denoising and deblocking)

<p align="center">
  <img src="figs/method.jpg" width="50%">
</p>

---

##  Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.10](https://pytorch.org/)


## Dataset Preparation

We train our network with  [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 images).

---

## Training (1 V100 GPUs)

```bash


#denoising
CUDA_VISIBLE_DEVICES=0 python3 main_train_dncnn_denoising_b.py -opt ./options/train_cdncnn_denoising_b.json

#deblocking
#CUDA_VISIBLE_DEVICES=0 python3 main_train_dncnn3_deblocking_b.py -opt ./options/train_dncnn3_deblocking_b.json 
```


---

## :european_castle: Model Zoo

Please download checkpoints from [Google Drive](https://drive.google.com/drive/folders/1MRZejm6JqnQKRXnLCxZmryRCatWnk31g).

---
## Testing
```bash
#x2
CUDA_VISIBLE_DEVICES=4 python3 bbcu/test.py -opt options/test/bbcu/test_BBCUL_x2.yml

CUDA_VISIBLE_DEVICES=4 python3 bbcu/test.py -opt options/test/bbcu/test_BBCUM_x2.yml

#x4

CUDA_VISIBLE_DEVICES=4 python3 bbcu/test.py -opt options/test/bbcu/test_BBCUL_x4.yml

CUDA_VISIBLE_DEVICES=4 python3 bbcu/test.py -opt options/test/bbcu/test_BBCUM_x4.yml
```
---
## Results
<p align="center">
  <img src="figs/quan-SR.jpg" width="90%">
</p>
<p align="center">
  <img src="figs/qual-SR.jpg" width="90%">
</p>

---

## BibTeX
    @InProceedings{xia2022basic,
      title={Basic Binary Convolution Unit for Binarized Image Restoration Network},
      author={Xia, Bin and Zhang, Yulun and Wang, Yitong and Tian, Yapeng and Yang, Wenming and Timofte, Radu and Van Gool, Luc},
      journal={ICLR},
      year={2023}
    }

## 📧 Contact

If you have any question, please email `zjbinxia@gmail.com`.
