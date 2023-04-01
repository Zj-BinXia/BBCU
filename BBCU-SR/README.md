# BBCU-SR

This project is the official implementation of 'Basic Binary Convolution Unit for Binarized Image Restoration Network', ICLR2023
> **Basic Binary Convolution Unit for Binarized Image Restoration Network [[Paper](https://arxiv.org/pdf/2210.00405.pdf)] [[Project](https://github.com/Zj-BinXia/BBCU)]**

This is code for BBCU-SR (for super-resolution)

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

Train x4 SR BBCUL

```bash

CUDA_VISIBLE_DEVICES=3 bash ./scripts/dist_train.sh 1 ./options/train/bbcu/train_BBCUL_x4.yml --auto_resume
```

Train x2 SR BBCUL

```bash

CUDA_VISIBLE_DEVICES=4 bash ./scripts/dist_train.sh 1 ./options/train/bbcu/train_BBCUL_x2.yml --auto_resume
```

Train x4 SR BBCUM

```bash

CUDA_VISIBLE_DEVICES=3 bash ./scripts/dist_train.sh 1 ./options/train/bbcu/train_BBCUM_x4.yml --auto_resume
```

Train x2 SR BBCUM

```bash

CUDA_VISIBLE_DEVICES=4 bash ./scripts/dist_train.sh 1 ./options/train/bbcu/train_BBCUM_x2.yml --auto_resume
```



---

## :european_castle: Model Zoo

Please download checkpoints from [Google Drive](https://drive.google.com/drive/folders/1MRZejm6JqnQKRXnLCxZmryRCatWnk31g).

---
## Testing
```bash
python3  kdsrgan/test.py -opt options/test_kdsrgan_x4ST.yml 
```
---
## Results
<p align="center">
  <img src="images/quan.jpg" width="90%">
</p>


---

## BibTeX

    @InProceedings{xia2022knowledge,
      title={Knowledge Distillation based Degradation Estimation for Blind Super-Resolution},
      author={Xia, Bin and Zhang, Yulun and Wang, Yitong and Tian, Yapeng and Yang, Wenming and Timofte, Radu and Van Gool, Luc},
      journal={ICLR},
      year={2023}
    }

## 📧 Contact

If you have any question, please email `zjbinxia@gmail.com`.
