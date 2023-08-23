# BBCU

This project is the official implementation of 'Basic Binary Convolution Unit for Binarized Image Restoration Network', ICLR2023
> **Basic Binary Convolution Unit for Binarized Image Restoration Network [[Paper](https://arxiv.org/pdf/2210.00405.pdf)] [[Project](https://github.com/Zj-BinXia/BBCU)]**


We provide [Pretrained Models](https://drive.google.com/drive/folders/1MRZejm6JqnQKRXnLCxZmryRCatWnk31g?usp=sharing) for super-resolution, denoising, and deblocking.

<p align="center">
  <img src="figs/method.jpg" width="50%">
</p>

---

##  Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.10](https://pytorch.org/)


### Installation

1. Clone repo

    ```bash
    git clone git@github.com:Zj-BinXia/BBCU.git
    ```

2. If you want to train or test BBCU for super-resolution

    ```bash
    cd BBCU-SR
    ```
    
3. If you want to train or test BBCU for denoising and deblocking

    ```bash
    cd BBCU-denoiseAndblocking
    ```
**It is notable that our amplification factor k for residual alignment is used to balance the value range gap of full-precision residual branch and binarized Conv branch as input image range is 0-1. The best k∗ is related to the number of feature channels n, which empirically fits k∗ = 130n/64. You can adjust it according to your network setting.**

**More details please see the README in folder of BBCU-SR and BBCU-denoiseAndblocking** 

---
## BibTeX

    @article{xia2022basic,
      title={Basic Binary Convolution Unit for Binarized Image Restoration Network},
      author={Xia, Bin and Zhang, Yulun and Wang, Yitong and Tian, Yapeng and Yang, Wenming and Timofte, Radu and Van Gool, Luc},
      journal={ICLR},
      year={2023}
    }

## 📧 Contact

If you have any question, please email `zjbinxia@gmail.com`.
