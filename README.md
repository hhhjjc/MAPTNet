# Multi-Scale Adaptive Prototype Transformer Network for Few-shot Strip Steel Surface Defect Segmentation

We propose a few-shot segmentation method named Multi-scale Adaptive Prototype Transformer Network (MAPTNet), which aims to integrate multi-scale feature aggregation and improve adaptability of defect detection for diverse and complex defect scenarios.

![method](https://github.com/user-attachments/assets/7d77e0f7-68e1-4833-add8-189e3fd6b71b)




## Get Started

### Environment

- python == 3.8.19
- torch == 2.3.0
- torchvision == 0.18.0
- cuda == 12.1
- tqdm == 4.66.5
- opencv-python == 4.10.0.84
- tensorboardX == 2.6.2.2

### Dataset

Please download the following dataset from [CPANet](https://github.com/VDT-2048/CPANet) and put them into the `../FSSD-12` directory.
#add
For this project, we use the integrated ESDIs-SOD dataset, which is based on the original A3Net's datasets. The integrated dataset combines multiple steel surface defect image sets with standardized annotations to provide a comprehensive benchmark for few-shot segmentation tasks.
* Original data source: [A3Net](https://github.com/VDT-2048/A3Net) - We would like to thank the authors for their valuable contribution
* Our [integrated ESDIs-SOD dataset](https://drive.google.com/file/d/1_WgpoqHX-u5X_KDEFkkmMkeTnWuXMXyl/view)

If you use this dataset in your research, please cite both our paper and the original A3Net paper.

## Models

We have adopted the same procedures as [CPANet](https://github.com/VDT-2048/CPANet) for the pre-trained backbones, placing them in the `./initmodel` directory. 

## Scripts

- First update the configurations in the `./config` for training or testing

- Train script
```
sh train.sh
```
- Test script
```
sh test.sh
```


## References

This repository owes its existence to the exceptional contributions of other projects:

* CPANet: https://github.com/VDT-2048/CPANet
* ...

Many thanks for their excellent work.

## Question
If you have any question, welcome email me at 'hhhjjc@hdu.edu.cn'


## BibTeX

If you find our work and this repository useful. Please consider giving a star and citation.

```bibtex
@article{huang2025multi,
  title={Multi-Scale Adaptive Prototype Transformer Network for Few-shot Strip Steel Surface Defect Segmentation},
  author={Huang, Jiacheng and Wu, Yong and Zhou, Xiaofei and Lin, Jia and Chen, Zhangping and Zhang, Guodao and Xia, Lei and Zhang, Jiyong},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2025},
  publisher={IEEE}
}
```
