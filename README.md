# Fusion Label Enhancement for Multi-Label Learning

Code for "Fusion Label Enhancement for Multi-Label Learning" in IJCAI-ECAI 2022.

If you use the code in this repo for your work, please cite the following bib entries:

```
@inproceedings{DBLP:conf/ijcai/0002AXG22,
  author    = {Xingyu Zhao and
               Yuexuan An and
               Ning Xu and
               Xin Geng},
  editor    = {Luc De Raedt},
  title     = {Fusion Label Enhancement for Multi-Label Learning},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI} 2022, Vienna, Austria, 23-29 July
               2022},
  pages     = {3773--3779},
  publisher = {ijcai.org},
  year      = {2022},
  url       = {https://doi.org/10.24963/ijcai.2022/524},
  doi       = {10.24963/ijcai.2022/524},
}
```

## Enviroment

Python >= 3.8.0

Pytorch >= 1.10.0

## Getting started

- Create directory `./datasets/data`
- Change directory to `./datasets/data`
- Download [datasets](https://drive.google.com/drive/folders/1v2XAZX4d7GCDct5VzFPOxtcM4_2z2dVO?usp=sharing)


## Test the pre-trained model

```
python Test_flem.py
```

## Running

```
python run_flem.py
```


## Acknowledgment

Our project references the dataset in the following paper.

Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John M. Winn, Andrew Zisserman:
The Pascal Visual Object Classes Challenge: A Retrospective. International Journal of Computer Vision. 2015, 111(1): 98-136.

