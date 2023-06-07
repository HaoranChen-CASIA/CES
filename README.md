# 
This repository contains the official implementation of the paper 
"Quantitative Registration Quality Assessment of Serial Section Electron Microscopy Images With Learned Features".


![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg?style=plastic) ![PyTorch 1.13.0](https://img.shields.io/badge/pytorch-1.13.0-green.svg?style=plastic) 

## Using the Code
### Requirements
This code has been developed under Python 3.9, PyTorch 1.13.0, and Ubuntu 16.04.

In addition to the above libraries, the python environment can be set as follows:

```shell
conda create -n CES
conda activate CES
pip3 install opencv-python torch
pip3 install scipy pickle scikit-learn scikit-image matplotlib
```

To simply compute CES metric value for a matching-pair, please use the following code.
### Compute CES for a matching-pair
```Register
python compute_ces.py --reference chunk_ref.png --moving chunk_mov.png
```

To re-implement the experiments in the paper, it is recommended to download the dataset used in this paper.
### Datasets in the paper

[CSTCloud]()

### Sensitivity of CES to Section Thickness
```Register
python test_6_thickness.py --root_dir './data/mito_dxy6_z1_c' --seq_length 1000
```

### Sensitivity of CES to Horizontal translation-distance
```Register
python test_4_category.py --root_dir './data/EXP2_FlyEM_OCG/Dataset_z32nm' --seq_length 640
```

### Performance on RQA-Classification tasks
```Register
python classifier_3category.py --root_dir './data/EXP2_FlyEM_OCG/Dataset_z32nm' --seq_length 640
```

To cite this paper, please use
### Citation
```
coming soon
```
