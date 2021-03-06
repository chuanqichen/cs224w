# cs224w

## Setup: 

### 1: Install from requirements.txt (using conda or pip install) 

### 2: Install [pytorch](https://developer.nvidia.com/cuda-zone)  

On MAC or Linux, it's very simple: 
```
pip install torch==1.6.0 
```

On Windows, please follow this link [pytorch](https://developer.nvidia.com/cuda-zone):   
No CUDA
To install PyTorch via pip, and do not have a CUDA-capable system or do not require CUDA, in the above selector, choose OS: Windows, Package: Pip and CUDA: None. Then, run the command that is presented to you.

With CUDA
To install PyTorch via pip, and do have a CUDA-capable system, in the above selector, choose OS: Windows, Package: Pip and the CUDA version suited to your machine. Often, the latest CUDA version is better. Then, run the command that is presented to you.


### 3: Install pytorch-geometric  

Ensure that at least PyTorch 1.4.0 is installed:
```
$ python -c "import torch; print(torch.__version__)"
>>> 1.6.0
```
Find the CUDA version PyTorch was installed with:
```
$ python -c "import torch; print(torch.version.cuda)"
>>> 10.2
```

Install the relevant packages:
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```
where ${CUDA} and ${TORCH} should be replaced by your specific CUDA version (cpu, cu92, cu101, cu102, cu110) and PyTorch version (1.4.0, 1.5.0, 1.6.0, 1.7.0), respectively. For example, for PyTorch 1.7.0/1.7.1 and CUDA 11.0, type:

For example in my case, I have PyTorch 1.6.0 and CUDA 10.2, so I will install with following commands:
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install torch-geometric
```

[pytorch-geometric Reference](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)


### 4: Install ogb 
```
pip install ogb
python -c "import ogb; print(ogb.__version__)"
Note: This should print "1.2.6". Otherwise, please update the version by
pip install -U ogb
```

## Reference: 
1. AGE: baseline implementation of AGE network 
[AGE reference](https://github.com/thunlp/AGE)

2. ddi: how to load ddi dataset for link prediction tasks 
[ddi reference](https://github.com/omri1348/LRGA/tree/master/ogb/examples/linkproppred)

