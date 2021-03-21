# cs224w

## Model 



### Reference performance for OGB:

| Model              |Test Hits@20 (%) | Validation Hits@20(%)  | Parameters    | Hardware |
| ------------------ |--------------   | --------------- | -------------- |----------|
|MAD Learning        |	67.81 ± 2.94	 |70.10 ± 0.82	  |1,228,897	|Geforce GTX 1080 Ti (11GB GPU)|
|LRGA + GCN 	       | 62.30 ± 9.12	   | 66.75 ± 0.58	|1,576,081	|Tesla P100 (16GB GPU)|
|LRGA + GraphSage + Node2Vec	|61.23 ± 13.62	|68.27 ± 0.96		|Tesla V100 (32GB)|
|Deeper GCN Augment	|31.52 ± 8.27	| 56.92 ± 1.33	| 319,555	|Tesla V100 (32GB)|
|LRGA + GCN Aug	|55.08 ± 13.44		| 66.03 ± 2.48		| 1,576,081		| Tesla V100 (32GB)|
|LRGA + GCN Aug	|66.55 ± 8.70		| 69.85 ± 0.60		| 3,807,121		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec + RNN	|70.66 ± 5.88		| 69.58 ± 0.90		| 3,807,121		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec:SkipConnect		| 65.33 ± 8.21		| 71.19 ± 0.57		| 3,126,985		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec:SkipConnect		| 65.91 ± 11.22		| 71.66 ± 1.38		| 3,126,985		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec:SkipConnect		| 69.61 ± 04.39		| 70.51 ± 1.00		| 3,126,985		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec:SkipConnect		| 68.74 ± 6.47		| 70.46 ± 1.42		| 3,126,985		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec:SkipConnect		| 23.34 ± 24.69		| 68.85 ± 0.65		| 4,749,913		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 56.22 ± 10.15		| 65.52 ± 0.87		| 1,576,081		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 60.80 ± 10.45		| 67.28 ± 0.74		| 3,807,121		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 73.41 ± 7.15		| 70.13 ± 0.50		| 3,807,121		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 65.91 ± 10.11		| 70.98 ±0.65		| 3,807,121		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 67.96 ± 10.41		| 71.33 ± 0.52		| 3,807,121		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 62.69 ± 5.65		| 67.28 ± 2.10		| 3,807,121		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 45.18 ± 14.67		| 62.29 ± 6.09		| 4,749,913		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 72.91 ± 5.07		| 70.62 ± 1.01		| 3,807,121		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 50.33 ± 13.37		| 67.52 ± 0.75		| 3,126,985		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 50.33 ± 13.37		| 67.52 ± 0.75		| 3,126,985		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 64.48 ± 14.18		| 65.32 ± 5.56		| 4,749,913		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 73.41 ± 7.15		| 70.13 ± 0.50		| 3,807,121		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 73.51 ± 8.69		| 70.55 ± 0.31		| 5,168,401		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 74.24 ± 14.18		| 71.54 ± 0.61		| 6,693,521		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 73.13 ± 13.79		| 71.73 ± 0.65		| 7,100,401		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 77.94 ± 9.20		| 71.81 ± 0.73		| 7,517,521		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 74.17 ± 13.97		| 72.03 ± 0.59		| 8,382,481		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 75.88 ± 10.28		| 71.65 ± 0.53		| 9,288,401		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 73.85 ± 8.71		| 72.25 ± 0.47		| 10,235,281		| Tesla V100 (32GB)|
|LRGA + GCN Aug + Node2Vec		| 66.07 ± 20.60		| 70.97 ± 2.76		| 12,251,921		| Tesla V100 (32GB)|




## Setup: 

### 0: Install cuda driver like [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=7&target_type=runfilelocal) 

```
Install instruction in Linux: 
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run

Update environment variable (you can add it into  ~/.bashrc): 
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64"
PATH="$PATH:/usr/local/cuda-10.2/bin"
```

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
1. AGE: [baseline implementation of AGE network](https://github.com/thunlp/AGE)

2. ddi: [how to load ddi dataset for link prediction tasks](https://github.com/omri1348/LRGA/tree/master/ogb/examples/linkproppred)

