<h1 align="center">
  CompGCN
</h1>

<h4 align="center">Composition-Based Multi-Relational Graph Convolutional Networks</h4>

<p align="center">
  <a href="https://iclr.cc/"><img src="http://img.shields.io/badge/ICLR-2020-4b44ce.svg"></a>
  <a href="https://arxiv.org/abs/1911.03082"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
  <a href="https://iclr.cc/virtual/poster_BylA_C4tPr.html"><img src="http://img.shields.io/badge/Video-ICLR-green.svg"></a>
  <a href="https://medium.com/@mgalkin/knowledge-graphs-iclr-2020-f555c8ef10e3"><img src="http://img.shields.io/badge/Blog-Medium-B31B1B.svg"></a>
  <a href="https://github.com/malllabiisc/CompGCN/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  </a>
</p>


<h2 align="center">
  Overview of CompGCN
  <img align="center"  src="./overview.png" alt="...">
</h2>
Given node and relation embeddings, CompGCN performs a composition operation φ(·) over each edge in the neighborhood of a central node (e.g. Christopher Nolan above). The composed embeddings are then convolved with specific filters WO and WI for original and inverse relations respectively. We omit self-loop in the diagram for clarity. The message from all the neighbors are then aggregated to get an updated embedding of the central node. Also, the relation embeddings are transformed using a separate weight matrix. Please refer to the paper for details.

# Dependencies

We evaluated the CompGCN repository with the following configurations.

## pyTorch 1.4, Python 3.x, <=3.7, CPU
- Dependencies can be installed using `pip install -r requirements_old.txt`.
  - Note: If issues arise installing torch/torch_scatter or when executing the code, try to install them manually, with the following command:
  ```commandline
  pip install --no-index torch_scatter==2.0.4 -f https://data.pyg.org/whl/torch-1.4.0+cpu.html
  ```
  ```commandline
  pip install torch==1.4.0 --extra-index-url https://download.pytorch.org/whl/cpu
  ```
## pyTorch 1.4, Python 3.x, <=3.7, CUDA 10.1
- Install all the requirements from `pip install -r requirements_old.txt`.
  - Note: If issues arise installing torch/torch_scatter or when executing the code, try to install them manually, with the following command:
  ```commandline
  pip install --no-index torch_scatter==2.0.4 -f https://data.pyg.org/whl/torch-1.4.0%2Bcu101.html
  ``` 
  For pyTorch 1.4 you need to manually search to correct version [here](https://download.pytorch.org/whl/cu101/torch/), because the official installation methods are broken [[1](https://github.com/pytorch/pytorch/issues/37113)]. For example here we are installing pyTorch 1.4 for the Windows platform on a mobile chip:
  ```commandline
    pip install https://download.pytorch.org/whl/cu101/torch-1.4.0-cp37-cp37m-win_amd64.whl
    ```
## pyTorch 1.13.1, Python 3.9, CUDA 11.7
- Install all the requirements from `pip install -r requirements.txt`.
  - Note: If issues arise installing torch/torch_scatter or when executing the code, try to install them manually, with the following command:
  ```commandline
  pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  ```
  ```commandline
  pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
  ```

# Dataset:

- We use the codex-l, codex-m and codex-s datasets for knowledge graph link prediction. 
- codex-l, codex-m and codex-s are included in the `data` directory. 

# Training model:
- Execute `./preprocess.sh` for extracting the dataset and setting up the folder hierarchy for experiments.

- Commands for reproducing the reported results on link prediction:

  ```shell
  ##### with TransE Score Function
  # CompGCN (Composition: Subtraction)
  python run.py -score_func transe -opn sub -gamma 9 -hid_drop 0.1 -init_dim 200
  
  # CompGCN (Composition: Multiplication)
  python run.py -score_func transe -opn mult -gamma 9 -hid_drop 0.2 -init_dim 200
  
  # CompGCN (Composition: Circular Correlation)
  python run.py -score_func transe -opn corr -gamma 40 -hid_drop 0.1 -init_dim 200
  
  ##### with DistMult Score Function
  # CompGCN (Composition: Subtraction)
  python run.py -score_func distmult -opn sub -gcn_dim 150 -gcn_layer 2 
  
  # CompGCN (Composition: Multiplication)
  python run.py -score_func distmult -opn mult -gcn_dim 150 -gcn_layer 2 
  
  # CompGCN (Composition: Circular Correlation)
  python run.py -score_func distmult -opn corr -gcn_dim 150 -gcn_layer 2 
  
  ##### with ConvE Score Function
  # CompGCN (Composition: Subtraction)
  python run.py -score_func conve -opn sub -ker_sz 5
  
  # CompGCN (Composition: Multiplication)
  python run.py -score_func conve -opn mult
  
  # CompGCN (Composition: Circular Correlation)
  python run.py -score_func conve -opn corr
  
  # CompGCN (Composition: Subtraction) with disabled GNN encoder
  python run.py -score_func conve -opn sub -ker_sz 5 -data codex-s  -k_w 10 -k_h 10 -gcn_dim 100 -disable_gnn_encoder 1

  #####  CompGCN with CTKGC ScoreFunction
  python run.py -score_func ctkgc -init_dim 100 -embed_dim 100 -opn sub -data codex-s

  ##### CompGCN with ConvKB ScoreFunction
  python run.py -score_func convkb -init_dim 100 -embed_dim 100 -num_filt 3 -opn sub
  
  ##### CompGCN with Unstructured Score Function
  python run.py -score_func unstructured -opn sub -gamma 9 -hid_drop 0.1 -init_dim 200
  
  ##### Overall BEST:
  python run.py -name best_model -score_func conve -opn corr 
  
  #### Evaluate an already trained Model:
  python run.py -name XXX -score_func conve -opn corr -evaluate
  ```

  - `-score_func` denotes the link prediction score function 
  - `-opn` is the composition operation used in **CompGCN**. It can take the following values:
    - `sub` for subtraction operation:  Φ(e_s, e_r) = e_s - e_r
    - `mult` for multiplication operation:  Φ(e_s, e_r) = e_s * e_r
    - `corr` for circular-correlation: Φ(e_s, e_r) = e_s ★ e_r
  - `-name` is some name given for the run (used for storing model parameters)
  - `-model` is name of the model `compgcn'.
  - `-gpu` for specifying the GPU to use
  - Rest of the arguments can be listed using `python run.py -h`
### Citation:
Please cite the following paper if you use this code in your work.
```bibtex
@inproceedings{
    vashishth2020compositionbased,
    title={Composition-based Multi-Relational Graph Convolutional Networks},
    author={Shikhar Vashishth and Soumya Sanyal and Vikram Nitin and Partha Talukdar},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=BylA_C4tPr}
}
```
For any clarification, comments, or suggestions please create an issue or contact [Shikhar](http://shikhar-vashishth.github.io).
