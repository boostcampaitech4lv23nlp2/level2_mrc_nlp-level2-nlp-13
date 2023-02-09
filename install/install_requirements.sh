#!/bin/bash
### install requirements for pstage3 baseline
# pip requirements
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install tabulate==0.9.0
pip install ray==2.2.0
pip install wandb==0.13.10
pip install pytz==2022.7.1
pip install omegaconf==2.3.0
pip install datasets==2.9.0
pip install transformers==4.26.0
pip install rank-bm25==0.2.2
pip install scikit-learn==1.2.1
pip install tqdm
pip install pandas

# faiss install (if you want to)
pip install faiss-gpu==1.7.2
