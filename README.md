# crypto_proj
- models.py contains the model implementations
- data_encryption.py contains everything related to the raw data loading, chunking, and the encryption
- main.ipynb is the main script to run the training experiments

# 1. To replicate the python environment
Run the following lines in the terminal (conda is required): 
```
conda create -n crypto python=3.12
conda activate crypto
conda install -c numpy pandas scipy scikit-learn pycryptodome jupyter
pip install torch torchvision torchaudio tenseal matplotlib tqdm
```