import torch
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from training import HybridCNNDeepFM, IEEEFraudDataset

DEVICE= "cuda" if torch.cuda.is_available() else "cpu"
checkpoint= 'D:/project/modules/module_output/best_fraud_model.pth'
data_train= 'D:/project/data/merge/train_processed.csv'

def extract_embeddings(model, loader):
    model.eval()
    X_emb, y_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            emb = model.extract_features(x)   # CNN embedding
            X_emb.append(emb.cpu().numpy())
            y_all.append(y.numpy())

    return np.vstack(X_emb), np.concatenate(y_all)

def check_fraud_rate_and_auc(model, loader):
    model.eval()
