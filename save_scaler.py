import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE_DIR  = Path("UCI HAR Dataset")
TRAIN_DIR = BASE_DIR / "train"

raw = pd.read_csv(BASE_DIR/"features.txt",sep=r"\s+",header=None,names=["i","n"])["n"].tolist()
seen,unames = {},[]
for n in raw:
    if n in seen: seen[n]+=1; unames.append(f"{n}_{seen[n]}")
    else:         seen[n]=0;  unames.append(n)

X_tr = pd.read_csv(TRAIN_DIR/"X_train.txt",sep=r"\s+",header=None,names=unames,dtype=np.float32)

scaler = StandardScaler()
scaler.fit(X_tr)

joblib.dump(scaler, "results/scaler.joblib")
print("Saved scaler.joblib to results/")
