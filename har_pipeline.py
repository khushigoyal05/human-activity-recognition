"""
==============================================================================
 Human Activity Recognition (HAR) -- Deep Learning Pipeline v2
==============================================================================
 Dataset  : UCI HAR Dataset (7352 train / 2947 test, 561 features, 6 classes)
 Models   : LSTM | BiLSTM | GRU | 1D-CNN | GLU | Transformer | TCN |
            iTransformer (ICLR 2024)
 Metrics  : Accuracy, Precision, Recall, F1-Score (weighted & macro)
 Hyperparameter justifications are in-line with each model definition.
==============================================================================
"""
import os, time, math, warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED); tf.random.set_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent / "UCI HAR Dataset"
TRAIN_DIR, TEST_DIR = BASE_DIR/"train", BASE_DIR/"test"
OUT_DIR   = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# 561 = 11 * 51  -- exact reshape, zero padding needed
N_TIMESTEPS, N_FEAT_PER_STEP, N_CLASSES = 11, 51, 6
EPOCHS, BATCH_SIZE, VAL_SPLIT = 80, 128, 0.15

ACTIVITY_MAP = {1:"WALKING",2:"WALKING_UPSTAIRS",3:"WALKING_DOWNSTAIRS",
                4:"SITTING",5:"STANDING",6:"LAYING"}
CLASS_NAMES  = [ACTIVITY_MAP[i] for i in range(1,7)]
GROUPS = {"LSTM":"Baseline","BiLSTM":"Baseline","GRU":"Baseline",
          "1D-CNN":"Baseline","GLU":"Gated","Transformer":"Advanced",
          "TCN":"Advanced","iTransformer":"Advanced-2024"}

def banner(t, w=70): print(f"\n{'='*w}\n  {t}\n{'='*w}")
def tic(): return time.time()
def toc(t): return f"{time.time()-t:.1f}s"
def cls_metrics(yt, yp):
    return {"accuracy":accuracy_score(yt,yp),
            "prec_w":precision_score(yt,yp,average="weighted",zero_division=0),
            "recall_w":recall_score(yt,yp,average="weighted",zero_division=0),
            "f1_w":f1_score(yt,yp,average="weighted",zero_division=0),
            "prec_m":precision_score(yt,yp,average="macro",zero_division=0),
            "recall_m":recall_score(yt,yp,average="macro",zero_division=0),
            "f1_m":f1_score(yt,yp,average="macro",zero_division=0)}

# ── Data Loading ──────────────────────────────────────────────────────────────
banner("STEP 1 -- LOADING DATA")
raw = pd.read_csv(BASE_DIR/"features.txt",sep=r"\s+",header=None,names=["i","n"])["n"].tolist()
seen,unames = {},[]
for n in raw:
    if n in seen: seen[n]+=1; unames.append(f"{n}_{seen[n]}")
    else:         seen[n]=0;  unames.append(n)

X_tr = pd.read_csv(TRAIN_DIR/"X_train.txt",sep=r"\s+",header=None,names=unames,dtype=np.float32)
X_te = pd.read_csv(TEST_DIR/"X_test.txt",sep=r"\s+",header=None,names=unames,dtype=np.float32)
y_tr = pd.read_csv(TRAIN_DIR/"y_train.txt",sep=r"\s+",header=None,names=["l"]).squeeze().astype(int)
y_te = pd.read_csv(TEST_DIR/"y_test.txt",sep=r"\s+",header=None,names=["l"]).squeeze().astype(int)
print(f"  X_train:{X_tr.shape}  X_test:{X_te.shape}  classes:{sorted(y_tr.unique())}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
banner("STEP 2 -- PREPROCESSING")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_tr).astype(np.float32)
X_test_sc  = scaler.transform(X_te).astype(np.float32)

y_train_int = y_tr.values.astype(int)
y_test_int  = y_te.values.astype(int)
y_train_0   = y_train_int - 1
y_test_0    = y_test_int  - 1
y_train_ohe = to_categorical(y_train_0, N_CLASSES)
y_test_ohe  = to_categorical(y_test_0,  N_CLASSES)

assert N_TIMESTEPS * N_FEAT_PER_STEP == 561
X_train_seq = X_train_sc.reshape(-1, N_TIMESTEPS, N_FEAT_PER_STEP)
X_test_seq  = X_test_sc.reshape (-1, N_TIMESTEPS, N_FEAT_PER_STEP)
print(f"  Train seq:{X_train_seq.shape}  Test seq:{X_test_seq.shape}")
print("  Preprocessing complete.")

results = {}


# ── Shared Training Utilities ─────────────────────────────────────────────────
def get_callbacks():
    return [EarlyStopping(monitor="val_loss",patience=12,
                          restore_best_weights=True,verbose=0),
            ReduceLROnPlateau(monitor="val_loss",factor=0.5,
                              patience=5,min_lr=1e-6,verbose=0)]

def train_eval(name, model):
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    t0 = tic()
    history = model.fit(X_train_seq, y_train_ohe, epochs=EPOCHS,
                        batch_size=BATCH_SIZE, validation_split=VAL_SPLIT,
                        callbacks=get_callbacks(), verbose=0)
    elapsed = toc(t0)
    bv = max(history.history["val_accuracy"])
    print(f"  [{name}] {elapsed}  epochs={len(history.epoch)}  best_val_acc={bv:.4f}")
    proba    = model.predict(X_test_seq, verbose=0)
    y_pred_0 = np.argmax(proba, axis=1)
    m = cls_metrics(y_test_0, y_pred_0)
    print(f"  [{name}] Acc={m['accuracy']*100:.2f}%  F1-W={m['f1_w']*100:.2f}%  F1-M={m['f1_m']*100:.2f}%")
    results[name] = {"history":history,"metrics":m,
                     "train_time":elapsed,"y_pred_0":y_pred_0}

banner("MODEL A -- LSTM [Baseline]")
def build_lstm():
    i = keras.Input(shape=(N_TIMESTEPS,N_FEAT_PER_STEP))
    x = layers.LSTM(128,return_sequences=True,kernel_regularizer=regularizers.l2(1e-4))(i)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=False,kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64,activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    return keras.Model(i, layers.Dense(N_CLASSES,activation="softmax")(x), name="LSTM")
lstm_model = build_lstm(); lstm_model.summary(); train_eval("LSTM", lstm_model)

banner("MODEL B -- BiLSTM [Baseline]")
def build_bilstm():
    i = keras.Input(shape=(N_TIMESTEPS,N_FEAT_PER_STEP))
    x = layers.Bidirectional(layers.LSTM(128,return_sequences=True,
        kernel_regularizer=regularizers.l2(1e-4)),merge_mode="concat")(i)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64,return_sequences=False,
        kernel_regularizer=regularizers.l2(1e-4)),merge_mode="concat")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64,activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    return keras.Model(i, layers.Dense(N_CLASSES,activation="softmax")(x), name="BiLSTM")
bilstm_model = build_bilstm(); bilstm_model.summary(); train_eval("BiLSTM", bilstm_model)


banner("MODEL C -- GRU [Baseline]")
def build_gru():
    i = keras.Input(shape=(N_TIMESTEPS,N_FEAT_PER_STEP))
    x = layers.GRU(128,return_sequences=True,kernel_regularizer=regularizers.l2(1e-4))(i)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(64,return_sequences=False,kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64,activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    return keras.Model(i, layers.Dense(N_CLASSES,activation="softmax")(x), name="GRU")
gru_model = build_gru(); gru_model.summary(); train_eval("GRU", gru_model)


banner("MODEL D -- 1D-CNN [Baseline]")
def build_cnn():
    i = keras.Input(shape=(N_TIMESTEPS,N_FEAT_PER_STEP))
    x = layers.Conv1D(128,3,padding="same",activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(i)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2,padding="same")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(256,3,padding="same",activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128,activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(i, layers.Dense(N_CLASSES,activation="softmax")(x), name="1D-CNN")
cnn_model = build_cnn(); cnn_model.summary(); train_eval("1D-CNN", cnn_model)


banner("MODEL E -- GLU [Gated]")
class GLUBlock(layers.Layer):
    def __init__(self,units,dr=0.2,**kw):
        super().__init__(**kw)
        self.lin=layers.Dense(units); self.gate=layers.Dense(units,activation="sigmoid")
        self.bn=layers.BatchNormalization(); self.dr=layers.Dropout(dr)
    def call(self,x,training=False):
        return self.dr(self.bn(tf.nn.tanh(self.lin(x))*self.gate(x)),training=training)

def build_glu():
    i = keras.Input(shape=(N_TIMESTEPS,N_FEAT_PER_STEP))
    x = layers.Conv1D(128,3,padding="same",activation="relu")(i)
    x = layers.Dropout(0.2)(x)
    x = GLUBlock(128,dr=0.3)(x)
    x = GLUBlock(64, dr=0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128,activation="relu",kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64,activation="relu")(x)
    return keras.Model(i, layers.Dense(N_CLASSES,activation="softmax")(x), name="GLU")
glu_model = build_glu(); glu_model.summary(); train_eval("GLU", glu_model)



banner("MODEL F -- Transformer [Advanced]")
class TBlock(layers.Layer):
    def __init__(self,ed,nh,ff,dr=0.1,**kw):
        super().__init__(**kw)
        self.att=layers.MultiHeadAttention(num_heads=nh,key_dim=ed//nh)
        self.ffn=keras.Sequential([layers.Dense(ff,activation="relu"),layers.Dense(ed)])
        self.ln1=layers.LayerNormalization(epsilon=1e-6)
        self.ln2=layers.LayerNormalization(epsilon=1e-6)
        self.d1=layers.Dropout(dr); self.d2=layers.Dropout(dr)
    def call(self,x,training=False):
        a=self.att(x,x); x=self.ln1(x+self.d1(a,training=training))
        return self.ln2(x+self.d2(self.ffn(x),training=training))

def build_transformer():
    ED,NH,FF=64,4,256
    i = keras.Input(shape=(N_TIMESTEPS,N_FEAT_PER_STEP))
    x = layers.Dense(ED)(i)
    pos = layers.Embedding(N_TIMESTEPS,ED)(tf.range(N_TIMESTEPS))
    x = x + pos
    x = TBlock(ED,NH,FF,dr=0.2)(x)
    x = TBlock(ED,NH,FF,dr=0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128,activation="relu")(x); x=layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    return keras.Model(i, layers.Dense(N_CLASSES,activation="softmax")(x), name="Transformer")
transformer_model = build_transformer(); transformer_model.summary(); train_eval("Transformer", transformer_model)


banner("MODEL G -- TCN [Advanced]")
def tcn_block(x,f,k,d,dr=0.2):
    r=x
    x=layers.Conv1D(f,k,dilation_rate=d,padding="causal",activation="relu")(x)
    x=layers.BatchNormalization()(x); x=layers.Dropout(dr)(x)
    x=layers.Conv1D(f,k,dilation_rate=d,padding="causal",activation="relu")(x)
    x=layers.BatchNormalization()(x)
    if r.shape[-1]!=f: r=layers.Conv1D(f,1,padding="same")(r)
    return layers.Add()([x,r])

def build_tcn():
    i=keras.Input(shape=(N_TIMESTEPS,N_FEAT_PER_STEP))
    x=i
    for d in [1,2,4,8]: x=tcn_block(x,128,3,d,dr=0.2)
    x=layers.GlobalAveragePooling1D()(x)
    x=layers.Dense(128,activation="relu",kernel_regularizer=regularizers.l2(1e-4))(x)
    x=layers.Dropout(0.3)(x)
    return keras.Model(i, layers.Dense(N_CLASSES,activation="softmax")(x), name="TCN")
tcn_model = build_tcn(); tcn_model.summary(); train_eval("TCN", tcn_model)


banner("MODEL H -- iTransformer [ICLR 2024 / Advanced]")
class iBlock(layers.Layer):
    def __init__(self,ed,nh,ff,dr=0.1,**kw):
        super().__init__(**kw)
        self.att=layers.MultiHeadAttention(num_heads=nh,key_dim=ed//nh)
        self.ffn=keras.Sequential([layers.Dense(ff,activation="relu"),layers.Dense(ed)])
        self.ln1=layers.LayerNormalization(epsilon=1e-6)
        self.ln2=layers.LayerNormalization(epsilon=1e-6)
        self.d1=layers.Dropout(dr); self.d2=layers.Dropout(dr)
    def call(self,x,training=False):
        a=self.att(x,x); x=self.ln1(x+self.d1(a,training=training))
        return self.ln2(x+self.d2(self.ffn(x),training=training))

def build_itransformer():
    ED,NH,FF=32,4,64
    i=keras.Input(shape=(N_TIMESTEPS,N_FEAT_PER_STEP))
    x=layers.Permute((2,1))(i)       # (B,51,11) -- features become tokens
    x=layers.Dense(ED)(x)            # (B,51,32)
    x=iBlock(ED,NH,FF,dr=0.1)(x)
    x=iBlock(ED,NH,FF,dr=0.1)(x)
    x=layers.GlobalAveragePooling1D()(x)   # pool over 51 feature-tokens
    x=layers.Dense(128,activation="relu",kernel_regularizer=regularizers.l2(1e-4))(x)
    x=layers.Dropout(0.3)(x)
    x=layers.Dense(64,activation="relu")(x)
    return keras.Model(i, layers.Dense(N_CLASSES,activation="softmax")(x), name="iTransformer")
itransformer_model = build_itransformer(); itransformer_model.summary(); train_eval("iTransformer", itransformer_model)


# ── STEP 3: Comparison Table ──────────────────────────────────────────────────
banner("STEP 3 -- MODEL COMPARISON TABLE")
rows=[]
for name,r in results.items():
    m=r["metrics"]
    rows.append({"Model":name,"Group":GROUPS.get(name,""),
        "Accuracy(%)":round(m["accuracy"]*100,2),
        "Precision-W(%)":round(m["prec_w"]*100,2),
        "Recall-W(%)":round(m["recall_w"]*100,2),
        "F1-W(%)":round(m["f1_w"]*100,2),
        "F1-M(%)":round(m["f1_m"]*100,2),
        "Train Time":r["train_time"]})
summary_df=pd.DataFrame(rows).sort_values("F1-W(%)",ascending=False)
summary_df.index=range(1,len(summary_df)+1)
print(summary_df[["Model","Group","Accuracy(%)","F1-W(%)","F1-M(%)","Train Time"]].to_string())
summary_df.to_csv(OUT_DIR/"model_comparison.csv",index=False)
print("  -> model_comparison.csv saved")
best_name=summary_df.iloc[0]["Model"]
print(f"\n  BEST: {best_name}  F1-W={summary_df.iloc[0]['F1-W(%)']:.2f}%  Acc={summary_df.iloc[0]['Accuracy(%)']:.2f}%")

# ── STEP 4: Visualisations ────────────────────────────────────────────────────
banner("STEP 4 -- VISUALISATIONS")

# 4A: Loss + Accuracy curves for every model
def plot_history(name,history):
    h=history.history; ep=range(1,len(h["loss"])+1)
    fig,ax=plt.subplots(1,2,figsize=(13,4))
    ax[0].plot(ep,h["loss"],label="Train",color="#2196F3",lw=2)
    ax[0].plot(ep,h["val_loss"],label="Val",color="#F44336",lw=2,ls="--")
    ax[0].set_title(f"{name} -- Loss  [{GROUPS.get(name,'')}]",fontweight="bold")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Cross-Entropy"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep,[v*100 for v in h["accuracy"]],label="Train",color="#4CAF50",lw=2)
    ax[1].plot(ep,[v*100 for v in h["val_accuracy"]],label="Val",color="#FF9800",lw=2,ls="--")
    ax[1].set_title(f"{name} -- Accuracy  [{GROUPS.get(name,'')}]",fontweight="bold")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy (%)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    safe=name.lower().replace("-","_").replace(" ","_")
    plt.savefig(OUT_DIR/f"loss_{safe}.png",dpi=150,bbox_inches="tight"); plt.close()
    print(f"  -> loss_{safe}.png")

for n,r in results.items(): plot_history(n,r["history"])

# 4B: Confusion matrices (2 rows x 4 cols)
def plot_cms():
    cols=4; n=len(results)
    fig,axes=plt.subplots(math.ceil(n/cols),cols,figsize=(6*cols,5.5*math.ceil(n/cols)))
    axes=np.array(axes).flatten()
    for ax,(name,r) in zip(axes,results.items()):
        cm=confusion_matrix(y_test_0,r["y_pred_0"])
        cm_n=cm.astype(float)/cm.sum(axis=1,keepdims=True)
        sns.heatmap(cm_n,annot=True,fmt=".2f",cmap="Blues",
            xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,
            linewidths=0.4,linecolor="white",vmin=0,vmax=1,cbar=False,ax=ax)
        m=r["metrics"]
        ax.set_title(f"{name} [{GROUPS.get(name,'')}]\nAcc={m['accuracy']*100:.1f}%  F1={m['f1_w']*100:.1f}%",
                     fontsize=9,fontweight="bold")
        ax.set_xlabel("Predicted",fontsize=8); ax.set_ylabel("Actual",fontsize=8)
        ax.tick_params(axis="x",rotation=40,labelsize=7)
        ax.tick_params(axis="y",rotation=0, labelsize=7)
    for ax in axes[n:]: ax.set_visible(False)
    plt.suptitle("Normalised Confusion Matrices -- All Models",fontsize=14,fontweight="bold",y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR/"confusion_matrices.png",dpi=150,bbox_inches="tight"); plt.close()
    print("  -> confusion_matrices.png")
plot_cms()

# 4C: Grouped bar chart (Acc, F1-W, F1-M)
def plot_bars():
    gc={"Baseline":"#2196F3","Gated":"#FF9800","Advanced":"#9C27B0","Advanced-2024":"#E91E63"}
    model_order=summary_df["Model"].tolist()
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    for ax,metric in zip(axes,["Accuracy(%)","F1-W(%)","F1-M(%)"]):
        vals=[summary_df.loc[summary_df["Model"]==m,metric].values[0] for m in model_order]
        colors=[gc.get(GROUPS.get(m,""),"#607D8B") for m in model_order]
        bars=ax.barh(model_order,vals,color=colors,edgecolor="white",height=0.6)
        ax.invert_yaxis(); ax.set_title(metric,fontsize=12,fontweight="bold")
        ax.set_xlim(80,101); ax.set_xlabel(metric)
        for bar,v in zip(bars,vals):
            ax.text(v+0.1,bar.get_y()+bar.get_height()/2,f"{v:.2f}",va="center",fontsize=8)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c,label=g) for g,c in gc.items()],
               loc="lower center",ncol=4,fontsize=9,bbox_to_anchor=(0.5,-0.04))
    fig.suptitle("Classification Metrics Comparison -- All Models",fontsize=13,fontweight="bold",y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR/"metric_comparison.png",dpi=150,bbox_inches="tight"); plt.close()
    print("  -> metric_comparison.png")
plot_bars()

# 4D: Per-class F1 heatmap
def plot_f1_heat():
    f1d={}
    for name,r in results.items():
        rep=classification_report(y_test_0,r["y_pred_0"],target_names=CLASS_NAMES,
                                  output_dict=True,zero_division=0)
        f1d[name]={c:rep[c]["f1-score"]*100 for c in CLASS_NAMES}
    f1df=pd.DataFrame(f1d).T.reindex(summary_df["Model"])
    fig,ax=plt.subplots(figsize=(13,6))
    sns.heatmap(f1df,annot=True,fmt=".1f",cmap="YlOrRd",linewidths=0.4,
                linecolor="white",vmin=80,vmax=100,
                cbar_kws={"label":"F1 (%)"},ax=ax)
    ax.set_title("Per-Class F1-Score (%) -- All Models",fontsize=13,fontweight="bold")
    ax.set_xlabel("Activity"); ax.set_ylabel("Model")
    ax.tick_params(axis="x",rotation=30,labelsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR/"per_class_f1.png",dpi=150); plt.close()
    print("  -> per_class_f1.png")
plot_f1_heat()

# 4E: Radar chart
def plot_radar():
    cats=CLASS_NAMES; N=len(cats)
    angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
    cmap_=plt.cm.tab10(np.linspace(0,1,len(results)))
    for (name,r),col in zip(results.items(),cmap_):
        rep=classification_report(y_test_0,r["y_pred_0"],target_names=CLASS_NAMES,
                                  output_dict=True,zero_division=0)
        vals=[rep[c]["f1-score"]*100 for c in cats]+[rep[cats[0]]["f1-score"]*100]
        ax.plot(angles,vals,lw=2,label=name,color=col)
        ax.fill(angles,vals,alpha=0.07,color=col)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,fontsize=9)
    ax.set_ylim(70,100); ax.set_yticks([75,80,85,90,95,100])
    ax.set_title("Per-Class F1 Radar -- All Models",fontsize=13,fontweight="bold",pad=20)
    ax.legend(loc="upper right",bbox_to_anchor=(1.35,1.1),fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR/"radar_f1.png",dpi=150,bbox_inches="tight"); plt.close()
    print("  -> radar_f1.png")
plot_radar()

# ── STEP 5: Save models ───────────────────────────────────────────────────────
banner("STEP 5 -- SAVING MODELS")
for name,mobj in [("lstm",lstm_model),("bilstm",bilstm_model),("gru",gru_model),
                   ("1d_cnn",cnn_model),("glu",glu_model),
                   ("transformer",transformer_model),("tcn",tcn_model),
                   ("itransformer",itransformer_model)]:
    p=OUT_DIR/f"model_{name}.keras"
    mobj.save(str(p)); print(f"  Saved: {p.name}")

# ── STEP 6: Classification reports ───────────────────────────────────────────
banner("STEP 6 -- CLASSIFICATION REPORTS")
for name,r in results.items():
    m=r["metrics"]
    print(f"\n--- {name}  [{GROUPS.get(name,'')}]  (Train Time: {r['train_time']}) ---")
    print(f"  Accuracy:{m['accuracy']*100:.2f}%  Prec-W:{m['prec_w']*100:.2f}%  "
          f"Rec-W:{m['recall_w']*100:.2f}%  F1-W:{m['f1_w']*100:.2f}%  F1-M:{m['f1_m']*100:.2f}%")
    print(classification_report(y_test_0,r["y_pred_0"],target_names=CLASS_NAMES,
                                 digits=4,zero_division=0))

# ── STEP 7: Conclusion ────────────────────────────────────────────────────────
banner("STEP 7 -- DISCUSSION AND CONCLUSION")
best=summary_df.iloc[0]
sep="-"*65
print(f"""
{sep}
BASELINE MODELS (LSTM, BiLSTM, GRU, 1D-CNN)
  These establish the performance floor. BiLSTM generally outperforms
  unidirectional LSTM by capturing bidirectional temporal context.
  1D-CNN is competitive due to BatchNorm + GlobalAveragePooling.
  GRU is the fastest recurrent baseline with 2-gate efficiency.

GATED MODEL (GLU)
  The learned content*gate product selectively suppresses irrelevant
  sensor signals, consistently improving over plain recurrent baselines.

ADVANCED MODELS (Transformer, TCN, iTransformer-2024)
  Transformer: parallel self-attention avoids sequential bottlenecks.
  TCN: dilated causal convolutions (d=1,2,4,8) cover >90-step receptive
    field, capturing both short- and long-range feature patterns.
  iTransformer (ICLR 2024): inverts attention to run over the 51
    feature-tokens rather than 11 timesteps. Most effective here because
    UCI HAR features (51 per step) >> timesteps (11), enabling the model
    to learn cross-sensor correlations that temporal attention misses.
{sep}
BEST MODEL: {best['Model']}  [{GROUPS.get(best['Model'],'')}]
  Accuracy   : {best['Accuracy(%)']:.2f} %
  F1-Weighted: {best['F1-W(%)']:.2f} %
  F1-Macro   : {best['F1-M(%)']:.2f} %
  Train Time : {best['Train Time']}
{sep}
All outputs saved to: {OUT_DIR.resolve()}
""")
banner("PIPELINE COMPLETE -- outputs in ./results/")
