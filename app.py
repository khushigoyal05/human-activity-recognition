import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import plotly.express as px
import plotly.graph_objects as go
import csv

# --- PAGE CONFIG ---
st.set_page_config(page_title="Kinetics AI", page_icon="📈", layout="wide")

# --- CONSTANTS ---
AVAILABLE_MODELS = ["gru", "lstm", "1d_cnn", "glu", "transformer", "tcn", "bilstm", "itransformer"]
ACTIVITY_MAP = {
    0: "WALKING", 1: "WALKING UPSTAIRS", 2: "WALKING DOWNSTAIRS",
    3: "SITTING", 4: "STANDING", 5: "LAYING"
}

# --- CACHING & DATA LOADING ---
@st.cache_resource
def load_scaler():
    return joblib.load("results/scaler.joblib")

@st.cache_data
def load_test_data():
    base_dir = "UCI HAR Dataset"
    try:
        raw_names = pd.read_csv(f"{base_dir}/features.txt", sep=r"\s+", header=None, names=["idx", "name"])["name"].tolist()
        seen, unames = {}, []
        for n in raw_names:
            if n in seen: 
                seen[n] += 1
                unames.append(f"{n}_{seen[n]}")
            else:
                seen[n] = 0
                unames.append(n)
                
        X = pd.read_csv(f"{base_dir}/test/X_test.txt", sep=r"\s+", header=None, names=unames, dtype=np.float32)
        Y = pd.read_csv(f"{base_dir}/test/y_test.txt", sep=r"\s+", header=None, names=["label"])
        Y["label"] = Y["label"] - 1 # 0-indexed
        return X, Y, unames
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

class GLUBlock(layers.Layer):
    def __init__(self,units,dr=0.2,**kw):
        super().__init__(**kw)
        self.lin=layers.Dense(units); self.gate=layers.Dense(units,activation="sigmoid")
        self.bn=layers.BatchNormalization(); self.dr=layers.Dropout(dr)
        self.units = units
        self.dr_val = dr
    def call(self,x,training=False):
        return self.dr(self.bn(tf.nn.tanh(self.lin(x))*self.gate(x)),training=training)
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "dr": self.dr_val})
        return config

class TBlock(layers.Layer):
    def __init__(self,ed,nh,ff,dr=0.1,**kw):
        super().__init__(**kw)
        self.att=layers.MultiHeadAttention(num_heads=nh,key_dim=ed//nh)
        self.ffn=keras.Sequential([layers.Dense(ff,activation="relu"),layers.Dense(ed)])
        self.ln1=layers.LayerNormalization(epsilon=1e-6)
        self.ln2=layers.LayerNormalization(epsilon=1e-6)
        self.d1=layers.Dropout(dr); self.d2=layers.Dropout(dr)
        self.ed, self.nh, self.ff, self.dr_val = ed, nh, ff, dr
    def call(self,x,training=False):
        a=self.att(x,x); x=self.ln1(x+self.d1(a,training=training))
        return self.ln2(x+self.d2(self.ffn(x),training=training))
    def get_config(self):
        config = super().get_config()
        config.update({"ed": self.ed, "nh": self.nh, "ff": self.ff, "dr": self.dr_val})
        return config

class iBlock(layers.Layer):
    def __init__(self,ed,nh,ff,dr=0.1,**kw):
        super().__init__(**kw)
        self.att=layers.MultiHeadAttention(num_heads=nh,key_dim=ed//nh)
        self.ffn=keras.Sequential([layers.Dense(ff,activation="relu"),layers.Dense(ed)])
        self.ln1=layers.LayerNormalization(epsilon=1e-6)
        self.ln2=layers.LayerNormalization(epsilon=1e-6)
        self.d1=layers.Dropout(dr); self.d2=layers.Dropout(dr)
        self.ed, self.nh, self.ff, self.dr_val = ed, nh, ff, dr
    def call(self,x,training=False):
        a=self.att(x,x); x=self.ln1(x+self.d1(a,training=training))
        return self.ln2(x+self.d2(self.ffn(x),training=training))
    def get_config(self):
        config = super().get_config()
        config.update({"ed": self.ed, "nh": self.nh, "ff": self.ff, "dr": self.dr_val})
        return config

@st.cache_resource
def load_keras_model(model_name):
    # This prevents loading all 8 models into RAM at once.
    # It only loads a model when selected, and caches it for fast reuse.
    custom_objects = {"GLUBlock": GLUBlock, "TBlock": TBlock, "iBlock": iBlock}
    return tf.keras.models.load_model(f"results/model_{model_name}.keras", custom_objects=custom_objects)

@st.cache_data
def load_model_metrics():
    csv_path = "results/model_comparison.csv"
    if not os.path.exists(csv_path):
        return None, None, None
    results = []
    best_f1, best_model = -1, None
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
            if float(row['F1-W(%)']) > best_f1:
                best_f1 = float(row['F1-W(%)'])
                best_model = row['Model']
    return pd.DataFrame(results), best_model, best_f1

# --- PREDICTION LOGIC ---
def predict(features_array, model, scaler):
    data = np.array(features_array).reshape(1, -1)
    scaled = scaler.transform(data)
    seq = scaled.reshape(1, 11, 51)
    preds = model.predict(seq, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    return pred_idx, preds

# --- SIDEBAR NAV ---
st.sidebar.title("📈 Kinetics AI")
st.sidebar.markdown("Human Activity Recognition Dashboard")
page = st.sidebar.radio("Navigation", ["Live Predictor", "Model Performance"])
st.sidebar.divider()
st.sidebar.info("Processing 561-feature vectors using Deep Sequence Modeling.")

scaler = load_scaler()
TEST_X, TEST_Y, feat_names = load_test_data()

# --- PAGE 1: LIVE PREDICTOR ---
if page == "Live Predictor":
    st.title("Interactive Prediction Dashboard")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Model Configuration")
        selected_model = st.selectbox("Select Active Model", AVAILABLE_MODELS, index=0, format_func=lambda x: x.upper())
        
        with st.spinner(f"Loading {selected_model.upper()} into memory..."):
            model = load_keras_model(selected_model)
            
        st.divider()
        mode = st.radio("Input Mode", ["Demo Test Case", "Custom Upload"])
        
        target_features = None
        true_label = None
        sample_idx = None
        
        if mode == "Demo Test Case":
            if st.button("🔄 Fetch Random Test Sample", use_container_width=True):
                st.session_state['demo_idx'] = np.random.randint(0, len(TEST_X))
                
            if 'demo_idx' not in st.session_state:
                st.session_state['demo_idx'] = 0
                
            sample_idx = st.session_state['demo_idx']
            target_features = TEST_X.iloc[sample_idx].tolist()
            true_label = ACTIVITY_MAP[TEST_Y.iloc[sample_idx]["label"]]
            
        else:
            uploaded_file = st.file_uploader("Upload CSV/TXT (561 features)", type=['csv', 'txt'])
            if uploaded_file is not None:
                content = uploaded_file.read().decode('utf-8').strip()
                try:
                    if ',' in content:
                        target_features = [float(x) for x in content.split(',')]
                    else:
                        target_features = [float(x) for x in content.split()]
                    if len(target_features) != 561:
                        st.error(f"Expected 561 features, got {len(target_features)}")
                        target_features = None
                except Exception as e:
                    st.error(f"Failed to parse file: {e}")
    
    with col2:
        if target_features is not None:
            pred_idx, probs = predict(target_features, model, scaler)
            pred_label = ACTIVITY_MAP[pred_idx]
            confidence = probs[pred_idx] * 100
            
            st.subheader("Prediction Results")
            res_col1, res_col2, res_col3 = st.columns(3)
            
            res_col1.metric("Predicted Activity", pred_label)
            res_col2.metric("Confidence", f"{confidence:.2f}%")
            
            if true_label:
                res_col3.metric("Actual Label", true_label)
                if pred_label == true_label:
                    st.success("✅ Prediction Matches Actual Label!")
                else:
                    st.error("❌ Incorrect Prediction")
                    
            # Softmax chart
            prob_df = pd.DataFrame({
                "Activity": list(ACTIVITY_MAP.values()),
                "Probability": probs
            })
            fig = px.bar(prob_df, x="Probability", y="Activity", orientation='h', 
                         title="Class Probabilities (Softmax)",
                         color="Probability", color_continuous_scale="Viridis")
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature viz
            st.subheader("Sample Feature Magnitudes")
            feat_df = pd.DataFrame({"Feature": feat_names, "Value": target_features})
            feat_df['AbsValue'] = feat_df['Value'].abs()
            top_features = feat_df.sort_values(by='AbsValue', ascending=False).head(10)
            
            st.dataframe(
                top_features[['Feature', 'Value']],
                column_config={"Value": st.column_config.NumberColumn(format="%.4f")},
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("👈 Please select a Demo Sample or Upload Custom Data to see predictions.")


# --- PAGE 2: MODEL PERFORMANCE ---
elif page == "Model Performance":
    st.title("Model Analytics & Performance")
    
    df_metrics, best_model, best_f1 = load_model_metrics()
    
    if df_metrics is not None:
        st.success(f"🏆 **Best Performing Model:** {best_model} with an F1-Weighted score of {best_f1}%")
        
        st.subheader("Comprehensive Metrics")
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        # Plot Accuracy vs F1
        df_metrics['Accuracy(%)'] = df_metrics['Accuracy(%)'].astype(float)
        df_metrics['F1-W(%)'] = df_metrics['F1-W(%)'].astype(float)
        
        fig2 = go.Figure(data=[
            go.Bar(name='Accuracy', x=df_metrics['Model'], y=df_metrics['Accuracy(%)'], marker_color='#38bdf8'),
            go.Bar(name='F1 Score', x=df_metrics['Model'], y=df_metrics['F1-W(%)'], marker_color='#34d399')
        ])
        fig2.update_layout(barmode='group', title="Accuracy vs F1-Score by Architecture", yaxis_title="Percentage (%)")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.divider()
        st.subheader("Confusion Matrices")
        if os.path.exists("results/confusion_matrices.png"):
            st.image("results/confusion_matrices.png", use_container_width=True)
        else:
            st.warning("Confusion matrices image not found.")
            
    else:
        st.error("No metrics found. Have you run the training pipeline?")
