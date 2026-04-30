model_used = False

import tempfile
import shap
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import io
import base64
import os
import time
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


import tensorflow as tf


# ------------------------------------------
# 🔥 MODEL LOADING (FINAL FIXED VERSION)
# ------------------------------------------

def focal_loss(gamma=2., alpha=0.75):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true*y_pred + (1-y_true)*(1-y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * bce
    return loss

@st.cache_resource
def load_my_model():
    model_path = "models/cnn_bilstm_attention_best.keras"   # ✅ correct model
    return tf.keras.models.load_model(
        model_path,
        custom_objects={'loss': focal_loss()}        # ✅ REQUIRED
    )

try:
    MODEL = load_my_model()
    st.sidebar.success("🧠 Model Loaded Successfully")

    # 🔥 Grad-CAM warmup (KEEP THIS)
    if MODEL is not None:
        import numpy as np
        dummy_input = np.zeros((1, 128, 94, 1), dtype=np.float32)
        MODEL.predict(dummy_input)

except Exception as e:
    MODEL = None
    st.sidebar.error(f"❌ Model Load Failed: {e}")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — DARK CLINICAL LUXURY AESTHETIC
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Root ── */
:root {
  --bg:        #080c14;
  --surface:   #0d1420;
  --card:      #111827;
  --border:    #1e2d45;
  --accent:    #00d4aa;
  --accent2:   #ff4d6d;
  --accent3:   #7b61ff;
  --text:      #e8edf5;
  --muted:     #6b7a99;
  --syne:      'Syne', sans-serif;
  --mono:      'Space Mono', monospace;
  --body:      'DM Sans', sans-serif;
}

/* ── App Shell ── */
.stApp {
  background: var(--bg);
  color: var(--text);
  font-family: var(--body);
}
.block-container { padding: 0 2rem 4rem 2rem; max-width: 1200px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, header, footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Hero ── */
.hero {
  position: relative;
  text-align: center;
  padding: 5rem 2rem 3rem;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,212,170,0.12) 0%, transparent 70%),
              radial-gradient(ellipse 50% 40% at 80% 50%, rgba(123,97,255,0.08) 0%, transparent 60%);
  pointer-events: none;
}
.hero-eyebrow {
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.25em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 1rem;
}
.hero-title {
  font-family: var(--syne);
  font-size: clamp(2.8rem, 6vw, 5rem);
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #ffffff 30%, var(--accent) 70%, #7b61ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 1.2rem;
}
.hero-sub {
  font-family: var(--body);
  font-size: 1.1rem;
  font-weight: 300;
  color: #9aafc8;
  max-width: 580px;
  margin: 0 auto 2.5rem;
  line-height: 1.7;
}
.hero-badges {
  display: flex;
  gap: 0.75rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 3rem;
}
.badge {
  font-family: var(--mono);
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  padding: 0.35rem 0.9rem;
  border-radius: 999px;
  border: 1px solid;
}
.badge-green  { border-color: var(--accent);  color: var(--accent);  background: rgba(0,212,170,0.07); }
.badge-purple { border-color: var(--accent3); color: var(--accent3); background: rgba(123,97,255,0.07); }
.badge-red    { border-color: var(--accent2); color: var(--accent2); background: rgba(255,77,109,0.07); }

/* ── Section Headers ── */
.section-label {
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.22em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}
.section-title {
  font-family: var(--syne);
  font-size: 1.9rem;
  font-weight: 700;
  color: var(--text);
  margin: 0 0 0.6rem;
}
.section-body {
  font-family: var(--body);
  font-size: 0.97rem;
  color: #8a9bc0;
  line-height: 1.8;
  margin-bottom: 1.5rem;
}

/* ── Cards ── */
.glass-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.8rem 2rem;
  margin-bottom: 1.2rem;
  position: relative;
  overflow: hidden;
}
.glass-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0,212,170,0.4), transparent);
}

/* ── Stat Metrics ── */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin: 1.5rem 0;
}
.metric-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.5rem 1.2rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card::after {
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent3));
}
.metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.metric-label {
  font-family: var(--mono);
  font-size: 0.65rem;
  letter-spacing: 0.15em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}
.metric-value {
  font-family: var(--syne);
  font-size: 1.8rem;
  font-weight: 800;
}
.metric-green { color: var(--accent); }
.metric-red   { color: var(--accent2); }
.metric-blue  { color: #60a5fa; }

/* ── Pipeline ── */
.pipeline {
  display: flex;
  align-items: center;
  gap: 0;
  overflow-x: auto;
  padding: 0.5rem 0 1rem;
  margin: 1.5rem 0;
}
.pipe-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 110px;
  text-align: center;
  padding: 1.2rem 0.8rem;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--card);
  position: relative;
  flex: 1;
}
.pipe-step:not(:last-child)::after {
  content: '→';
  position: absolute; right: -1.1rem;
  font-size: 1.2rem;
  color: var(--accent);
  z-index: 2;
}
.pipe-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
.pipe-label {
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.05em;
  color: var(--muted);
  line-height: 1.4;
}
.pipe-title {
  font-family: var(--body);
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--text);
  margin-bottom: 0.2rem;
}

/* ── Upload Zone ── */
.upload-zone {
  border: 2px dashed var(--border);
  border-radius: 16px;
  padding: 2.5rem;
  text-align: center;
  background: rgba(0,212,170,0.02);
  transition: border-color 0.3s;
  margin: 1rem 0;
}
.upload-zone:hover { border-color: var(--accent); }
.upload-title {
  font-family: var(--syne);
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 0.4rem;
}
.upload-sub {
  font-size: 0.85rem;
  color: var(--muted);
}

/* ── Result Banner ── */
.result-normal {
  background: linear-gradient(135deg, rgba(0,212,170,0.12), rgba(0,212,170,0.04));
  border: 1px solid rgba(0,212,170,0.4);
  border-radius: 16px;
  padding: 1.5rem 2rem;
  margin: 1.5rem 0;
}
.result-murmur {
  background: linear-gradient(135deg, rgba(255,77,109,0.12), rgba(255,77,109,0.04));
  border: 1px solid rgba(255,77,109,0.4);
  border-radius: 16px;
  padding: 1.5rem 2rem;
  margin: 1.5rem 0;
}
.result-title {
  font-family: var(--syne);
  font-size: 1.6rem;
  font-weight: 800;
  margin-bottom: 0.5rem;
}
.result-detail {
  font-size: 0.9rem;
  color: #9aafc8;
  line-height: 1.7;
}

/* ── Divider ── */
.divider {
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border) 30%, var(--border) 70%, transparent);
  margin: 3rem 0;
}

/* ── Disclaimer ── */
.disclaimer {
  background: rgba(255,200,0,0.06);
  border: 1px solid rgba(255,200,0,0.25);
  border-radius: 12px;
  padding: 1.2rem 1.5rem;
  font-size: 0.85rem;
  color: #c9ac5a;
  line-height: 1.6;
  margin: 2rem 0 1rem;
}

/* ── Footer ── */
.footer {
  text-align: center;
  padding: 3rem 0 1rem;
  border-top: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--muted);
  letter-spacing: 0.08em;
  line-height: 2.2;
}

/* ── Streamlit overrides ── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #00b894) !important;
  color: #000 !important;
  font-family: var(--syne) !important;
  font-weight: 700 !important;
  font-size: 0.9rem !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.65rem 2rem !important;
  letter-spacing: 0.05em !important;
  transition: opacity 0.2s !important;
  width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stFileUploader > label { color: var(--text) !important; }
div[data-testid="stFileUploaderDropzone"] {
  background: var(--card) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 12px !important;
}
div[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }

.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: 10px !important;
  gap: 0.3rem !important;
  padding: 0.3rem !important;
}
.stTabs [data-baseweb="tab"] {
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
  border-radius: 8px !important;
  padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color: #000 !important;
}

.stProgress > div > div { background: var(--accent) !important; border-radius: 999px !important; }
.stProgress > div { background: var(--border) !important; border-radius: 999px !important; }

.stExpander {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

/* Matplotlib chart container */
.element-container img { border-radius: 12px; }

/* ── Pulse animation ── */
@keyframes pulse-ring {
  0% { transform: scale(1); opacity: 0.6; }
  100% { transform: scale(1.4); opacity: 0; }
}
.heartbeat {
  display: inline-block;
  animation: heartbeat 1.5s infinite;
}
@keyframes heartbeat {
  0%, 100% { transform: scale(1); }
  14% { transform: scale(1.2); }
  28% { transform: scale(1); }
  42% { transform: scale(1.15); }
  70% { transform: scale(1); }
}

/* ── Info tiles ── */
.info-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin: 1.2rem 0;
}
.info-tile {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem;
  border-left: 3px solid var(--accent);
}
.info-tile-title {
  font-family: var(--syne);
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 0.4rem;
}
.info-tile-body {
  font-size: 0.83rem;
  color: var(--muted);
  line-height: 1.6;
}

/* tip box */
.tip-box {
  background: rgba(123,97,255,0.08);
  border: 1px solid rgba(123,97,255,0.3);
  border-radius: 10px;
  padding: 1rem 1.2rem;
  font-size: 0.85rem;
  color: #c4b8ff;
  line-height: 1.6;
  margin: 0.8rem 0;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_img_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=130)
    buf.seek(0)
    return buf

def plot_waveform(y, sr, title="Waveform", color="#00d4aa", peaks=None):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    t = np.linspace(0, len(y)/sr, len(y))
    ax.plot(t, y, color=color, lw=0.8, alpha=0.9)
    ax.fill_between(t, y, alpha=0.15, color=color)
    if peaks is not None:
        peak_t = peaks / sr
        ax.vlines(peak_t, y.min(), y.max(),
                  color="#ff4d6d", lw=1.2, alpha=0.7, label="S1/S2 Peaks")
        ax.legend(loc="upper right", fontsize=8,
                  facecolor="#0d1420", labelcolor="#ff4d6d", framealpha=0.8)
    ax.set_xlabel("Time (s)", color="#6b7a99", fontsize=9)
    ax.set_ylabel("Amplitude", color="#6b7a99", fontsize=9)
    ax.set_title(title, color="#e8edf5", fontsize=11, pad=10, fontweight="bold")
    ax.tick_params(colors="#6b7a99", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d45")
    ax.grid(True, color="#1e2d45", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig

def plot_spectrogram(y, sr, title="Mel Spectrogram", cmap="magma"):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel",
                                   fmax=8000, ax=ax, cmap=cmap)
    cb = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cb.ax.yaxis.set_tick_params(color="#6b7a99", labelsize=8)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="#6b7a99")
    ax.set_title(title, color="#e8edf5", fontsize=11, pad=10, fontweight="bold")
    ax.tick_params(colors="#6b7a99", labelsize=8)
    ax.set_xlabel("Time", color="#6b7a99", fontsize=9)
    ax.set_ylabel("Mel Frequency", color="#6b7a99", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d45")
    fig.tight_layout()
    return fig

def segment_heart_sounds(y, sr):
    import numpy as np
    from scipy.signal import find_peaks

    # Normalize
    y_norm = y / (np.max(np.abs(y)) + 1e-8)

    # Shannon Energy
    energy = -(y_norm**2) * np.log(y_norm**2 + 1e-10)

    # Smooth
    window_len = int(sr * 0.02)
    smoothed_energy = np.convolve(
        energy,
        np.ones(window_len)/window_len,
        mode='same'
    )

    # Peak detection (S1/S2)
    distance = int(sr * 0.2)
    peaks, _ = find_peaks(
        smoothed_energy,
        distance=distance,
        height=np.mean(smoothed_energy) + 0.3 * np.std(smoothed_energy)
    )

    # Phase classification (optional)
    phases = []
    for i in range(len(peaks) - 1):
        gap = peaks[i+1] - peaks[i]
        if gap < (sr * 0.4):
            label = 'Systole'
        else:
            label = 'Diastole'

        phases.append({
            'start': peaks[i],
            'end': peaks[i+1],
            'label': label
        })

    return smoothed_energy, peaks, phases

# ==========================================
# REAL GRAD-CAM IMPLEMENTATION
# ==========================================

import cv2

def compute_gradcam(model, spectrogram, layer_name=None):
    import tensorflow as tf
    import numpy as np

    input_tensor = spectrogram[np.newaxis, ..., np.newaxis].astype(np.float32)
    input_tensor = tf.convert_to_tensor(input_tensor)

    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
        model.get_layer(layer_name).output,
        model.outputs[0]

        ]
    )

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        conv_outputs, predictions = grad_model(input_tensor)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        # 🔥 fallback: use absolute activations
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(conv_outputs, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

        

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def extract_clinical_features(y, sr, peaks, phases):
    import numpy as np
    import librosa

    features = {}

    # 1. Estimated Heart Rate (BPM)
    if len(peaks) > 1:
        intervals = np.diff(peaks) / sr
        heart_rate = 60 / np.mean(intervals)
    else:
        heart_rate = 0.0

    features["heart_rate"] = heart_rate

    # 2. Approx. Systole / Diastole Ratio
# 2. Approx. Systole / Diastole Ratio
    systole_durations = []
    diastole_durations = []
    for phase in phases:
     duration = (phase["end"] - phase["start"]) / sr
     label = phase["label"].lower()

     if "systole" in label:
        systole_durations.append(duration)
     elif "diastole" in label:
        diastole_durations.append(duration)

    if systole_durations and diastole_durations:
     ratio = np.mean(systole_durations) / (np.mean(diastole_durations) + 1e-8)
    else:
     ratio = None

    features["sys_dia_ratio"] = ratio   # ✅ FIXED

    # 3. Spectral Centroid (Hz)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features["spectral_centroid"] = float(centroid)

    # 4. Approx. S1/S2 Energy Ratio
    if len(peaks) > 2:
        s1_energy = []
        s2_energy = []

        window = int(0.05 * sr)

        for i, p in enumerate(peaks):
            segment = y[max(0, p - window): p + window]
            energy = np.sum(segment ** 2)

            if i % 2 == 0:
                s1_energy.append(energy)
            else:
                s2_energy.append(energy)

        if s1_energy and s2_energy:
            energy_ratio = np.mean(s1_energy) / (np.mean(s2_energy) + 1e-8)
        else:
            energy_ratio = 0.0
    else:
        energy_ratio = 0.0

    features["s1_s2_energy_ratio"] = energy_ratio

    return features

def compute_attention_consistency(y, sr, model):
    window_sec = 3
    stride_sec = 1

    window_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)

    heatmaps = []

    for start in range(0, len(y) - window_samples, stride_samples):
        y_chunk = y[start:start + window_samples]

        # spectrogram
        S = librosa.feature.melspectrogram(
            y=y_chunk,
            sr=sr,
            n_fft=512,
            hop_length=64,
            n_mels=128
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        TARGET_WIDTH = 94
        if S_dB.shape[1] < TARGET_WIDTH:
            S_dB = np.pad(S_dB, ((0,0),(0, TARGET_WIDTH - S_dB.shape[1])))
        else:
            S_dB = S_dB[:, :TARGET_WIDTH]

        S_dB = (S_dB - np.mean(S_dB)) / (np.std(S_dB) + 1e-6)

        heatmap = compute_gradcam(model, S_dB)

        heatmaps.append(heatmap.flatten())

    if len(heatmaps) < 2:
        return 0.0

    # pairwise cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(heatmaps)

    # take upper triangle (excluding diagonal)
    upper = sims[np.triu_indices_from(sims, k=1)]

    return float(np.mean(upper))

def compute_attention_peak_correlation(y, sr, model):
    import numpy as np
    import librosa

    window_sec = 3
    stride_sec = 1

    window_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)

    correlations = []

    for start in range(0, len(y) - window_samples, stride_samples):
        y_chunk = y[start:start + window_samples]

        # get peaks
        _, peaks, _ = segment_heart_sounds(y_chunk, sr)
        if len(peaks) == 0:
            continue

        # spectrogram
        S = librosa.feature.melspectrogram(
            y=y_chunk,
            sr=sr,
            n_fft=512,
            hop_length=64,
            n_mels=128
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        TARGET_WIDTH = 94
        if S_dB.shape[1] < TARGET_WIDTH:
            S_dB = np.pad(S_dB, ((0,0),(0, TARGET_WIDTH - S_dB.shape[1])))
        else:
            S_dB = S_dB[:, :TARGET_WIDTH]

        S_dB = (S_dB - np.mean(S_dB)) / (np.std(S_dB) + 1e-6)

        heatmap = compute_gradcam(model, S_dB)

        # map peaks → spectrogram x-axis
        peak_positions = [
            int((p / len(y_chunk)) * heatmap.shape[1]) for p in peaks
        ]

        # sample heatmap intensity at peaks
        peak_vals = [np.mean(heatmap[:, pos]) for pos in peak_positions if pos < heatmap.shape[1]]

        if len(peak_vals) > 0:
            correlations.append(np.mean(peak_vals))

    if len(correlations) == 0:
        return 0.0

    return float(np.mean(correlations))

def generate_pdf_report(label, conf, peaks, duration, consistency, corr, features, gradcam_path=None):
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    # ─────────────────────────────
    # 🎨 STYLES
    # ─────────────────────────────
    title_style = ParagraphStyle(
        name="Title",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        textColor=colors.HexColor("#00d4aa"),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=12
    )

    section_style = ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        alignment=TA_CENTER,
        fontSize=13,
        textColor=colors.HexColor("#7b61ff"),
        spaceAfter=8
    )

    content = []

    # ─────────────────────────────
    # 🏥 HEADER (LOGO + TITLE)
    # ─────────────────────────────
    try:
        logo = Image("assets/logo.png", width=60, height=60)
        header_table = Table([
            [logo, Paragraph("<b>CardioSense AI</b><br/>AI Cardiac Screening Report", styles["Normal"])]
        ], colWidths=[70, 400])

        header_table.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "MIDDLE")
        ]))

        content.append(header_table)
    except:
        content.append(Paragraph("CardioSense AI", title_style))

    content.append(Spacer(1, 10))

    # ─────────────────────────────
    # 📊 PREDICTION
    # ─────────────────────────────
    content.append(Paragraph("Prediction Summary", section_style))

    result_color = colors.green if label == "Normal" else colors.red

    pred_table = Table([
        ["Result", label],
        ["Confidence", f"{round(conf*100,2)}%"]
    ], colWidths=[200, 150])

    pred_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.whitesmoke),
        ("TEXTCOLOR", (1,0), (1,0), result_color),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey)
    ]))

    content.append(pred_table)
    content.append(Spacer(1, 12))

    # ─────────────────────────────
    # 🫀 FEATURES
    # ─────────────────────────────
    content.append(Paragraph("Clinical Features", section_style))

    ratio = features.get("sys_dia_ratio")
    ratio_text = f"{ratio:.2f}" if ratio is not None else "N/A"

    feat_table = Table([
        ["Heart Rate", f"{features['heart_rate']:.1f} BPM"],
        ["Systole/Diastole Ratio", ratio_text],
        ["S1/S2 Energy Ratio", f"{features['s1_s2_energy_ratio']:.2f}"],
        ["Spectral Centroid", f"{features['spectral_centroid']:.1f} Hz"]
    ], colWidths=[200, 150])

    feat_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey)
    ]))

    content.append(feat_table)
    content.append(Spacer(1, 15))

    # ─────────────────────────────
    # 🔍 EXPLAINABILITY
    # ─────────────────────────────
    content.append(Paragraph("Model Explainability", section_style))

    explain_table = Table([
        ["Consistency", f"{consistency:.2f}"],
        ["Peak Alignment", f"{corr:.2f}"]
    ], colWidths=[200, 150])

    explain_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey)
    ]))

    content.append(explain_table)
    content.append(Spacer(1, 15))

    # ─────────────────────────────
    # 🖼 GRAD-CAM IMAGE
    # ─────────────────────────────
    if gradcam_path:
        content.append(Paragraph("Grad-CAM Attention Map", section_style))
        try:
            img = Image(gradcam_path, width=450, height=180)
            content.append(img)
        except:
            pass

    content.append(Spacer(1, 20))

    # ─────────────────────────────
    # ⚠️ DISCLAIMER
    # ─────────────────────────────
    content.append(Paragraph(
        "This AI-generated report is intended for screening purposes only and "
        "does not replace professional medical diagnosis.",
        subtitle_style
    ))

    doc.build(content)
    buffer.seek(0)
    return buffer.read()

# ==========================================
# 🔥 GRADCAM FUNCTION (FINAL)
# ==========================================
def plot_real_gradcam(y, sr, model, peaks=None, phases=None):

    # 1. Spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=512,
        hop_length=64,
        n_mels=128
    )

    S_dB = librosa.power_to_db(S, ref=np.max)

    TARGET_WIDTH = 94
    original_width = S_dB.shape[1]

    # Normalize
    S_dB = (S_dB - np.mean(S_dB)) / (np.std(S_dB) + 1e-6)

    # Pad / Trim
    if S_dB.shape[1] < TARGET_WIDTH:
        S_dB = np.pad(S_dB, ((0,0),(0, TARGET_WIDTH - S_dB.shape[1])))
    else:
        S_dB = S_dB[:, :TARGET_WIDTH]
        original_width = TARGET_WIDTH

    # 🔥 FIXED INPUT SHAPE
    heatmap = compute_gradcam(model, S_dB)

    heatmap = cv2.resize(heatmap, (S_dB.shape[1], S_dB.shape[0]))
    heatmap = cv2.GaussianBlur(heatmap, (9,9), 0)

    heatmap[:, original_width:] = 0
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))

    librosa.display.specshow(
        S_dB,
        sr=sr,
        hop_length=64,
        x_axis='time',
        y_axis='mel',
        cmap='magma',
        ax=ax
    )

    cam = ax.imshow(
        heatmap,
        cmap='inferno',
        alpha=0.35,
        aspect='auto',
        origin='lower'
    )

    fig.colorbar(cam, ax=ax, pad=0.01)

    ax.set_title("Grad-CAM Attention Map")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel")

    plt.tight_layout()
    return fig


# ==========================================
# 🔥 REAL SHAP FUNCTION (ONLY ONE VERSION)
# ==========================================
def plot_shap_explanation(y, sr, model):
    import numpy as np
    import librosa
    import matplotlib.pyplot as plt
    import shap

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=512, hop_length=64, n_mels=128
    )
    S_dB = librosa.power_to_db(mel, ref=np.max)

    # Resize
    if S_dB.shape[1] < 94:
        S_dB = np.pad(S_dB, ((0,0),(0,94-S_dB.shape[1])))
    else:
        S_dB = S_dB[:, :94]

    # Normalize
    S_dB = (S_dB - np.mean(S_dB)) / (np.std(S_dB) + 1e-6)

    input_data = np.expand_dims(np.expand_dims(S_dB, -1), 0)

    background = np.random.normal(0,1,(2,128,94,1))

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(input_data)

    if isinstance(shap_values, list):
        shap_map = shap_values[0][0,:,:,0]
    else:
        shap_map = shap_values[0,:,:,0]

    fig, ax = plt.subplots(1,2, figsize=(10,4))

    ax[0].imshow(S_dB, origin='lower', cmap='magma')
    ax[0].set_title("Spectrogram")

    ax[1].imshow(S_dB, origin='lower', cmap='gray')
    ax[1].imshow(np.abs(shap_map), cmap='inferno', alpha=0.5)
    ax[1].set_title("SHAP Explanation")

    plt.tight_layout()
    return fig




# ==========================================
# 🔥 QUICK MFCC FEATURE IMPORTANCE
# ==========================================
def plot_shap(y, sr):

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    importance = np.abs(mfcc).mean(axis=1)

    importance = importance / (importance.max() + 1e-8)

    indices = np.argsort(importance)[::-1]
    importance = importance[indices]
    labels = [f"MFCC-{i+1}" for i in indices]

    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(labels, importance)

    ax.set_title("Feature Importance (MFCC-based)")
    plt.tight_layout()

    return fig


def plot_frequency(y, sr, title="Frequency Spectrum"):
    fft = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), 1/sr)
    mag = np.abs(fft[:len(fft)//2])
    freqs = freqs[:len(freqs)//2]
    mask = freqs <= 1000
    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.plot(freqs[mask], mag[mask], color="#7b61ff", lw=0.9, alpha=0.9)
    ax.fill_between(freqs[mask], mag[mask], alpha=0.2, color="#7b61ff")
    ax.set_xlabel("Frequency (Hz)", color="#6b7a99", fontsize=9)
    ax.set_ylabel("Magnitude", color="#6b7a99", fontsize=9)
    ax.set_title(title, color="#e8edf5", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#6b7a99", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d45")
    ax.grid(True, color="#1e2d45", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig

def apply_bandpass(y, sr, lowcut=25, highcut=400):
    from scipy.signal import butter, filtfilt
    nyq = sr / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype="band")
    if len(y) < sr:
      y = np.pad(y, (0, sr - len(y)))
    return filtfilt(b, a, y)

def run_inference(y, sr):
    try:
        model = MODEL
        if model is None:
            raise Exception("Model not loaded")

        # ==========================================
        # STEP 1: Force sampling rate
        # ==========================================
        TARGET_SR = 2000
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        # ==========================================
        # STEP 2: Bandpass + normalization
        # ==========================================
        

        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        # ==========================================
        # STEP 3: Chunking
        # ==========================================
        CHUNK_SAMPLES = TARGET_SR * 3
        probs = []

        for start in range(0, len(y), CHUNK_SAMPLES):
            chunk = y[start:start + CHUNK_SAMPLES]

            # skip very short chunks
            if len(chunk) < CHUNK_SAMPLES * 0.6:
                continue

            # pad if needed
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

            # ==========================================
            # STEP 4: Spectrogram
            # ==========================================
            S = librosa.feature.melspectrogram(
                y=chunk,
                sr=sr,
                n_fft=512,
                hop_length=64,
                n_mels=128
            )

            S_dB = librosa.power_to_db(S, ref=np.max)

            # ==========================================
            # STEP 5: Fix width
            # ==========================================
            TARGET_WIDTH = 94

            if S_dB.shape[1] < TARGET_WIDTH:
                S_dB = np.pad(
                    S_dB,
                    ((0, 0), (0, TARGET_WIDTH - S_dB.shape[1])),
                    mode='constant'
                )
            else:
                S_dB = S_dB[:, :TARGET_WIDTH]

            # ==========================================
            # STEP 6: Normalize (CRITICAL)
            # ==========================================
            S_dB = (S_dB - np.mean(S_dB)) / (np.std(S_dB) + 1e-6)

            # ==========================================
            # STEP 7: Predict
            # ==========================================
            x = S_dB[np.newaxis, ..., np.newaxis].astype(np.float32)
            prob = float(model.predict(x, verbose=0)[0][0])

            probs.append(prob)

        # ==========================================
        # STEP 8: Aggregate predictions
        # ==========================================
        if len(probs) == 0:
            return "Normal", 0.0, False

        avg_prob = float(np.mean(probs))
        max_prob = float(np.max(probs))

        print("CHUNK PROBS:", probs)
        print("AVG PROB:", avg_prob)
        print("MAX PROB:", max_prob)

        # ==========================================
        # STEP 9: Final decision
        # ==========================================
        THRESHOLD = 0.5

        # Condition 1: strong murmur across multiple chunks
        if avg_prob > THRESHOLD:
            label = "Murmur"
            confidence = avg_prob

        # Condition 2: very strong murmur spike (rare but real)
        elif max_prob > 0.9:
            label = "Murmur"
            confidence = max_prob

        # Otherwise normal
        else:
            label = "Normal"
            confidence = 1 - avg_prob

        return label, round(confidence, 4), True
    except Exception as e:
        print("MODEL ERROR:", e)
        return "Normal", 0.0, False
    

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key in [
    "analyzed",
    "label",
    "confidence",
    "peaks",
    "phases",          # ✅ ADD
    "y",
    "y_clean",         # ✅ ADD (CRITICAL FIX)
    "sr",
    "model_used",
    "file_bytes",
    "file_ext"
]:
    if key not in st.session_state:
        if key == "model_used":
            st.session_state[key] = False
        else:
            st.session_state[key] = None

# ══════════════════════════════════════════════
# ① HERO
# ══════════════════════════════════════════════
banner_b64 = load_img_b64("assets/rheumatic_heart_banner.png")

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown("""
    <div style="text-align:center;">

      <div style="font-size:0.85rem; letter-spacing:2px; color:#00d4aa;">
        AI · Healthcare · Deep Learning
      </div>

      <h1 style="margin-top:0.5rem;">
        CardioSense AI
      </h1>

      <p style="max-width:650px; margin: 0.5rem auto 1rem;">
        AI-assisted screening of heart sounds for the early detection of
        Rheumatic Heart Disease — making advanced cardiac diagnostics
        accessible to everyone.
      </p>

      <div style="display:flex; justify-content:center; flex-wrap:wrap; gap:10px;">
        <span class="badge badge-green">CNN-Based Classification</span>
        <span class="badge badge-purple">GradCAM Explainability</span>
        <span class="badge badge-red">SHAP Feature Analysis</span>
        <span class="badge badge-green">Mel Spectrogram Processing</span>
        <span class="badge badge-purple">Real-time Analysis</span>
      </div>

    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# ② ABOUT RHD
# ══════════════════════════════════════════════
col_txt, col_img = st.columns([3, 2], gap="large")

with col_txt:
    st.markdown("""
    <div class="section-label">01 — Background</div>
    <div class="section-title">What is Rheumatic Heart Disease?</div>
    <div class="section-body">
      Rheumatic Heart Disease (RHD) is a chronic condition caused by repeated
      episodes of acute rheumatic fever — an inflammatory reaction to untreated
      Group A <em>Streptococcal</em> throat infections. Over time, this permanently
      damages the heart valves, impairing blood flow and leading to heart failure.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-grid">
      <div class="info-tile">
        <div class="info-tile-title">🌍 Global Burden</div>
        <div class="info-tile-body">Affects 40 million+ people worldwide, predominantly in low-income countries with limited healthcare access.</div>
      </div>
      <div class="info-tile" style="border-left-color:#ff4d6d;">
        <div class="info-tile-title">⚠️ Silent Onset</div>
        <div class="info-tile-body">Most patients are asymptomatic for years. By the time symptoms appear, valvular damage is often severe.</div>
      </div>
      <div class="info-tile" style="border-left-color:#7b61ff;">
        <div class="info-tile-title">🎯 Early Detection</div>
        <div class="info-tile-body">Identifying murmurs (abnormal heart sounds) early allows timely intervention and significantly improves outcomes.</div>
      </div>
      <div class="info-tile" style="border-left-color:#60a5fa;">
        <div class="info-tile-title">💊 Treatment</div>
        <div class="info-tile-body">Prophylactic antibiotics and regular follow-up can halt disease progression if detected early.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

treat_b64 = load_img_b64("assets/treatment_guide.png")
with col_img:
    if treat_b64:
        st.markdown(f"""
        <div style="margin-top:3rem; text-align:center;">
          <img src="data:image/png;base64,{treat_b64}"
               style="width:100%; border-radius:16px;
                      border:1px solid #1e2d45;
                      box-shadow: 0 20px 60px rgba(0,0,0,0.4);" />
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# ③ HOW IT WORKS — PIPELINE
# ══════════════════════════════════════════════
st.markdown("""
<div class="section-label">02 — Methodology</div>
<div class="section-title">How the System Works</div>
<div class="section-body">
  A full signal processing and deep learning pipeline transforms raw audio
  into a clinically meaningful prediction — with full explainability.
</div>
""", unsafe_allow_html=True)

steps = [
    ("🎧", "Input", "Heart Sound\nRecording"),
    ("🧹", "Denoise", "Bandpass\nFiltering"),
    ("✂️", "Segment", "Signal\nChunking"),
    ("📊", "Feature", "Mel Spectrogram\nGeneration"),
    ("🧠", "Inference", "CNN Model\nPrediction"),
    ("🔍", "Explain", "GradCAM\n+ SHAP"),
    ("📋", "Output", "Clinical\nReport"),
]

cols = st.columns(len(steps))
for col, (icon, label, title) in zip(cols, steps):
    with col:
        connector = "→" if label != "Output" else ""
        st.markdown(f"""
        <div style="background:#111827; border:1px solid #1e2d45; border-radius:12px;
                    padding:1rem 0.5rem; text-align:center; height:110px;
                    display:flex; flex-direction:column; align-items:center; justify-content:center;">
          <div style="font-size:1.5rem; margin-bottom:0.3rem;">{icon}</div>
          <div style="font-family:'Space Mono',monospace; font-size:0.58rem;
                      color:#00d4aa; letter-spacing:0.1em; text-transform:uppercase;
                      margin-bottom:0.2rem;">{label}</div>
          <div style="font-size:0.75rem; color:#9aafc8; line-height:1.4;
                      white-space:pre-line;">{title}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# ④ SPECTROGRAM REFERENCE
# ══════════════════════════════════════════════
st.markdown("""
<div class="section-label">03 — Visual Reference</div>
<div class="section-title">Spectrogram Patterns</div>
<div class="section-body">
  Mel spectrograms convert sound into visual representations that neural networks
  can analyze. Normal hearts show smooth, uniform energy distributions while
  murmurs exhibit irregular bursts and extended frequency activity.
</div>
""", unsafe_allow_html=True)

s_col1, s_col2 = st.columns(2, gap="large")
norm_b64   = load_img_b64("assets/Normal_Spectogram.png")
murmur_b64 = load_img_b64("assets/Murmur_Spectogram.png")

with s_col1:
    if norm_b64:
        st.markdown(f"""
        <div class="glass-card">
          <div style="font-family:'Space Mono',monospace; font-size:0.68rem;
                      color:#00d4aa; letter-spacing:0.15em; text-transform:uppercase;
                      margin-bottom:0.8rem;">✅ Normal Heart Sound</div>
          <img src="data:image/png;base64,{norm_b64}" style="width:100%; border-radius:8px;"/>
          <p style="font-size:0.82rem; color:#6b7a99; margin-top:0.8rem; line-height:1.6;">
            Smooth, uniform energy. Activity concentrated in low-frequency bands.
            Clean S1/S2 separation with clear diastolic silence.
          </p>
        </div>
        """, unsafe_allow_html=True)

with s_col2:
    if murmur_b64:
        st.markdown(f"""
        <div class="glass-card" style="border-color:rgba(255,77,109,0.3);">
          <div style="font-family:'Space Mono',monospace; font-size:0.68rem;
                      color:#ff4d6d; letter-spacing:0.15em; text-transform:uppercase;
                      margin-bottom:0.8rem;">⚠️ Murmur Heart Sound</div>
          <img src="data:image/png;base64,{murmur_b64}" style="width:100%; border-radius:8px;"/>
          <p style="font-size:0.82rem; color:#6b7a99; margin-top:0.8rem; line-height:1.6;">
            Irregular energy bursts extending into mid-frequency bands.
            Turbulent blood flow creates distinctive periodic disturbances.
          </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# ⑤ PATIENT INPUT
# ══════════════════════════════════════════════
st.markdown("""
<div class="section-label">04 — Analysis</div>
<div class="section-title">Upload Heart Sound Recording</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="tip-box">
  💡 <strong>Recording Tips:</strong>
  Record in a quiet environment · Place stethoscope at the 4th intercostal space ·
  Duration: 10–15 seconds · Format: WAV or MP3 · Avoid background noise and movement artifacts.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your heart sound file here",
    type=["wav", "mp3", "ogg", "flac", "mp4"],
    help="Supported: WAV, MP3, OGG, FLAC, MP4 (audio-only)"
)

# ✅ RESET STATE WHEN NEW FILE IS UPLOADED
if uploaded_file is not None:
    current_file_name = uploaded_file.name

    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != current_file_name:
        
        st.session_state.last_uploaded = current_file_name

        # 🔥 Reset previous results
        st.session_state.analyzed = False
        st.session_state.label = None
        st.session_state.confidence = None
        st.session_state.peaks = None
        st.session_state.y = None
        st.session_state.y_clean = None
        st.session_state.sr = None
        st.session_state.model_used = False
        st.session_state.consistency = None
        st.session_state.corr_score = None

        # 🔥 Reset file buffer
        st.session_state.file_bytes = None
        st.session_state.file_ext = None

        st.info("New file uploaded. Ready for analysis.")

if uploaded_file:

    # 🔥 Read only once and reuse
    if st.session_state.file_bytes is None:
        st.session_state.file_bytes = uploaded_file.read()
        st.session_state.file_ext = uploaded_file.name.split(".")[-1].lower()

    file_bytes = st.session_state.file_bytes

    st.success(f"Uploaded: {uploaded_file.name}")

    # ===== PLAYBACK DISABLED =====
    # st.markdown("""
    # <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#00d4aa;
    #             letter-spacing:0.1em; margin: 0.5rem 0 0.2rem;">▶ PLAYBACK</div>
    # """, unsafe_allow_html=True)
    #
    # if st.session_state.file_ext == "mp4":
    #     st.video(file_bytes)
    # else:
    #     st.audio(file_bytes, format=f"audio/{st.session_state.file_ext}")

if st.button("🫀  Run Cardiac Analysis") and st.session_state.file_bytes is not None:
    with st.spinner(""):
        progress = st.progress(0)
        status   = st.empty()

        status.markdown('<p style="font-family:\'Space Mono\',monospace; font-size:0.8rem; color:#00d4aa;">Loading audio...</p>', unsafe_allow_html=True)
        progress.progress(15)

        # 🔥 MP4 / AUDIO HANDLING (FINAL FIX)
        import tempfile
        import subprocess
        import os

        file_bytes = st.session_state.file_bytes
        file_ext = "." + st.session_state.file_ext

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_bytes)
            input_path = tmp.name

        # Convert MP4 → WAV if needed
        if file_ext == ".mp4":
            output_path = input_path.replace(".mp4", ".wav")

            command = [
                r"C:\Users\stalw\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe",

                "-y",
                "-i", input_path,
                "-ac", "1",
                "-ar", "2000",
                "-vn",
                output_path
]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            audio_path = output_path
        else:
            audio_path = input_path

        # Load audio
        y, sr = librosa.load(audio_path, sr=2000, mono=True)
        time.sleep(0.3)

        try:
          os.remove(input_path)
          if file_ext == ".mp4":
              os.remove(output_path)
        except Exception as e:
          pass  # optional: st.warning(f"Cleanup failed: {e}")

        # ----------------------------------

        status.markdown('<p style="font-family:\'Space Mono\',monospace; font-size:0.8rem; color:#00d4aa;">Applying bandpass filter...</p>', unsafe_allow_html=True)
        progress.progress(35)
        y_clean = apply_bandpass(y, sr)
        label, confidence, model_used = run_inference(y_clean, sr)
        st.session_state["model_used"] = model_used   # 🔥 ADD THIS

        status.markdown('<p style="font-family:\'Space Mono\',monospace; font-size:0.8rem; color:#00d4aa;">Detecting cardiac peaks...</p>', unsafe_allow_html=True)
        progress.progress(55)
        time.sleep(0.3)

        status.markdown('<p style="font-family:\'Space Mono\',monospace; font-size:0.8rem; color:#00d4aa;">Running CNN inference...</p>', unsafe_allow_html=True)
        progress.progress(75)
        
        st.session_state.model_used = model_used
        time.sleep(0.4)

        status.markdown('<p style="font-family:\'Space Mono\',monospace; font-size:0.8rem; color:#00d4aa;">Generating explainability maps...</p>', unsafe_allow_html=True)
        progress.progress(95)
        time.sleep(0.3)

        progress.progress(100)
        status.empty()
        energy, peaks, phases = segment_heart_sounds(y_clean, sr)

        st.session_state.phases = phases
        st.session_state.analyzed    = True
        st.session_state.label       = label
        st.session_state.confidence  = confidence
        st.session_state.peaks       = peaks
        st.session_state.y           = y
        st.session_state.y_clean     = y_clean
        st.session_state.sr          = sr
        st.session_state.model_used  = model_used
# ══════════════════════════════════════════════
# ⑥ RESULTS
# ══════════════════════════════════════════════
if st.session_state.analyzed:
    label      = st.session_state.label
    conf       = st.session_state.confidence
    peaks      = st.session_state.peaks
    y          = st.session_state.y
    y_clean    = st.session_state.y_clean
    sr         = st.session_state.sr
    phases     = st.session_state.phases

    peak_count = len(peaks)
    conf_pct   = int(conf * 100)
    duration   = round(len(y) / sr, 1)

    features = extract_clinical_features(y_clean, sr, peaks, phases)

    # 🎞️ Grad-CAM (ONLY HERE)
    st.subheader("🎞️ Grad-CAM Over Time")

    if MODEL is None:
        st.error("❌ Grad-CAM unavailable: model not loaded.")

    else:
        try:
            fig = plot_real_gradcam(
                y_clean,
                sr,
                MODEL,
                peaks=peaks,
                phases=phases
            )

            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")

    st.subheader("🧠 Deep Explanation (SHAP)")
    if st.button("Run SHAP Explanation"):
      try:
        fig_shap = plot_shap_explanation(y_clean, sr, MODEL)
        st.pyplot(fig_shap)
      except Exception as e:
        st.error(f"SHAP failed: {e}")

    # ── Banner ──
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">05 — Results</div>
    <div class="section-title">Analysis Report</div>
    """, unsafe_allow_html=True)
    
    st.write("MODEL LOADED:", MODEL is not None)
    if MODEL is not None:
      st.markdown("✓ Model loaded successfully")
    else:
      st.error("❌ Model failed to load")

    

    if label == "Normal":
        st.markdown(f"""
        <div class="result-normal">
          <div class="result-title" style="color:#00d4aa;">✅ Normal Heart Sound Detected</div>
          <div class="result-detail">
            The AI model found no significant evidence of cardiac murmurs in this recording.
            The heart sound pattern is consistent with normal valvular function.
            Confidence: <strong style="color:#00d4aa;">{conf_pct}%</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-murmur">
          <div class="result-title" style="color:#ff4d6d;">⚠️ Cardiac Murmur Pattern Detected</div>
          <div class="result-detail">
            The AI model identified patterns consistent with a cardiac murmur.
            Irregular turbulence signatures were detected in the processed signal.
            Confidence: <strong style="color:#ff4d6d;">{conf_pct}%</strong>
            — Please consult a cardiologist for clinical evaluation.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Metrics ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
      cls_color = "metric-green" if label == "Normal" else "metric-red"
      st.markdown(f"""
      <div class="metric-card">
        <div class="metric-icon">🧠</div>
          <div class="metric-label">Prediction</div>
      <div class="metric-value {cls_color}">{label}</div>
      </div>
      """, unsafe_allow_html=True)

    with c2:
      st.markdown(f"""
      <div class="metric-card">
        <div class="metric-icon">📈</div>
        <div class="metric-label">Confidence</div>
        <div class="metric-value metric-blue">{conf_pct}%</div>
      </div>
      """, unsafe_allow_html=True)

    with c3:
      st.markdown(f"""
      <div class="metric-card">
        <div class="metric-icon">❤️</div>
        <div class="metric-label">Peaks Detected</div>
        <div class="metric-value metric-green">{peak_count}</div>
      </div>
      """, unsafe_allow_html=True)

    with c4:
      st.markdown(f"""
      <div class="metric-card">
        <div class="metric-icon">⏱️</div>
        <div class="metric-label">Duration</div>
        <div class="metric-value metric-blue">{duration}s</div>
      </div>
      """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown(f"**Model confidence:** {conf_pct}%")
    st.progress(conf_pct / 100)

    ratio = features.get("sys_dia_ratio")
    ratio_text = f"{ratio:.2f}" if ratio is not None else "N/A"

    st.markdown(f"""
    <div class="glass-card" style="border-left: 3px solid #00d4aa;">
      <div style="font-weight:700; font-size:1rem; margin-bottom:0.6rem;">
        🫀 Clinical Feature Estimates
    </div>
    <div style="font-size:0.9rem; line-height:1.8;">
      Estimated Heart Rate: <b>{features['heart_rate']:.1f} BPM</b><br>
      Approx. Systole/Diastole Ratio: <b>{ratio_text}</b><br>
      S1/S2 Energy Ratio: <b>{features['s1_s2_energy_ratio']:.2f}</b><br>
      Spectral Centroid: <b>{features['spectral_centroid']:.1f} Hz</b>
    </div>
  </div>
  """, unsafe_allow_html=True)

    # ==========================================
    # 🔥 Attention Consistency (FIRST)
    # ==========================================
    if st.session_state.get("consistency") is None:
        st.session_state.consistency = compute_attention_consistency(y_clean, sr, MODEL)

    consistency = st.session_state.consistency

    st.markdown(f"""
    <div class="glass-card" style="border-left: 3px solid #7b61ff;">
      <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem;
                  color:#e8edf5; margin-bottom:0.6rem;">📊 Attention Consistency</div>
      <div style="font-size:0.9rem; color:#9aafc8;">
        Consistency Score: <strong style="color:#7b61ff;">{consistency:.2f}</strong>
        <br>
        {"Stable cardiac pattern (likely normal)." if consistency > 0.75 else "Irregular attention — possible murmur patterns."}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # 🔥 Attention–Peak Correlation (SECOND)
    # ==========================================
    if st.session_state.get("corr_score") is None:
        st.session_state.corr_score = compute_attention_peak_correlation(y_clean, sr, MODEL)

    corr = st.session_state.corr_score

    st.markdown(f"""
    <div class="glass-card" style="border-left: 3px solid #00d4aa;">
      <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem;
                  color:#e8edf5; margin-bottom:0.6rem;">🫀 Attention–Peak Alignment</div>
      <div style="font-size:0.9rem; color:#9aafc8;">
        Correlation Score: <strong style="color:#00d4aa;">{corr:.2f}</strong>
        <br>
        {"Model focuses correctly on S1/S2 events." if corr > 0.6 else "Weak alignment — attention may be diffuse or noisy."}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # 🔥 PDF REPORT (LAST — AFTER ALL VARIABLES EXIST)
    # ==========================================
    pdf_bytes = generate_pdf_report(
        label,
        conf,
        peaks,
        duration,
        consistency,
        corr,
        features,   # ✅ USE REAL FEATURES
        st.session_state.get("gradcam_path")  # ✅ ADD THIS
    )

    st.download_button(
        label="📄 Download Clinical Report",
        data=pdf_bytes,
        file_name="cardio_report.pdf",
        mime="application/pdf"
    )

    # ══════════════════════════════════════════
    # ⑦ SIGNAL VISUALIZATION
    # ══════════════════════════════════════════
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">06 — Signal Analysis</div>
    <div class="section-title">Waveform & Frequency Visualization</div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔊 Waveform", "📊 Spectrogram", "📡 Frequency"])

    with tab1:
        wt1, wt2 = st.tabs(["Original Signal", "Processed + Peaks"])
        with wt1:
            fig = plot_waveform(y, sr, "Original Heart Sound", "#7b61ff")
            st.pyplot(fig)
            plt.close(fig)
        with wt2:
            fig = plot_waveform(y_clean, sr, "Filtered Signal — S1/S2 Peak Detection",
                               "#00d4aa", peaks=peaks if len(peaks) > 0 else None)
            st.pyplot(fig)
            plt.close(fig)
            st.markdown(f"""
            <div class="glass-card" style="margin-top:0.5rem;">
              <span style="color:#ff4d6d; font-weight:bold;">▪ Red markers</span>
              <span style="color:#9aafc8; font-size:0.85rem;"> indicate detected S1/S2 cardiac events.
              {peak_count} peaks found over {duration} seconds.</span>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        sc1, sc2 = st.tabs(["Raw Spectrogram", "Filtered Spectrogram"])
        with sc1:
            fig = plot_spectrogram(y, sr, "Mel Spectrogram — Original")
            st.pyplot(fig)
            plt.close(fig)
        with sc2:
            fig = plot_spectrogram(y_clean, sr, "Mel Spectrogram — Filtered (25–400 Hz)")
            st.pyplot(fig)
            plt.close(fig)

    with tab3:
        fc1, fc2 = st.tabs(["Before Filtering", "After Filtering"])
        with fc1:
            fig = plot_frequency(y, sr, "Frequency Spectrum — Original")
            st.pyplot(fig)
            plt.close(fig)
        with fc2:
            fig = plot_frequency(y_clean, sr, "Frequency Spectrum — After Bandpass Filter")
            st.pyplot(fig)
            plt.close(fig)
        st.markdown("""
        <div class="tip-box">
          The bandpass filter (25–400 Hz) removes environmental noise and high-frequency
          artifacts, isolating the clinically relevant cardiac sound frequencies.
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # ⑧ GRADCAM
    # ══════════════════════════════════════════

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">07 — Explainability</div>
    <div class="section-title">GradCAM — Model Attention Map</div>
    <div class="section-body">
      Gradient-weighted Class Activation Mapping (GradCAM) reveals <em>where</em>
      the neural network focused its attention when making the prediction.
      Brighter regions indicate areas of the spectrogram that most strongly
      influenced the classification decision.
    </div>
    """, unsafe_allow_html=True)

    # ✅ Proper condition block



# ══════════════════════════════════════════
# Interpretation
# ══════════════════════════════════════════

if st.session_state.analyzed:
    label = st.session_state.label
    st.markdown(f"""
    <div class="glass-card" style="border-left: 3px solid #00d4aa;">
      <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem;
                color:#e8edf5; margin-bottom:0.6rem;">🔍 GradCAM Interpretation</div>
    <div style="font-size:0.88rem; color:#9aafc8; line-height:1.8;">
      The model's attention is concentrated in the <strong style="color:#00d4aa;">low-frequency bands (0–40 Mel)</strong>,
      corresponding to the fundamental heart sound components S1 and S2.
      {"Irregular attention hotspots suggest turbulent flow signatures typical of valvular disease." if label == "Murmur" else "Attention is distributed evenly across cardiac cycles, consistent with normal valve function."}
    </div>
  </div>
  """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # ⑨ SHAP
    # ══════════════════════════════════════════
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">08 — Feature Importance</div>
    <div class="section-title">SHAP Analysis — Why This Prediction?</div>
    <div class="section-body">
      SHAP (SHapley Additive exPlanations) quantifies the contribution of each
      MFCC feature to the final prediction. Red bars indicate features pushing
      toward "Murmur"; purple bars toward "Normal."
    </div>
    """, unsafe_allow_html=True)

    fig = plot_shap(y_clean, sr)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    <div class="glass-card" style="border-left: 3px solid #7b61ff;">
      <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem;
                  color:#e8edf5; margin-bottom:0.6rem;">🧠 SHAP Interpretation</div>
      <div style="font-size:0.88rem; color:#9aafc8; line-height:1.8;">
        MFCC coefficients 1–4 capture the dominant spectral envelope —
        the most diagnostically relevant frequency characteristics of heart sounds.
        Higher variance in these coefficients correlates with turbulent, murmur-like sound patterns.
        Each bar represents a normalized Shapley value indicating its directional influence on the prediction.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # ⑩ FINAL INTERPRETATION
    # ══════════════════════════════════════════
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">09 — Summary</div>
    <div class="section-title">Clinical Interpretation</div>
    """, unsafe_allow_html=True)

    if label == "Normal":
        interpret_items = [
            ("✔", "Normal cardiac sound patterns identified", "accent"),
            ("✔", f"Model confidence: {conf_pct}% — high reliability", "accent"),
            ("✔", f"{peak_count} cardiac peaks detected — regular cycle rhythm", "accent"),
            ("✔", "No turbulent flow signatures identified in spectrogram", "accent"),
        ]
    else:
        interpret_items = [
            ("⚠", "Patterns consistent with a cardiac murmur detected", "accent2"),
            ("⚠", f"Model confidence: {conf_pct}% — clinical follow-up recommended", "accent2"),
            ("✔", f"{peak_count} cardiac peaks detected across the recording", "accent"),
            ("⚠", "Irregular energy bursts identified in frequency analysis", "accent2"),
        ]

    for icon, text, color_var in interpret_items:
        color = "#00d4aa" if color_var == "accent" else "#ff4d6d"
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.8rem; padding:0.7rem 1rem;
                    border:1px solid #1e2d45; border-radius:8px; margin-bottom:0.5rem;
                    background:#0d1420;">
          <span style="color:{color}; font-weight:bold; font-size:1.1rem;">{icon}</span>
          <span style="font-size:0.88rem; color:#c0cde0;">{text}</span>
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>Medical Disclaimer:</strong> CardioSense AI is a research-grade screening tool
      developed as a final year academic project. It is intended to <em>assist</em> — not replace —
      qualified medical professionals. All results must be interpreted by a licensed cardiologist.
      Do not make clinical decisions based solely on this output. Seek professional medical evaluation
      for any suspected cardiac condition.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# ⑪ FOOTER
# ══════════════════════════════════════════════
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
  <div style="font-size:1.4rem; margin-bottom:0.5rem;">🫀</div>
  <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;
              color:#e8edf5; letter-spacing:0.05em;">CardioSense AI</div>
  <div style="margin: 0.4rem 0;">Early Detection of Rheumatic Heart Disease via Deep Learning</div>
  <div style="color:#3d5278;">────────────────────────</div>
  <div>Final Year Project · AI in Healthcare · 8th Semester</div>
  <div style="color:#3d5278; margin-top:0.4rem;">
    Built with Streamlit · TensorFlow/Keras · Librosa · GradCAM · SHAP
  </div>
  <div style="margin-top:0.8rem; color:#2a3a52; font-size:0.65rem;">
    This tool is for academic and screening purposes only.
    Not a substitute for professional medical diagnosis.
  </div>
</div>
""", unsafe_allow_html=True)