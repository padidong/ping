
import os
import time
from typing import List

import numpy as np
from PIL import Image

import streamlit as st

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models

from torch.serialization import add_safe_globals, safe_globals

# allowlist Lightning wrappers if present (only if you trust your checkpoint source)
try:
    from lightning.fabric.wrappers import _FabricModule
    add_safe_globals([_FabricModule])
except Exception:
    pass

def _torch_load(path, map_location, weights_only=None):
    # Handle PyTorch <2.6 (no weights_only) and >=2.6 (default True)
    try:
        if weights_only is None:
            return torch.load(path, map_location=map_location)
        else:
            return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # for older torch versions without weights_only
        return torch.load(path, map_location=map_location)



# ------------------ Fixed Settings ------------------
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEAN = [0.4850, 0.4560, 0.4060]
STD  = [0.2290, 0.2240, 0.2250]
IMAGE_SIZE = 224  # EfficientNet-B0 recommended
MODEL_PATH = "models/efficientnet_b0_fold0.pt"  # <- fixed single model path


st.set_page_config(
    page_title="Skin Lesion — EfficientNet-B0 (7-class)",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
      #MainMenu, header, footer {visibility: hidden;}
      .header {
        display:flex; align-items:center; gap:12px;
        padding: 10px 14px; margin-bottom: 12px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(59,130,246,.12), rgba(16,185,129,.12));
        border: 1px solid rgba(148,163,184,.35);
      }
      .chip {
        display:inline-block; padding: 2px 10px; font-size: .78rem;
        background: rgba(148,163,184,.15); border: 1px solid rgba(148,163,184,.35);
        border-radius: 999px; margin-left: 8px;
      }
      .warn {
        padding: 10px 14px; border: 1px dashed rgba(234,179,8,.6);
        border-radius: 12px; background: rgba(250,204,21,.08);
      }
      .ok { color: #22c55e; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="header">
      <h2 style="margin: 6px 0;">🩺 Skin Lesion Classifier — EfficientNet-B0</h2>
      <span class="chip">7 classes: akiec, bcc, bkl, df, mel, nv, vasc</span>
      <span class="chip">Fixed model: efficientnet_b0_fold0.pt</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------ Utils ------------------
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_efficientnet_b0(num_classes: int) -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m

@st.cache_resource(show_spinner=False)
def load_fixed_model(path: str, num_classes: int, dev):
    """
    Robust loader that tries:
    1) TorchScript (jit)
    2) Full model object
    3) State dict onto EfficientNet-B0 head
    """
    # 1) TorchScript
    try:
        m = torch.jit.load(path, map_location=dev)
        m.eval()
        return m.to(dev)
    except Exception:
        pass

    # 2) Full model
    try:
        obj = torch.load(path, map_location=dev)
        m = obj.get("model", obj) if isinstance(obj, dict) else obj
        try: m.eval()
        except Exception: pass
        return m.to(dev)
    except Exception:
        pass

    # 3) State dict
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m = build_efficientnet_b0(num_classes=num_classes)
    try:
        m.load_state_dict(sd, strict=False)
    except Exception:
        sd2 = {k.replace("module.", ""): v for k, v in sd.items()}
        m.load_state_dict(sd2, strict=False)
    m.eval()
    return m.to(dev)

def preprocess(pil: Image.Image, image_size: int, mean: List[float], std: List[float], dev):
    tfm = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    x = tfm(pil).unsqueeze(0).to(dev)
    return x

@torch.inference_mode()
def predict(model: nn.Module, x: torch.Tensor):
    logits = model(x)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()[0]


# ------------------ Load model once ------------------
dev = device()
try:
    with st.spinner("Loading EfficientNet-B0 model..."):
        MODEL = load_fixed_model(MODEL_PATH, num_classes=len(CLASSES), dev=dev)
    st.success("Model loaded ✅")
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# ------------------ Disclaimer ------------------
st.markdown(
    """
    <div class="warn"><strong>⚠️ Medical Disclaimer:</strong> For education/research only.
    Not a medical device. Do not use for diagnosis.</div>
    """,
    unsafe_allow_html=True,
)

# ------------------ Uploader ------------------
st.markdown("### 📤 อัปโหลดภาพผิวหนัง (JPG/PNG/BMP/TIFF)")
files = st.file_uploader(
    "เลือกไฟล์ภาพ (อัปโหลดหลายไฟล์ได้)",
    type=["jpg","jpeg","png","bmp","tif","tiff"],
    accept_multiple_files=True
)

# ------------------ Inference ------------------
if files:
    for f in files:
        name = f.name
        pil = Image.open(f).convert("RGB")

        c1, c2 = st.columns([1.0, 1.0])
        with c1:
            st.image(pil, use_column_width=True, caption=name)

        with c2:
            start = time.time()
            x = preprocess(pil, IMAGE_SIZE, MEAN, STD, dev)
            probs = predict(MODEL, x)
            elapsed = (time.time() - start) * 1000.0

            idx = int(np.argmax(probs))
            pred_class = CLASSES[idx]
            pred_prob = float(probs[idx])

            st.markdown(f"**ผลการทำนาย:** `{pred_class}`  (ความเชื่อมั่น {pred_prob:.2%})")
            st.caption(f"Inference: ~{elapsed:.1f} ms")

            with st.expander("ดูความน่าจะเป็นของทุกคลาส (Top-7)"):
                for c, p in sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True):
                    st.write(f"- **{c}**: {p:.2%}")
else:
    st.info("อัปโหลดภาพเพื่อดูผลการทำนายจากโมเดล EfficientNet-B0")

# ------------------ Footer ------------------
st.markdown("<br><div style='text-align:center;opacity:.8;'><small>© 2025 — EfficientNet-B0 Skin Lesion Demo</small></div>", unsafe_allow_html=True)
