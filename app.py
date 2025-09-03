
import os
import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import altair as alt

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models


st.set_page_config(
    page_title="Skin Lesion Classifier — EfficientNet (7-class)",
    page_icon="🩺",
    layout="wide",
)

# ---------------- UI Styles ----------------
st.markdown(
    """
    <style>
      #MainMenu, header, footer {visibility: hidden;}
      .app-header {
        display:flex; align-items:center; gap:12px;
        padding: 10px 14px; margin-bottom: 8px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(59,130,246,.10), rgba(16,185,129,.10));
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
    <div class="app-header">
      <h2 style="margin: 6px 0;">🩺 Skin Lesion Classifier — EfficientNet (7-class)</h2>
      <span class="chip">akiec, bcc, bkl, df, mel, nv, vasc</span>
      <span class="chip">EfficientNet (torchvision)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------- Constants --------------
DEFAULT_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DEFAULT_MEAN = [0.4850, 0.4560, 0.4060]
DEFAULT_STD  = [0.2290, 0.2240, 0.2250]

# Suggested default input sizes per EfficientNet variant
EFFICIENTNET_IMG_SUGGEST = {
    "efficientnet_b0": 224,
    "efficientnet_b1": 240,
    "efficientnet_b2": 260,
    "efficientnet_b3": 300,
    "efficientnet_b4": 380,
    "efficientnet_b5": 456,
    "efficientnet_b6": 528,
    "efficientnet_b7": 600,
}

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------- Model Utils --------------
def build_efficientnet(backbone: str, num_classes: int) -> nn.Module:
    backbone = backbone.lower()
    ctor = getattr(models, backbone, None)
    if ctor is None:
        raise ValueError(f"Unknown EfficientNet backbone: {backbone}")
    m = ctor(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m

@st.cache_resource(show_spinner=False)
def load_model(model_path: str, load_mode: str, backbone: str, num_classes: int, device: torch.device):
    if load_mode == "TorchScript (.pt/.ts)":
        m = torch.jit.load(model_path, map_location=device)
        m.eval()
        return m.to(device)
    elif load_mode == "Full model (torch.save(model))":
        obj = torch.load(model_path, map_location=device)
        m = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        try: m.eval()
        except Exception: pass
        return m.to(device)
    elif load_mode == "State dict (+EfficientNet backbone)":
        sd = torch.load(model_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        m = build_efficientnet(backbone, num_classes)
        try:
            m.load_state_dict(sd, strict=False)
        except Exception:
            new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
            m.load_state_dict(new_sd, strict=False)
        m.eval()
        return m.to(device)
    else:
        raise ValueError("Unknown load mode")

def build_transform(image_size: int, mean: List[float], std: List[float]):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def preprocess(img: Image.Image, image_size: int, mean: List[float], std: List[float], device):
    tfm = build_transform(image_size, mean, std)
    x = tfm(img).unsqueeze(0).to(device)
    return x

@torch.inference_mode()
def predict_proba(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    logits = model(x)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    p = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return p

# ---------- Grad-CAM (simple) for EfficientNet ----------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self.fwd_hook)
        self.h2 = target_layer.register_full_backward_hook(self.bwd_hook)

    def fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        if isinstance(out, (list, tuple)): out = out[0]
        score = out[:, class_idx].sum()
        score.backward(retain_graph=True)

        A = self.activations       # [B,C,H,W]
        dA = self.gradients        # [B,C,H,W]
        weights = dA.mean(dim=(2,3), keepdim=True)      # [B,C,1,1]
        cam = (weights * A).sum(dim=1, keepdim=True)    # [B,1,H,W]
        cam = torch.relu(cam)
        cam = cam[0,0].cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

    def remove(self):
        self.h1.remove(); self.h2.remove()

def overlay_cam_on_image(pil: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import matplotlib.cm as cm
    from PIL import Image
    heat = (cm.get_cmap('jet')(cam) * 255).astype(np.uint8)[:,:,:3]
    heat = Image.fromarray(heat).resize(pil.size, resample=Image.BILINEAR)
    return Image.blend(pil.convert("RGB"), heat, alpha)

# -------------- Sidebar --------------
with st.sidebar:
    st.title("⚙️ Settings")

    default_model_path = os.environ.get("MODEL_PATH", "models/efficientnet.pt")
    model_path = st.text_input("Model path (.pt/.pth/.ts)", value=default_model_path)

    load_mode = st.selectbox(
        "Load mode",
        ["TorchScript (.pt/.ts)", "Full model (torch.save(model))", "State dict (+EfficientNet backbone)"],
        index=2
    )

    eff_list = [f"efficientnet_b{i}" for i in range(8)]
    backbone = st.selectbox("Backbone", eff_list, index=0)

    # Suggest image size based on backbone
    suggested = EFFICIENTNET_IMG_SUGGEST.get(backbone, 224)
    image_size = st.number_input("Image size", min_value=128, max_value=640, value=suggested, step=4)

    classes_text = st.text_input("Classes (comma-separated)", value=",".join(DEFAULT_CLASSES))
    classes = [c.strip() for c in classes_text.split(",") if c.strip()]
    num_classes = len(classes) if classes else 7

    mean_txt = st.text_input("Mean (RGB)", value="0.4850,0.4560,0.4060")
    std_txt = st.text_input("Std (RGB)", value="0.2290,0.2240,0.2250")
    try:
        mean = [float(x) for x in mean_txt.split(",")]
        std = [float(x) for x in std_txt.split(",")]
    except Exception:
        st.warning("Mean/Std parse error → fallback default")
        mean, std = DEFAULT_MEAN, DEFAULT_STD

    device = get_device()
    st.caption(f"Device: **{device}**")

    reload_btn = st.button("🔁 Load / Reload model")

# Load / cache model
if "model" not in st.session_state or reload_btn:
    try:
        with st.spinner("Loading model..."):
            st.session_state.model = load_model(model_path, load_mode, backbone, num_classes, device)
        st.sidebar.success("Model loaded ✅")
    except Exception as e:
        st.sidebar.error(f"Load failed: {e}")
        st.stop()

model = st.session_state.model

# -------------- Disclaimer --------------
st.markdown(
    """
    <div class="warn"><strong>⚠️ Medical Disclaimer:</strong> This demo is for <b>research/education</b> only.
    Not a medical device. Do not use for diagnosis or treatment.</div>
    """,
    unsafe_allow_html=True,
)

# -------------- Uploader --------------
st.markdown("### 📤 Upload dermoscopic image(s)")
files = st.file_uploader(
    "รองรับ: JPG, PNG, BMP, TIFF (หลายไฟล์ได้)",
    type=["jpg","jpeg","png","bmp","tif","tiff"],
    accept_multiple_files=True
)

# -------------- Inference --------------
if files:
    cols = st.columns([1.1, 1.3])
    with cols[0]: st.subheader("🖼️ Preview")
    with cols[1]: st.subheader("📈 Prediction")

    for f in files:
        name = f.name
        pil = Image.open(f).convert("RGB")

        c1, c2 = st.columns([1.1, 1.3])

        with c1:
            st.markdown(f"**{name}**")
            st.image(pil, use_column_width=True, caption="Original")

        with c2:
            start = time.time()
            x = preprocess(pil, image_size=image_size, mean=mean, std=std, device=device)
            probs = predict_proba(model, x)
            elapsed = (time.time() - start) * 1000.0

            df = pd.DataFrame({"class": classes, "probability": probs})
            df = df.sort_values("probability", ascending=False).reset_index(drop=True)
            top1 = df.iloc[0]

            st.markdown(f"<span class='ok'>Top-1: {top1['class']} ({top1['probability']:.2%})</span> &nbsp; <small>(~{elapsed:.1f} ms)</small>", unsafe_allow_html=True)

            chart = (
                alt.Chart(df).mark_bar().encode(
                    x=alt.X("probability:Q", axis=alt.Axis(format=".0%"), title="Probability"),
                    y=alt.Y("class:N", sort="-x", title="Class"),
                    tooltip=["class", alt.Tooltip("probability:Q", format=".2%")]
                ).properties(height=240)
            )
            st.altair_chart(chart, use_container_width=True)

            # Grad-CAM (target last conv block in features)
            with st.expander("🔎 Explainability — Grad-CAM (EfficientNet)"):
                try:
                    # find a conv layer near the end of features
                    target_layer = None
                    if hasattr(model, "features"):
                        # pick the last module that has conv-like output
                        for m in reversed(model.features):
                            target_layer = m
                            break
                    if target_layer is None:
                        st.info("ไม่พบ target layer ใน model.features")
                    else:
                        cam = GradCAM(model, target_layer)
                        cls_idx = int(np.argmax(probs))
                        heat = cam(x, cls_idx)
                        cam.remove()
                        overlay = overlay_cam_on_image(pil, heat, alpha=0.45)
                        st.image(overlay, use_column_width=True, caption=f"Grad-CAM overlay → class: {classes[cls_idx]}")
                except Exception as e:
                    st.info(f"Grad-CAM unavailable: {e}")

else:
    st.info("อัปโหลดรูปเพื่อเริ่มการพยากรณ์ด้วย EfficientNet")

# -------------- Footer --------------
st.markdown("<br><div style='text-align:center;opacity:.8;'><small>© 2025 — EfficientNet Skin Lesion Demo (education only)</small></div>", unsafe_allow_html=True)
