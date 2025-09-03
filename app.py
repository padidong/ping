
import os
import time
from typing import List, Dict

import numpy as np
from PIL import Image

import streamlit as st

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models

# Optional: timm (for models saved with timm EfficientNet)
try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

# For PyTorch 2.6+ safe/unsafe loading
from torch.serialization import add_safe_globals

# Allowlist Lightning wrapper if file came from Lightning (safe path still enforced)
try:
    from lightning.fabric.wrappers import _FabricModule
    add_safe_globals([_FabricModule])
except Exception:
    pass

# Allowlist timm EfficientNet class (for safe unpickler path)
try:
    from timm.models.efficientnet import EfficientNet as TIMM_EfficientNet
    add_safe_globals([TIMM_EfficientNet])
except Exception:
    pass


# ------------------ Fixed Settings ------------------
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEAN = [0.4850, 0.4560, 0.4060]
STD  = [0.2290, 0.2240, 0.2250]
IMAGE_SIZE = 224  # EfficientNet-B0
MODEL_PATH = "models/efficientnet_b0_fold0.pt"  # single fixed path


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
      .bad { color: #ef4444; }
      .disclaimer {
        padding: 12px 16px;
        border: 1px dashed rgba(234,179,8,.7);
        border-radius: 12px;
        background: rgba(250,204,21,.08);
        margin-bottom: 10px;
      }

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
      <span class="chip">timm support</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------ Utils ------------------
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_in_chans(model: nn.Module) -> int:
    # Try timm's conv_stem
    try:
        if hasattr(model, "conv_stem") and hasattr(model.conv_stem, "weight"):
            return int(model.conv_stem.weight.shape[1])
    except Exception:
        pass
    # Torchvision efficientnet: first conv is in features[0][0]
    try:
        first = getattr(model, "features", [])[0][0]
        if hasattr(first, "weight"):
            return int(first.weight.shape[1])
    except Exception:
        pass
    # Fallback: scan first conv2d
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return int(m.weight.shape[1])
    return 3  # default

def build_transform_adaptive(image_size: int, mean_rgb, std_rgb, in_chans: int):
    # If model expects 1 channel, convert to grayscale and use averaged mean/std
    if in_chans == 1:
        mean_scalar = float(sum(mean_rgb) / 3.0)
        std_scalar  = float(sum(std_rgb)  / 3.0)
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean_scalar], std=[std_scalar]),
        ])
    # If model expects 3 channels (common case)
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_rgb, std=std_rgb),
    ])

def build_torchvision_efficientnet_b0(num_classes: int) -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m

def build_timm_efficientnet_b0(num_classes: int) -> nn.Module:
    if not TIMM_AVAILABLE:
        raise RuntimeError("timm not installed")
    # Create timm model with correct num_classes
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    return m

def sd_looks_like_timm(sd_keys):
    # Heuristics: timm EfficientNet typically has 'conv_stem', 'bn1', 'blocks.0.0.conv_dw' etc.
    for k in sd_keys:
        if k.startswith("conv_stem.") or k.startswith("blocks.") or k.startswith("bn1."):
            return True
    return False

def sd_looks_like_torchvision(sd_keys):
    # Torchvision EfficientNet typically has 'features.0.0.', 'features.1.' etc.
    for k in sd_keys:
        if k.startswith("features."):
            return True
    return False

def try_load_state_dict_into_models(sd: Dict[str, torch.Tensor], num_classes: int, dev):
    # Clean prefixes
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    if any(k.startswith("model.") for k in sd.keys()):
        sd = {k.replace("model.", ""): v for k, v in sd.items()}

    keys = list(sd.keys())

    # Prefer timm if keys look like timm
    errs = []
    if TIMM_AVAILABLE and sd_looks_like_timm(keys):
        try:
            m = build_timm_efficientnet_b0(num_classes)
            m.load_state_dict(sd, strict=False)
            m.eval()
            return m.to(dev)
        except Exception as e:
            errs.append(f"timm load failed: {e}")

    # Try torchvision
    if sd_looks_like_torchvision(keys) or True:
        try:
            m = build_torchvision_efficientnet_b0(num_classes)
            m.load_state_dict(sd, strict=False)
            m.eval()
            return m.to(dev)
        except Exception as e:
            errs.append(f"torchvision load failed: {e}")

    raise RuntimeError("State dict load failed.\n" + "\n".join(errs))

def _torch_load(path, map_location, weights_only=None):
    try:
        if weights_only is None:
            return torch.load(path, map_location=map_location)
        else:
            return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)

@st.cache_resource(show_spinner=False)
def load_fixed_model(path: str, num_classes: int, dev):
    """
    Loader order (robust for PyTorch 2.6+ & timm):
    1) TorchScript
    2) SAFE: weights_only=True → build (timm or torchvision) and load state_dict
    3) UNSAFE: weights_only=False (only if file trusted) — requires dependencies (e.g., timm, lightning) installed
    4) Fallback: attempt to interpret any loaded object as state_dict-like
    """
    # 1) TorchScript
    try:
        m = torch.jit.load(path, map_location=dev)
        m.eval()
        return m.to(dev)
    except Exception:
        pass

    # 2) SAFE weights_only
    try:
        obj = _torch_load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            sd = obj.get("state_dict", obj)
            if isinstance(sd, dict):
                return try_load_state_dict_into_models(sd, num_classes, dev)
    except Exception as e:
        safe_err = str(e)

    # 3) UNSAFE (requires trust + installed classes)
    try:
        obj = _torch_load(path, map_location=dev, weights_only=False)
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return try_load_state_dict_into_models(obj["state_dict"], num_classes, dev)
        # If full model object (e.g., timm EfficientNet), try .eval() directly
        if hasattr(obj, "eval"):
            try:
                obj.eval()
                return obj.to(dev)
            except Exception:
                pass
        # Or object contains nested model
        if isinstance(obj, dict) and "model" in obj:
            m = obj["model"]
            try:
                m.eval()
                return m.to(dev)
            except Exception:
                pass
    except Exception as e:
        unsafe_err = str(e)
        # continue to fallback

    # 4) Fallback: try to interpret any object as state_dict
    try:
        any_obj = _torch_load(path, map_location="cpu", weights_only=None)
        if isinstance(any_obj, dict):
            cand = any_obj.get("state_dict", any_obj)
            if isinstance(cand, dict):
                return try_load_state_dict_into_models(cand, num_classes, dev)
    except Exception as e:
        fallback_err = str(e)

    msg = "Failed to load model.\n"
    if 'safe_err' in locals(): msg += f"- SAFE load error: {safe_err}\n"
    if 'unsafe_err' in locals(): msg += f"- UNSAFE load error: {unsafe_err}\n"
    if 'fallback_err' in locals(): msg += f"- Fallback error: {fallback_err}\n"
    raise RuntimeError(msg)

def preprocess(pil: Image.Image, image_size: int, mean: List[float], std: List[float], dev, in_chans: int):
    tfm = build_transform_adaptive(image_size, mean, std, in_chans)
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
    st.error(f"โหลดโมเดลไม่สำเร็จ:\n{e}")
    if not TIMM_AVAILABLE:
        st.info("คำแนะนำ: โมเดลนี้อาจสร้างจาก `timm` ให้ลองติดตั้งเพิ่ม:\n\n`pip install timm`")
    st.stop()


# ------------------ Disclaimer ------------------
DISCLAIMER_TEXT = "Medical Disclaimer: For education/research only. Not a medical device. Do not use for diagnosis."
st.markdown(f"<div class='disclaimer'><b>{DISCLAIMER_TEXT}</b></div>", unsafe_allow_html=True)
agree = st.checkbox("I have read and understood the medical disclaimer.", value=False)
if not agree:
    st.info("Please acknowledge the disclaimer above to proceed.")
    st.stop()

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
            x = preprocess(pil, IMAGE_SIZE, MEAN, STD, dev, st.session_state.get('in_chans', 3))
            try:
                probs = predict(MODEL, x)
                elapsed = (time.time() - start) * 1000.0

                idx = int(np.argmax(probs))
                pred_class = CLASSES[idx]
                pred_prob = float(probs[idx])

                st.markdown(f"**ผลการทำนาย:** `{pred_class}`  (ความเชื่อมั่น {pred_prob:.2%})")
                st.caption(f"Inference: ~{elapsed:.1f} ms")

                with st.expander("ดูความน่าจะเป็นของทุกคลาส"):
                    for c, p in sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True):
                        st.write(f"- **{c}**: {p:.2%}")
            except Exception as e:
                st.error("ไม่สามารถรันพยากรณ์ได้ (ดู Logs รายละเอียด)")
                st.info("Hint: ถ้าโมเดลเทรนแบบช่องสัญญาณภาพเป็น 1 (Grayscale) แอปจะแปลงให้แล้ว; ถ้ายังพังตรวจสอบว่าไฟล์คือ EfficientNet-B0 7-คลาสจริงหรือไม่ หรือแปลงเป็น state_dict-only ก่อนใช้.")
                raise

else:
    st.info("อัปโหลดภาพเพื่อดูผลการทำนายจากโมเดล EfficientNet-B0")

# ------------------ Footer ------------------
st.markdown("<br><div style='text-align:center;opacity:.8;'><small>© 2025 — EfficientNet-B0 Skin Lesion Demo</small></div>", unsafe_allow_html=True)
