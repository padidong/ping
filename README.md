
# Skin Lesion — EfficientNet-B0 (Single Model, timm-safe)

**Single fixed model:** `models/efficientnet_b0_fold0.pt`  
**Classes (7):** `akiec, bcc, bkl, df, mel, nv, vasc`  
**Normalize:** `mean=[0.4850,0.4560,0.4060]`, `std=[0.2290,0.2240,0.2250]`, `image_size=224`  

## What's new
- รองรับ **timm**: โหลด state_dict เข้า timm หรือ torchvision โดยอัตโนมัติ
- อนุญาต safe globals สำหรับ `timm.models.efficientnet.EfficientNet` และ Lightning `_FabricModule`
- ลอจิกโหลดแบบทนทาน: TorchScript → **weights_only=True** (safe) → **weights_only=False** (trusted) → fallback

## Quick Start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

# ใส่โมเดลของคุณชื่อ: models/efficientnet_b0_fold0.pt
streamlit run app.py
```
เปิด `http://localhost:8501` แล้วอัปโหลดรูป

> ถ้าโมเดลคุณถูกบันทึกแบบ **full object** จาก timm/Lightning ให้แน่ใจว่าติดตั้ง `timm` และ/หรือ `lightning` ด้วย

## Tools: Convert to state_dict only
```
python tools/convert_to_state_dict.py --src models/efficientnet_b0_fold0.pt --dst models/efficientnet_b0_fold0_state.pt
```
แล้วเปลี่ยน `MODEL_PATH` หรือรีเนมไฟล์แทน

**คำเตือน:** โหมด `weights_only=False` มีความเสี่ยงการรันโค้ดจากไฟล์ ทำเฉพาะเมื่อเชื่อใจแหล่งไฟล์จริง ๆ
