
# Skin Lesion Classifier — **EfficientNet** (7-class)

**Classes:** `akiec, bcc, bkl, df, mel, nv, vasc`  
**Normalize:** `mean=[0.4850, 0.4560, 0.4060]`, `std=[0.2290, 0.2240, 0.2250]`

- รองรับการโหลดโมเดล 3 รูปแบบ: TorchScript, Full model, หรือ State dict + เลือก EfficientNet backbone (b0–b7)
- แสดง Top-1 + กราฟความน่าจะเป็นทั้งหมด
- มี Grad-CAM (แบบง่าย) สำหรับ EfficientNet

## Quick Start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
# ใส่โมเดลไว้ใน models/efficientnet.pt (หรือชื่ออื่นก็ได้)
streamlit run app.py
```

## Notes
- ถ้าเป็น state_dict ต้องเลือก backbone ให้ตรงตอนเทรนและตั้ง num_classes=7
- แนะนำ image_size ตาม EfficientNet: b0=224, b3=300, b5=456, b7=600 (ปรับใน Sidebar)
- Grad-CAM เป็นเวอร์ชันง่าย อาจไม่รองรับโมเดลที่ export เป็น TorchScript

## Deploy
- เปิดพอร์ต:
  ```bash
  streamlit run app.py --server.address 0.0.0.0 --server.port 8501
  ```
- หรือรันเป็น systemd, หรือวางหลัง reverse proxy (nginx/Apache) เช่นเดียวกับสไตล์ Streamlit ปกติ
# ping
