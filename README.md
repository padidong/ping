
# Skin Lesion — EfficientNet-B0 (Single Model)

ใช้โมเดลเดียวตายตัว: `models/efficientnet_b0_fold0.pt`  
คลาส 7 ประเภท: `akiec, bcc, bkl, df, mel, nv, vasc`  
Normalize: `mean=[0.4850,0.4560,0.4060], std=[0.2290,0.2240,0.2250]`

## โครงสร้าง
```
skin-lesion-efficientnet-b0-single/
├─ app.py
├─ requirements.txt
├─ .streamlit/
│  └─ config.toml
├─ assets/
│  └─ styles.css
└─ models/
   ├─ efficientnet_b0_fold0.pt   # ← ใส่ไฟล์โมเดลของคุณชื่อนี้
   └─ README.txt
```

## วิธีรัน
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

# วางโมเดลไว้ที่ models/efficientnet_b0_fold0.pt
streamlit run app.py
```
เปิด `http://localhost:8501` แล้วอัปโหลดรูป ระบบจะแสดงผลว่าเป็นคลาสใด พร้อมความเชื่อมั่น (%)

> ตัวโหลดโมเดลพยายามอ่านแบบ: TorchScript → Full model → State dict (จะ map เข้ากับ EfficientNet-B0 ให้อัตโนมัติ)

## Deploy
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
หรือรันเป็น systemd แล้ววางหลัง reverse proxy ได้ตามสะดวก

**หมายเหตุสำคัญ:** แอปนี้เพื่อการศึกษา/วิจัยเท่านั้น ไม่ใช่อุปกรณ์ทางการแพทย์
