# nlp_tuning

Fine-tuning [WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased) สำหรับงาน **Multi-label Text Classification** ภาษาไทย โดยใช้ dataset [prachathai-67k](https://huggingface.co/datasets/PyThaiNLP/prachathai67k)

## งานที่โมเดลทำ

จำแนกบทความข่าวภาษาไทยว่าเกี่ยวข้องกับหมวดหมู่ใดบ้าง (1 บทความสามารถอยู่หลายหมวดได้พร้อมกัน):

| หมวดหมู่ | ความหมาย |
|---|---|
| politics | การเมือง |
| human_rights | สิทธิมนุษยชน |
| quality_of_life | คุณภาพชีวิต |
| international | ต่างประเทศ |
| social | สังคม |
| environment | สิ่งแวดล้อม |
| economics | เศรษฐกิจ |
| culture | วัฒนธรรม |
| labor | แรงงาน |
| national_security | ความมั่นคง |
| ict | เทคโนโลยี |
| education | การศึกษา |

## ผลการ Training

| Epoch | F1 Score (micro) | Eval Loss |
|---|---|---|
| 1 | 0.9297 | 0.1734 |
| 2 | 0.9343 | 0.1642 |
| **3** | **0.9360** | **0.1621** |

**Best F1 Score: 0.9360** (epoch 3)

### F1 Score คืออะไร?

F1 Score คือค่าที่วัดว่าโมเดลทายถูกแม่นแค่ไหน โดยรวมทั้ง 2 มิติ:
- **Precision** — ในสิ่งที่โมเดลบอกว่าใช่ มีถูกจริงกี่เปอร์เซ็นต์
- **Recall** — ในสิ่งที่ถูกต้องทั้งหมด โมเดลหาเจอกี่เปอร์เซ็นต์

ค่า **0.9360** หมายความว่าโมเดลทายถูกต้องประมาณ **93.6%** รวมทุกหมวดหมู่

### Training Loss

Loss ลดลงจาก **0.287** (ต้น epoch 1) เหลือ **0.141** (ปลาย epoch 3) แสดงว่าโมเดลเรียนรู้ได้ดีขึ้นต่อเนื่องตลอด 3 รอบ

## การตั้งค่า

| Parameter | ค่าที่ใช้ |
|---|---|
| Base model | WangchanBERTa (CamemBERT-base) |
| Dataset | prachathai-67k (54,379 train / 6,721 val / 6,789 test) |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Max token length | 128 |
| Optimizer | AdamW (default) |
| LR Schedule | Linear decay |
