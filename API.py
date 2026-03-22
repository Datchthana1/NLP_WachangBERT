from transformers import CamembertTokenizer, AutoModelForSequenceClassification
import torch

model_name = "Datchthana/wangchanberta-prachathai"
tokenizer = CamembertTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

LABELS = ["politics","human_rights","quality_of_life","international",
          "social","environment","economics","culture",
          "labor","national_security","ict","education"]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits)[0]
    return {label: round(prob.item(), 3) for label, prob in zip(LABELS, probs)}

text = "วันนี้เดินทางจากหอพักที่บางกะปีไปออกกำลังกายที่ลาดพร้าว​ โดยเดินทางจากรถมิเตอร์ไซต์เป็นเวลา 30 นาที ไปออกำลังกายส่วนตัวและขา รวมเวลา 2 ชั่วโมง โดยรถติดบริเวณปั๊มน้ำมันเนื่องจากสงครามอิหร่านและสหรัฐทำให้น้ำมันในประเทศลดลง"
prediction = predict(text)
print(dict(sorted(prediction.items(), key=lambda x: x[1], reverse=True)))

