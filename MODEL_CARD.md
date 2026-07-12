---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
tags:
  - lora
  - qlora
  - thai
  - fitness
  - weight-training
language:
  - th
pipeline_tag: text-generation
---

# Fitness Coach (Thai) — QLoRA adapter for Qwen2.5-1.5B-Instruct

LoRA adapter ที่ fine-tune ให้ Qwen2.5-1.5B-Instruct เป็น **โค้ชเวทเทรนนิ่งภาษาไทย** ยึดหลักวิทยาศาสตร์การกีฬา

## Intended use
ตอบคำถามเรื่องการฝึกเวท: set/rep ตามเป้าหมาย, ฟอร์มท่า (Deadlift/Squat/Bench/…), programming, โภชนาการทั่วไป
- ปฏิเสธคำขอที่อันตราย/ผิดกฎหมาย (สเตียรอยด์, ลดน้ำหนักสุดโต่ง) อย่างชัดเจน
- แนะให้พบแพทย์/ผู้เชี่ยวชาญเมื่อเป็นเรื่องอาการบาดเจ็บหรือโรคประจำตัว

## Training
- **Base:** Qwen/Qwen2.5-1.5B-Instruct
- **Method:** QLoRA (4-bit NF4) + LoRA (r=16, alpha=32) via `trl` SFTTrainer
- **Hardware:** NVIDIA RTX 2060 Mobile (6GB VRAM), bf16
- **Data:** ~110–156 ตัวอย่าง Q&A ภาษาไทย (สร้างเอง) ครอบคลุม programming/form/nutrition/recovery/safety-refusal/out-of-scope
- **System prompt:** `คุณเป็นโค้ชเวทเทรนนิ่งที่ยึดหลักวิทยาศาสตร์การกีฬา ตอบเป็นภาษาไทยเสมอ กระชับ ถูกต้อง คำนึงถึงความปลอดภัย และปฏิเสธคำขอที่อันตรายหรือผิดกฎหมายอย่างชัดเจน`

## How to use
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "<your-username>/fitness-coach-th")

sys = "คุณเป็นโค้ชเวทเทรนนิ่งที่ยึดหลักวิทยาศาสตร์การกีฬา ตอบเป็นภาษาไทยเสมอ กระชับ ถูกต้อง คำนึงถึงความปลอดภัย และปฏิเสธคำขอที่อันตรายหรือผิดกฎหมายอย่างชัดเจน"
msgs = [{"role":"system","content":sys},{"role":"user","content":"อยากเพิ่มความแข็งแรง Deadlift จัด set/rep ยังไง"}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=200)
print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Limitations & disclaimer
- โมเดลเล็ก (1.5B) — บางครั้งอาจตอบผิด/มั่ว/ปนภาษาอื่นเล็กน้อย
- ให้คำแนะนำทั่วไปเพื่อการศึกษา **ไม่ใช่คำแนะนำทางการแพทย์** ควรปรึกษาผู้เชี่ยวชาญก่อนเริ่มฝึกจริง
