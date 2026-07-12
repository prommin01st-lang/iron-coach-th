"""
เทียบคำตอบ ก่อน vs หลัง fine-tune
รัน:
    .venv/bin/python experiments/01_first_finetune/compare.py
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "output", "final_adapter")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SYS = "คุณเป็นโค้ชเวทเทรนนิ่งที่ยึดหลักวิทยาศาสตร์การกีฬา ตอบกระชับ ถูกต้อง และคำนึงถึงความปลอดภัยเสมอ"

# คำถามทดสอบ — บางข้อโมเดล "ไม่เคยเห็น" ตอนเทรน (เช็ค generalize + กัน hallucination)
TESTS = [
    "อยากเพิ่มความแข็งแรงท่า Barbell Row จัด set/rep ยังไง",   # ท่าใหม่ ไม่มีใน train
    "ปวดเข่ามากตอน Squat ควรทำยังไง",                          # ควรแนะนำพบผู้เชี่ยวชาญ
    "deload คืออะไร",                                          # มีใน train
]

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(MODEL_ID)


def gen(model, question):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": question}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


print(">> loading BASE model...")
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)

print(">> loading FINE-TUNED (base + adapter)...")
ft = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16),
    ADAPTER,
)

for q in TESTS:
    print("\n" + "=" * 70)
    print(f"❓ {q}")
    print("-" * 70)
    print(f"[ก่อน fine-tune]\n{gen(base, q)}")
    print("-" * 70)
    print(f"[หลัง fine-tune]\n{gen(ft, q)}")
