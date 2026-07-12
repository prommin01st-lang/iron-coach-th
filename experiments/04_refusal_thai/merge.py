"""
Merge LoRA adapter เข้า base model (fp16) → โมเดลเต็มพร้อม deploy
โหลดบน CPU (fp16 1.5B ~3GB) เพื่อไม่กิน VRAM

รัน: .venv/bin/python experiments/04_refusal_thai/merge.py
"""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = os.path.join(HERE, "output", "final_adapter")
OUT = os.path.join(HERE, "merged")

print(">> loading base (fp16, CPU)...")
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)

print(">> applying + merging adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()   # รวม LoRA เข้าน้ำหนักหลัก

print(f">> saving merged model to {OUT} ...")
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print(">> DONE merge")
