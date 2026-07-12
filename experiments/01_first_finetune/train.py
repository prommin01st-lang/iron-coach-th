"""
Fine-tune ตัวแรก: บอทเวทเทรนนิ่ง ด้วย QLoRA
โมเดล: Qwen2.5-0.5B-Instruct (เล็ก เหมาะเริ่มต้นบน RTX 2060 6GB)

รันจากโฟลเดอร์ Model Trainer:
    .venv/bin/python experiments/01_first_finetune/train.py
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ---- พาธ (อ้างอิงจากตำแหน่งไฟล์นี้ ทำให้รันจากที่ไหนก็ได้) ----
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(HERE, "output")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# ============================================================
# 1) Quantization config — โหลด base model แบบ 4-bit (หัวใจของ QLoRA)
#    ทำให้โมเดลกิน VRAM น้อยลง ~4 เท่า → เข้า 6GB ได้สบาย
# ============================================================
COMPUTE_DTYPE = torch.bfloat16               # bf16 = ไม่ต้องใช้ GradScaler → เลี่ยงบั๊ก fp16 บน Turing

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # โหลดน้ำหนักแบบ 4-bit
    bnb_4bit_quant_type="nf4",               # NF4 = รูปแบบ 4-bit ที่แม่นสุดสำหรับ LLM
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,    # ตอนคำนวณใช้ bf16
    bnb_4bit_use_double_quant=True,          # quantize ซ้ำอีกชั้น ประหยัด VRAM เพิ่ม
)

# ============================================================
# 2) โหลด tokenizer + model
# ============================================================
print(">> loading tokenizer & model (4-bit)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",                       # วางลง GPU อัตโนมัติ
    dtype=COMPUTE_DTYPE,                       # bf16 สอดคล้องกับ compute dtype และ bf16 training
)
model.config.use_cache = False               # ต้องปิดตอนเทรน (ใช้คู่ gradient checkpointing)

# ============================================================
# 3) LoRA config — เทรนแค่ adapter เล็ก ๆ (~0.1-1% ของพารามิเตอร์)
#    base model ถูกแช่แข็ง เราปรับแค่เมทริกซ์ที่แปะเข้าไป
# ============================================================
lora_config = LoraConfig(
    r=16,                 # rank ของ adapter — สูง = จุความรู้ได้มากขึ้น (เริ่ม 8-16)
    lora_alpha=32,        # สเกล มักตั้ง = 2 * r
    lora_dropout=0.05,    # กัน overfitting เล็กน้อย
    bias="none",
    task_type="CAUSAL_LM",
    # เลเยอร์ที่แปะ adapter (attention + MLP ของ Qwen)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# เตรียมโมเดล 4-bit ให้เทรนได้ + เปิด gradient checkpointing (ประหยัด VRAM)
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
# แปะ LoRA adapter
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# 4) โหลด dataset (รูปแบบ chat 'messages' — trl จะ apply chat template ให้เอง)
# ============================================================
print(">> loading dataset...")
train_ds = load_dataset("json", data_files=os.path.join(DATA, "train.jsonl"), split="train")
val_ds = load_dataset("json", data_files=os.path.join(DATA, "val.jsonl"), split="train")
print(f"   train={len(train_ds)} examples, val={len(val_ds)} examples")

# ============================================================
# 5) Training config
#    - batch เล็ก (1) + gradient accumulation (8) = จำลอง batch 8 โดยไม่กิน VRAM
#    - gradient checkpointing = แลกความเร็วเล็กน้อยเพื่อประหยัด VRAM
#    - paged_adamw_8bit = optimizer ประหยัดหน่วยความจำ
# ============================================================
sft_config = SFTConfig(
    output_dir=OUT,
    num_train_epochs=8,                    # ชุดข้อมูลเล็ก จึงวนหลายรอบ
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,         # effective batch = 1 * 8 = 8
    learning_rate=2e-4,                    # ค่ามาตรฐานสำหรับ LoRA
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=1,                       # ดู loss ทุก step
    eval_strategy="epoch",                 # ประเมินด้วย val ทุก epoch → ดู overfitting
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,                             # ใช้ bf16 (ไม่ต้องมี GradScaler → เสถียรบน Turing)
    fp16=False,
    gradient_checkpointing=False,          # เปิดไปแล้วใน prepare_model_for_kbit_training
    optim="paged_adamw_8bit",
    max_length=1024,                       # ตัดความยาว sequence กัน VRAM พุ่ง
    report_to="none",
)

# ============================================================
# 6) เทรน! (SFTTrainer จัดการ prepare_model_for_kbit_training + ใส่ LoRA ให้)
# ============================================================
trainer = SFTTrainer(
    model=model,               # โมเดลนี้แปะ LoRA + cast fp32 มาแล้ว จึงไม่ส่ง peft_config ซ้ำ
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
)

print(">> start training...")
trainer.train()

# ============================================================
# 7) เซฟ adapter (ไฟล์เล็กแค่หลัก MB — base model ไม่ถูกเซฟซ้ำ)
# ============================================================
final_dir = os.path.join(OUT, "final_adapter")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
print(f">> DONE. adapter saved to: {final_dir}")
