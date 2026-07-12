"""
เฟส 3: fine-tune ด้วยโมเดลใหญ่ขึ้น Qwen2.5-1.5B-Instruct (จาก 0.5B)
- ใช้ dataset เดิม + bf16 (บทเรียนจากเฟส 2: Turing ต้องใช้ bf16)
- เพิ่ม early stopping: เก็บ checkpoint ที่ eval_loss ดีที่สุด (กัน overfit)

รันจากโฟลเดอร์ Model Trainer:
    .venv/bin/python experiments/02_qwen1.5b/train.py
"""
import os
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(HERE, "output")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"     # <<< เปลี่ยนจาก 0.5B เป็น 1.5B
COMPUTE_DTYPE = torch.bfloat16

# 1) โหลด base model แบบ 4-bit (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
)

print(">> loading tokenizer & model (4-bit)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto", dtype=COMPUTE_DTYPE,
)
model.config.use_cache = False

# 2) LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3) dataset
print(">> loading dataset...")
train_ds = load_dataset("json", data_files=os.path.join(DATA, "train.jsonl"), split="train")
val_ds = load_dataset("json", data_files=os.path.join(DATA, "val.jsonl"), split="train")
print(f"   train={len(train_ds)} val={len(val_ds)}")

# 4) training config + early stopping (เก็บ checkpoint ที่ eval_loss ต่ำสุด)
sft_config = SFTConfig(
    output_dir=OUT,
    num_train_epochs=10,                   # ตั้งเผื่อไว้ แต่ early stopping จะหยุดเองถ้า eval แย่ลง
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,           # โหลด checkpoint ที่ดีที่สุดกลับมาตอนจบ
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    fp16=False,
    gradient_checkpointing=False,
    optim="paged_adamw_8bit",
    max_length=1024,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    # หยุดเทรนถ้า eval_loss ไม่ดีขึ้น 2 epochs ติด (early stopping)
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print(">> start training...")
trainer.train()

final_dir = os.path.join(OUT, "final_adapter")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
print(f">> DONE. adapter saved to: {final_dir}")
