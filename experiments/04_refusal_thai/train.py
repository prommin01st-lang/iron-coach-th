"""
เฟส 3 (ต่อ): exp04 — แก้จุดอ่อน refusal + บังคับตอบไทย
- dataset v3 (v2 + refusal เพิ่ม, system prompt ใหม่บังคับไทย+ปฏิเสธ)
- Qwen2.5-1.5B + QLoRA bf16 + early stopping (เหมือน exp03)

รัน: .venv/bin/python experiments/04_refusal_thai/train.py
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
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DT = torch.bfloat16

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=DT, bnb_4bit_use_double_quant=True)
print(">> loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", dtype=DT)
model.config.use_cache = False

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                        gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, lora)
model.print_trainable_parameters()

print(">> loading dataset v3...")
train_ds = load_dataset("json", data_files=os.path.join(DATA, "train_v3.jsonl"), split="train")
val_ds = load_dataset("json", data_files=os.path.join(DATA, "val_v3.jsonl"), split="train")
print(f"   train={len(train_ds)} val={len(val_ds)}")

cfg = SFTConfig(
    output_dir=OUT, num_train_epochs=8,
    per_device_train_batch_size=1, gradient_accumulation_steps=8,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
    logging_steps=2, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    bf16=True, fp16=False, gradient_checkpointing=False, optim="paged_adamw_8bit",
    max_length=1024, report_to="none",
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds,
                     processing_class=tok, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
print(">> start training...")
trainer.train()
final_dir = os.path.join(OUT, "final_adapter")
trainer.save_model(final_dir); tok.save_pretrained(final_dir)
print(f">> DONE. adapter saved to: {final_dir}")
