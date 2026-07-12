# 🎓 Roadmap: การเรียนรู้ Fine-Tuning โมเดล

> ปรับให้เข้ากับเครื่องของคุณ: **RTX 2060 Mobile (6GB VRAM)** · Ryzen 7 4800H · 31GB RAM
>
> เป้าหมาย: เข้าใจและทำ fine-tune โมเดลสำหรับงานเฉพาะทางได้จริง โดยเน้นเทคนิคประหยัด VRAM

---

## 🧭 ภาพรวม 5 เฟส

| เฟส | หัวข้อ | เวลาโดยประมาณ | ผลลัพธ์ |
|---|---|---|---|
| 0 | เตรียมเครื่อง (Environment) | 1 วัน | เทรนบน GPU ได้ |
| 1 | พื้นฐานที่ต้องรู้ | 3–5 วัน | เข้าใจศัพท์และหลักการ |
| 2 | Fine-tune ครั้งแรก (hands-on) | 3–5 วัน | โมเดลที่ fine-tune เองสำเร็จ 1 ตัว |
| 3 | เจาะลึกเทคนิค | 1–2 สัปดาห์ | เลือกเทคนิค/พารามิเตอร์ได้เอง |
| 4 | โปรเจกต์งานเฉพาะทางจริง | ต่อเนื่อง | โมเดลใช้งานจริงในงานคุณ |

---

## ⚠️ กฎเหล็กสำหรับ 6GB VRAM (อ่านก่อน)

1. **อย่าเทรน LLM จากศูนย์** — เป็นไปไม่ได้บนเครื่องนี้ ใช้ pre-trained model เสมอ
2. **ใช้ QLoRA** (4-bit) แทน full fine-tuning — ลด VRAM ได้ ~4–8 เท่า
3. **โมเดลที่ทำได้จริง**: ขนาด **0.5B – 3B parameters** (เช่น Qwen 2.5 0.5B/1.5B/3B, Llama 3.2 1B/3B, Gemma 2 2B)
4. **batch size เล็ก** (1–2) + **gradient accumulation** ชดเชย
5. เปิด **gradient checkpointing** และ **mixed precision (fp16/bf16)** เสมอ

---

## 📦 เฟส 0 — เตรียม Environment ✅ เสร็จแล้ว

- [x] ติดตั้ง `venv` แยกสภาพแวดล้อม (`.venv/`)
- [x] ติดตั้ง PyTorch 2.6.0 + CUDA 12.4 — `torch.cuda.is_available()` = True
- [x] ติดตั้ง `transformers`, `peft`, `bitsandbytes`, `datasets`, `accelerate`, `trl`
- [x] ทดสอบ GPU matmul + bitsandbytes 4-bit (QLoRA) → ผ่านทั้งคู่
- [ ] (ทางเลือก) ติดตั้ง **Unsloth** ภายหลัง — เร็ว/ประหยัด VRAM แต่ stack ปัจจุบันพอเริ่มได้แล้ว

📄 รันตรวจสภาพได้ทุกเมื่อ: `.venv/bin/python setup/check_gpu.py`

---

## 📚 เฟส 1 — พื้นฐานที่ต้องรู้

- [ ] **Fine-tuning คืออะไร** ต่างจาก pre-training / RAG / prompting อย่างไร
- [ ] **Transfer learning** — ทำไมถึงไม่ต้องเทรนจากศูนย์
- [ ] ศัพท์หลัก: epoch, batch size, learning rate, loss, overfitting
- [ ] **Tokenizer** — ข้อความกลายเป็นตัวเลขได้ยังไง
- [ ] **LoRA / QLoRA** — เทรนแค่บางส่วนของโมเดลได้ยังไง (หัวใจของการประหยัด VRAM)
- [ ] **Quantization** (4-bit / 8-bit) — ย่อโมเดลให้เข้า VRAM
- [ ] รูปแบบข้อมูลเทรน: instruction / chat format (เช่น `{"instruction", "input", "output"}`)

💡 *เมื่อไหร่ควร fine-tune vs. ใช้ RAG?* — ถ้าต้องการ "ความรู้ใหม่" มักใช้ RAG ดีกว่า; ถ้าต้องการ "พฤติกรรม/สไตล์/รูปแบบคำตอบเฉพาะทาง" ใช้ fine-tune

---

## 🛠️ เฟส 2 — Fine-tune ครั้งแรก (Hands-on)

เป้าหมาย: fine-tune โมเดลเล็กด้วย dataset ตัวอย่าง ให้ครบ loop

- [ ] โหลด base model เล็ก (แนะนำ **Qwen2.5-0.5B** — เบาสุด เหมาะเริ่มต้น)
- [ ] โหลด dataset สำเร็จรูปจาก Hugging Face (เช่น `databricks/databricks-dolly-15k`)
- [ ] ตั้งค่า QLoRA + training arguments
- [ ] เทรน แล้วดู loss ลดลง
- [ ] ทดสอบ inference ก่อน/หลัง fine-tune เทียบกัน
- [ ] เซฟ adapter (LoRA weights) — ไฟล์เล็กแค่หลัก MB

📄 โค้ดตัวอย่างจะอยู่ใน `experiments/01_first_finetune/`

---

## 🔬 เฟส 3 — เจาะลึกเทคนิค

- [ ] จัดการ **dataset ของตัวเอง** — ทำความสะอาด, แบ่ง train/val/test, จัด format
- [ ] ปรับ **hyperparameters**: learning rate, LoRA rank (r), alpha, epochs
- [ ] **ประเมินผล**: loss, perplexity, และการวัดเชิงคุณภาพ
- [ ] แก้ **overfitting / underfitting**
- [ ] **จัดการ VRAM หมด (OOM)**: ลด batch, ลด seq length, gradient accumulation
- [ ] **Merge adapter** เข้า base model + **export เป็น GGUF** เพื่อรันด้วย Ollama/llama.cpp
- [ ] เทคนิคเสริม: gradient checkpointing, packing, flash attention

---

## 🚀 เฟส 4 — โปรเจกต์งานเฉพาะทางจริง

- [ ] นิยามงานให้ชัด (input → output ที่ต้องการ)
- [ ] รวบรวม/สร้าง dataset ของงานจริง (คุณภาพ > ปริมาณ)
- [ ] เลือก base model ให้เหมาะกับงาน + ภาษา (ถ้างานภาษาไทย เลือกโมเดลที่รองรับไทยดี)
- [ ] เทรน → ประเมิน → ปรับ (วนซ้ำ)
- [ ] Deploy: รัน inference บนเครื่อง (Ollama / vLLM / transformers)

---

## 📖 แหล่งเรียนรู้แนะนำ

- Hugging Face — NLP Course & PEFT docs
- Unsloth — Notebooks & Documentation (เน้น VRAM ต่ำ)
- เอกสาร QLoRA (paper: "QLoRA: Efficient Finetuning of Quantized LLMs")

---

## ✅ ขั้นต่อไป

1. ยืนยัน **ประเภทงานเฉพาะทาง** ของคุณ (ข้อความ/รูป/ตาราง + ภาษา + มีข้อมูลไหม)
2. ให้ผมช่วยทำ **เฟส 0 (ติดตั้ง environment)** ให้พร้อมก่อน
3. ลุยเฟส 2 — fine-tune ตัวแรกด้วยกัน

> อัปเดตความคืบหน้าโดยติ๊ก `[x]` ในไฟล์นี้ได้เลย
