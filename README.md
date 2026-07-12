# 🏋️ Fitness Coach — Thai Weight-Training Q&A (Fine-tuned LLM)

โปรเจกต์เรียนรู้การ **fine-tune LLM** เพื่อสร้างผู้ช่วยตอบคำถามเฉพาะทาง: **โค้ชเวทเทรนนิ่งภาษาไทย** ที่ยึดหลักวิทยาศาสตร์การกีฬา ตอบเรื่อง set/rep, ฟอร์มท่า, โภชนาการ และ **ปฏิเสธคำขอที่อันตราย** (สเตียรอยด์, ลดน้ำหนักสุดโต่ง) พร้อมแนะให้พบผู้เชี่ยวชาญเมื่อเหมาะสม

เทรนบน **RTX 2060 (6GB VRAM)** ด้วย **QLoRA (4-bit)** บน Qwen2.5-1.5B-Instruct

## 📊 ผลลัพธ์ (Eval Harness)

วัดด้วยชุดทดสอบ `eval/testset.jsonl` (keyword rubric) — คะแนน PASS rate:

| โมเดล | dataset | คะแนนรวม | จุดเด่น |
|---|---|---|---|
| base Qwen2.5-1.5B | — | ~45% | (ก่อน fine-tune) |
| exp04 | 112 ข้อ | **76%** | ปฏิเสธหนักแน่น + ตอบไทย |
| exp05 | 156 ข้อ | 66% | กัน hallucination (แต่ refusal ลดลง — data imbalance) |

## 🗂️ โครงสร้าง

```
iron-coach-th/
├── ROADMAP.md              # แผนเรียนรู้ 5 เฟส
├── setup/                  # environment (check_gpu.py, requirements.txt)
├── data/                   # dataset (train/val jsonl) + สคริปต์ build
├── experiments/
│   ├── 01_first_finetune/  # Qwen2.5-0.5B (baseline)
│   ├── 02_qwen1.5b/        # ขยับเป็น 1.5B
│   ├── 03_dataset_v2/      # ขยาย dataset 90 ข้อ
│   ├── 04_refusal_thai/    # + refusal + บังคับไทย (best) + DEPLOY.md
│   └── 05_antihalluc/      # + อุดช่องโหว่ hallucination
├── eval/                   # eval harness (testset + run_eval.py)
└── notes/                  # บันทึกโปรเจกต์ + บทเรียน
```

## 🚀 เริ่มใช้งาน (reproduce)

```bash
# 1) ติดตั้ง environment
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r setup/requirements.txt
.venv/bin/python setup/check_gpu.py     # ตรวจ GPU + QLoRA พร้อม

# 2) เทรน (ตัวอย่าง exp04)
.venv/bin/python experiments/04_refusal_thai/train.py

# 3) ประเมินผล
.venv/bin/python eval/run_eval.py

# 4) deploy → Ollama  (ดู experiments/04_refusal_thai/DEPLOY.md)
```

## 🧠 บทเรียนสำคัญ
- **RTX 2060 (Turing) ต้องใช้ bf16 ไม่ใช่ fp16** ไม่งั้น GradScaler crash
- **ขนาดโมเดล + คุณภาพ/ปริมาณ data + system prompt** = คันโยกคุณภาพ
- **วัดเป็นระบบ (eval) พลิกข้อสรุปจาก vibe-check ได้**
- **Data balance สำคัญ** — เพิ่มข้อมูลหมวดหนึ่งมากไป ทำอีกหมวดถดถอย (catastrophic forgetting)
- **Deploy Qwen2.5 → Ollama:** ต้อง untie embeddings + แปลง GGUF ด้วย llama.cpp (Ollama converter ในตัวแปลงพลาด)

## ⚠️ Disclaimer
โมเดลนี้ให้คำแนะนำการออกกำลังกายทั่วไปเพื่อการศึกษา **ไม่ใช่คำแนะนำทางการแพทย์** และยังมีข้อจำกัด (โมเดลเล็ก 1.5B อาจตอบผิด/มั่วได้บ้าง) ควรปรึกษาผู้เชี่ยวชาญก่อนเริ่มโปรแกรมฝึกจริง

## License
โค้ด: MIT · โมเดลฐาน Qwen2.5-1.5B-Instruct อยู่ภายใต้ Qwen License
