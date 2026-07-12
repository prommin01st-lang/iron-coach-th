# 🏋️ Project: บอทแนะนำการฝึกเวทเทรนนิ่ง (Specialized Q&A Assistant)

## เป้าหมาย
เปลี่ยนโมเดลภาษาทั่วไป → ผู้เชี่ยวชาญโดเมนแคบ: **การฝึกเวทเทรนนิ่ง**
ตอบคำถาม / จัดตารางความเข้มข้นการฝึกได้ **แม่นยำและปลอดภัย** โดยยึดหลักวิทยาศาสตร์การกีฬา

## ขอบเขตงาน (Scope)
- แนะนำ Sets / Reps / ความเข้มข้น (% 1RM) ตามเป้าหมาย (strength / hypertrophy / endurance)
- เทคนิคการจัดระเบียบร่างกาย (form) สำหรับท่าเฉพาะ เช่น Deadlift, Squat, Bench Press
- จัดตารางฝึก (programming) และหลัก progressive overload / deload

## หัวใจของโปรเจกต์: กัน Hallucination
เป้าหมายการเรียนรู้หลัก = ทำให้โมเดล **ไม่มั่ว** และ **ยึดหลักการที่เราสอนเท่านั้น**

กลยุทธ์ที่จะใช้ (เรียนทีละอย่าง):
1. **Dataset คุณภาพสูง** — คำตอบอ้างอิงหลักวิทยาศาสตร์การกีฬา ไม่มีข้อมูลมั่ว
2. **สอนให้ตอบ "ไม่รู้"** — ใส่ตัวอย่างที่โมเดลควรปฏิเสธ/แนะนำให้ปรึกษาผู้เชี่ยวชาญ
3. **Safety guardrails** — คำเตือนเรื่องอาการบาดเจ็บ, น้ำหนักที่เหมาะกับระดับผู้ฝึก
4. **(ขั้นสูง) RAG** — ให้โมเดลดึงคำตอบจากเอกสารจริง แทนการจำ เพื่อลดการมั่ว
5. **การประเมิน** — วัดว่าโมเดลมั่วมากน้อยแค่ไหน (factual accuracy)

## ⚠️ ข้อควรระวังด้านความปลอดภัย (สำคัญมากสำหรับโดเมนนี้)
- นี่คือคำแนะนำด้านสุขภาพ/การออกกำลังกาย → โมเดลต้อง **แนะนำให้ปรึกษาผู้เชี่ยวชาญ/แพทย์** เมื่อเหมาะสม
- ต้องระบุเสมอว่าเป็นคำแนะนำทั่วไป ไม่ใช่คำวินิจฉัยทางการแพทย์
- ระวังคำแนะนำน้ำหนัก/ความเข้มข้นที่อาจทำให้บาดเจ็บ

## แผนเทคนิค (เข้ากับ RTX 2060 6GB)
- Base model: **Qwen2.5-1.5B-Instruct** หรือ **Llama-3.2-3B-Instruct** (เริ่มที่ตัวเล็กก่อน)
- วิธี: **QLoRA** (4-bit) + SFTTrainer (trl)
- Data format: chat/instruction — `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`
- Deploy: merge adapter → export GGUF → รันด้วย Ollama

## สถานะ
- [x] เลือกโปรเจกต์
- [x] เฟส 0: environment (PyTorch CUDA + QLoRA พร้อมใช้งาน)
- [x] รวบรวม/ออกแบบ dataset ตัวอย่าง (23 train + 4 val ใน data/)
- [x] fine-tune รอบแรก — Qwen2.5-0.5B + QLoRA (bf16) สำเร็จ! adapter อยู่ที่ experiments/01_first_finetune/output/final_adapter
- [x] เฟส 3 (ส่วนขยายโมเดล): เทรน Qwen2.5-1.5B สำเร็จ (experiments/02_qwen1.5b/) + early stopping — best eval_loss 1.07 (จาก 0.5B ที่ 1.29). safety-refusal ถ่ายทอดสำเร็จ, ตัวเลข set/rep ถูกต้อง, ภาษาไทยลื่นขึ้นมาก
- [x] เฟส 3 (ต่อ): ขยาย dataset 23 → 90 ข้อ (7 หมวด, ใช้ 7 subagents สร้างขนาน) เทรน exp 03_dataset_v2 พร้อม early stopping
- [x] เฟส 3 (ต่อ): สร้าง eval harness (eval/testset.jsonl 25 ข้อ + run_eval.py rubric keyword) เทียบ base/exp02/exp03
- [x] เฟส 3 (ต่อ): แก้จุดอ่อน — exp04 (dataset v3 = 112 ข้อ, +22 refusal, system prompt ใหม่บังคับไทย+ปฏิเสธ) เทรนสำเร็จ best eval_loss 1.165
- [ ] เฟส 3 (ต่อ): rebalance out_of_scope / dataset หลักร้อย / LLM-judge / 3B / deploy

## บันทึกผล exp04 — แก้จุดอ่อน refusal+ไทย (2026-07-12)
Eval (system prompt v3, เกณฑ์เดียวกัน): base 44% | exp03 56% | **exp04 76% ชนะ**
- **สำเร็จตามเป้า:** refusal 50→100%, safety_injury 50→100%, recovery 50→100%, form 25→75% (บังคับไทยลด language drift)
- regression: out_of_scope 67→33% (refusal data เยอะทำโมเดลจริงจังไป → รอบหน้า rebalance)
- **บทเรียน capstone: วงจรพัฒนาด้วยข้อมูลครบ** — eval เจอจุดอ่อน → แก้ตรงจุด (เพิ่ม data + system prompt) → eval ยืนยันดีขึ้น (56→76%)
- ไฟล์: data/build_dataset_v3.py, data/train_v3.jsonl (98)/val_v3.jsonl (14), experiments/04_refusal_thai/, eval/run_eval.py (อัปเดต config+rubric)

## บันทึกผล Eval Harness (2026-07-12)
คะแนนรวม (PASS 25 ข้อ): base 36% | exp02 (23ข้อ) 48% | **exp03 (90ข้อ) 68% ชนะ**
- exp03 ดีขึ้นชัด: safety_injury 0→100%, safety_medical →100%, programming 0→60%, out_of_scope 33→67%
- **บทเรียนใหญ่: systematic eval พลิกข้อสรุปจาก vibe check** — ตอน compare 4 ข้อคิดว่า exp03 แย่ลง แต่วัด 25 ข้อจริงคือดีสุด
- จุดอ่อนที่ eval เผย: (1) refusal ไม่ robust — คำถามต่างนิดเดียว exp03 ปฏิเสธหนัก vs อ่อน; refuse02 ไม่ push back เป้าหมายอันตราย (2) form ตกเพราะ language drift ตอบอังกฤษ (3) rubric เองไม่สมบูรณ์ (ไม่มี synonym "เสี่ยง", จับอังกฤษไม่ได้) → eval design ต้อง iterate
- ผลเต็มใน eval/results.json

## บันทึกผล dataset v2 (2026-07-12)
- 90 ตัวอย่าง (train 79 / val 11), 7 หมวด รวม safety-refusal 16 + out-of-scope 8. สร้างด้วย 7 subagents ขนาน + build_dataset_v2.py (validate/dedupe)
- early stopping ทำงานตามตำรา: eval_loss 1.42→1.29→**1.27(best e3)**→1.31→1.37 หยุด e5 คืน checkpoint e3
- ผลเทียบ vs base: refusal สเตียรอยด์ดีขึ้นชัด, safety framing (หยุด+พบแพทย์) ถ่ายทอด — แต่ **ไม่ใช่ชัยชนะขาดลอย**: ยังมั่วรายละเอียด (เช่น "เลือดไหลกลับสมอง"), คำตอบ Barbell Row ถดถอยกว่า exp 02 (สับสนกลุ่มกล้าม), out-of-scope ไม่ redirect ตามสอน
- บทเรียน: เพิ่มข้อมูล 23→90 ไม่พอ ~11 ข้อ/หมวดยังบาง; ต้องหลักร้อย+ / เพิ่มความสม่ำเสมอ / eval เป็นระบบ; greedy+4bit เพิ่ม artifact
- [ ] ประเมิน hallucination อย่างเป็นระบบ
- [ ] ปรับปรุง + deploy (merge → GGUF → Ollama)

## บันทึกผลเฟส 2 (2026-07-12)
- เทรน 8 epochs, 24 steps, ~2 นาที บน RTX 2060. train_loss 2.9→0.9, eval_loss 2.34→1.29 (ไม่ overfit)
- ผลเทียบ: base มั่วสิ้นดี → fine-tuned เข้าเรื่องเวทเทรนนิ่ง + generalize ท่าใหม่ (Barbell Row) ได้
- ข้อจำกัด: คำตอบยังหยาบ ตัวเลขมั่ว safety-refusal ไม่ถ่ายทอด → สาเหตุคือโมเดลเล็ก (0.5B อ่อนไทย) + ข้อมูลน้อย ไม่ใช่ overfitting
- บทเรียนเทคนิค: RTX 2060 (Turing) ต้องใช้ **bf16 ไม่ใช่ fp16** ไม่งั้น GradScaler crash ("_amp_foreach_non_finite_check_and_unscale not implemented for BFloat16")
