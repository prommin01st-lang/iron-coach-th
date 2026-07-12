# 🚀 Deploy exp04 → Ollama

โมเดล fine-tuned พร้อมใช้ใน Ollama ชื่อ **`fitness-coach`**

## วิธีใช้งาน (เปิดในเทอร์มินัลของคุณ)
```bash
ollama run fitness-coach
```
พิมพ์คำถามภาษาไทยได้เลย เช่น "อยากเพิ่มความแข็งแรง Deadlift จัดโปรแกรมยังไง"
(ออกจากแชตพิมพ์ `/bye`)

หรือเรียกผ่าน API:
```bash
curl http://localhost:11434/api/chat -d '{"model":"fitness-coach","messages":[{"role":"user","content":"..."}],"stream":false}'
```

## ขั้นตอนที่ทำ (reproducible)
1. **merge** adapter เข้า base fp16 → `merged/`  (`merge.py`)
2. **untie embeddings** → `merged_untied/` (Qwen2.5 ใช้ tied embeddings ต้องแยก lm_head ออกมา)
3. **แปลง GGUF** ด้วย llama.cpp converter (ไม่ใช้ Ollama ในตัว):
   ```bash
   python llama.cpp/convert_hf_to_gguf.py merged_untied --outfile fitness-coach-f16.gguf --outtype f16
   ```
4. **import** เข้า Ollama: `ollama create fitness-coach -f Modelfile` (FROM ...gguf)

## ⚠️ บทเรียนสำคัญ (gotcha)
Ollama **converter ในตัวแปลง Qwen2.5 พลาด** → โมเดลตอบเป็น "???" ทั้งหมด
- สาเหตุ: Qwen2.5 มี `tie_word_embeddings=True` (lm_head แชร์น้ำหนักกับ embedding, ไม่มี lm_head.weight แยกในไฟล์) → Ollama converter จับ vocab/tokenizer ผิด (เห็น `</s>` แบบ Llama)
- **วิธีแก้:** untie แล้ว materialize `lm_head.weight` + แปลง GGUF ด้วย **llama.cpp** (แม่นกว่า) แล้วค่อย import GGUF เข้า Ollama

## หมายเหตุคุณภาพ
โมเดล 1.5B + dataset 112 ข้อ ยังมี artifact เล็กน้อย (บางครั้งมีตัวอักษรจีนปน, out-of-scope redirect ไม่สวย) ตรงกับผล eval 76% — ถ้าอยากดีขึ้น: เพิ่ม dataset หลักร้อย + rebalance out-of-scope
