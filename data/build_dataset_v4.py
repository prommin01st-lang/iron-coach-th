"""
สร้าง dataset v4 = หมวดเดิมทั้งหมด (cat1-7) + refusal เพิ่ม (cat8)
ความต่างจาก v2:
  1) system prompt ใหม่ — บังคับ "ตอบเป็นภาษาไทยเสมอ" + สั่งปฏิเสธคำขออันตรายชัดเจน
     (normalize ทุก example ให้ใช้ prompt เดียวกัน โมเดลจะเรียนผูกพฤติกรรมกับ prompt นี้)
  2) มี refusal examples หลากหลายขึ้น (robustness)
"""
import json, os, glob

SCRATCH = "/tmp/claude-1000/-home-petanque-Work-GitHubWork/036d3757-5a61-4023-a123-62ac8a5f8b91/scratchpad"
HERE = os.path.dirname(os.path.abspath(__file__))
# system prompt ใหม่ (v4)
SYS = "คุณเป็นโค้ชเวทเทรนนิ่งที่ยึดหลักวิทยาศาสตร์การกีฬา ตอบเป็นภาษาไทยเสมอ กระชับ ถูกต้อง คำนึงถึงความปลอดภัย และปฏิเสธคำขอที่อันตรายหรือผิดกฎหมายอย่างชัดเจน"

all_ex, seen, dropped = [], set(), 0
for f in sorted(glob.glob(os.path.join(SCRATCH, "cat*.json"))):
    with open(f, encoding="utf-8") as fh:
        data = json.load(fh)
    kept = 0
    for ex in data:
        msgs = ex.get("messages", [])
        if [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
            dropped += 1; continue
        clean = [{"role": m["role"], "content": m["content"]} for m in msgs]
        if not all(isinstance(m["content"], str) and m["content"].strip() for m in clean):
            dropped += 1; continue
        clean[0]["content"] = SYS  # normalize เป็น system prompt v4
        key = clean[1]["content"].strip()
        if key in seen:
            dropped += 1; continue
        seen.add(key)
        all_ex.append({"messages": clean}); kept += 1
    print(f"  {os.path.basename(f):26s}: kept {kept}/{len(data)}")

train, val = [], []
for i, ex in enumerate(all_ex):
    (val if i % 8 == 7 else train).append(ex)

with open(os.path.join(HERE, "train_v4.jsonl"), "w", encoding="utf-8") as fh:
    for ex in train:
        fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
with open(os.path.join(HERE, "val_v4.jsonl"), "w", encoding="utf-8") as fh:
    for ex in val:
        fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\nรวม {len(all_ex)} | dropped {dropped} | train {len(train)} | val {len(val)}")
