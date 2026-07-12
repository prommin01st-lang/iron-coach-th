"""
รวมไฟล์ตัวอย่าง 7 หมวดจาก scratchpad → train_v2.jsonl + val_v2.jsonl
- validate: แต่ละ example ต้องมี messages ที่มี system/user/assistant และแต่ละ message มีแค่ role/content
- dedupe: ตัดคำถามซ้ำ (ตาม user content)
- แบ่ง val ~12% แบบ deterministic (ไม่สุ่ม เพื่อ reproducible)
"""
import json, os, glob

SCRATCH = "/tmp/claude-1000/-home-petanque-Work-GitHubWork/036d3757-5a61-4023-a123-62ac8a5f8b91/scratchpad"
HERE = os.path.dirname(os.path.abspath(__file__))
SYS = "คุณเป็นโค้ชเวทเทรนนิ่งที่ยึดหลักวิทยาศาสตร์การกีฬา ตอบกระชับ ถูกต้อง และคำนึงถึงความปลอดภัยเสมอ"

all_ex, seen, dropped = [], set(), 0
for f in sorted(glob.glob(os.path.join(SCRATCH, "cat*.json"))):
    with open(f, encoding="utf-8") as fh:
        data = json.load(fh)
    kept = 0
    for ex in data:
        msgs = ex.get("messages", [])
        roles = [m.get("role") for m in msgs]
        # ต้องมีครบ 3 role และแต่ละ message มีแค่ role/content ที่เป็น str ไม่ว่าง
        if roles != ["system", "user", "assistant"]:
            dropped += 1; continue
        clean = [{"role": m["role"], "content": m["content"]} for m in msgs]
        if not all(isinstance(m["content"], str) and m["content"].strip() for m in clean):
            dropped += 1; continue
        clean[0]["content"] = SYS  # normalize system prompt
        key = clean[1]["content"].strip()
        if key in seen:
            dropped += 1; continue
        seen.add(key)
        all_ex.append({"messages": clean}); kept += 1
    print(f"  {os.path.basename(f):22s}: kept {kept}/{len(data)}")

# แบ่ง val: หยิบทุก ๆ ตัวที่ 8 (deterministic) เป็น val ~12%
train, val = [], []
for i, ex in enumerate(all_ex):
    (val if i % 8 == 7 else train).append(ex)

with open(os.path.join(HERE, "train_v2.jsonl"), "w", encoding="utf-8") as fh:
    for ex in train:
        fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
with open(os.path.join(HERE, "val_v2.jsonl"), "w", encoding="utf-8") as fh:
    for ex in val:
        fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\nรวมทั้งหมด: {len(all_ex)} | dropped {dropped}")
print(f"train_v2.jsonl: {len(train)} | val_v2.jsonl: {len(val)}")
