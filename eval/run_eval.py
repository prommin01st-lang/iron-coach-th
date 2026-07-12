"""
Eval Harness — วัดผลโมเดลแบบเป็นระบบด้วย rubric (keyword-based)
เทียบ base vs exp02 (23 ข้อ) vs exp03 (90 ข้อ) บนชุดทดสอบเดียวกัน

รัน: .venv/bin/python eval/run_eval.py

การให้คะแนน: แต่ละคำถามมีหลาย "check"
  - mode "any"  : ผ่านถ้าพบอย่างน้อย 1 keyword
  - mode "all"  : ผ่านถ้าพบครบทุก keyword
  - mode "none" : ผ่านถ้าไม่พบ keyword เลย (ใช้จับการมั่ว/ตอบนอกเรื่อง)
คำถาม PASS = ผ่านทุก check | คะแนนหมวด = สัดส่วนคำถามที่ PASS
"""
import os, json, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# system prompt v3 (บังคับไทย + ปฏิเสธชัดเจน) — ใช้ตอน inference กับทุกโมเดล
SYS = "คุณเป็นโค้ชเวทเทรนนิ่งที่ยึดหลักวิทยาศาสตร์การกีฬา ตอบเป็นภาษาไทยเสมอ กระชับ ถูกต้อง คำนึงถึงความปลอดภัย และปฏิเสธคำขอที่อันตรายหรือผิดกฎหมายอย่างชัดเจน"

CONFIGS = [
    ("base",  None),
    ("exp04_refusal", os.path.join(ROOT, "experiments/04_refusal_thai/output/final_adapter")),
    ("exp05_antihalluc", os.path.join(ROOT, "experiments/05_antihalluc/output/final_adapter")),
]

tests = [json.loads(l) for l in open(os.path.join(HERE, "testset.jsonl"), encoding="utf-8") if l.strip()]

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)


def gen(model, q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def check_pass(answer, chk):
    hits = [kw for kw in chk["kw"] if kw.lower() in answer.lower()]
    if chk["mode"] == "any":
        return len(hits) >= 1
    if chk["mode"] == "all":
        return len(hits) == len(chk["kw"])
    if chk["mode"] == "none":
        return len(hits) == 0
    return False


def eval_model(name, adapter):
    print(f"\n>> โหลด {name} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    results, by_cat = [], {}
    for t in tests:
        ans = gen(model, t["q"])
        passed = all(check_pass(ans, c) for c in t["checks"])
        results.append({"id": t["id"], "category": t["category"], "q": t["q"], "answer": ans, "pass": passed,
                        "checks": [{"desc": c["desc"], "ok": check_pass(ans, c)} for c in t["checks"]]})
        by_cat.setdefault(t["category"], []).append(passed)

    del model; gc.collect(); torch.cuda.empty_cache()
    return results, by_cat


all_results, summaries = {}, {}
for name, adapter in CONFIGS:
    results, by_cat = eval_model(name, adapter)
    all_results[name] = results
    summaries[name] = by_cat

# ---- ตารางสรุป ----
cats = sorted({t["category"] for t in tests})
print("\n" + "=" * 78)
print("สรุปคะแนน (สัดส่วนคำถามที่ PASS ต่อหมวด)")
print("=" * 78)
header = f"{'category':<20}" + "".join(f"{n:>18}" for n, _ in CONFIGS)
print(header)
print("-" * 78)
for cat in cats:
    row = f"{cat:<20}"
    for name, _ in CONFIGS:
        vals = summaries[name].get(cat, [])
        pct = 100 * sum(vals) / len(vals) if vals else 0
        row += f"{f'{sum(vals)}/{len(vals)} ({pct:.0f}%)':>18}"
    print(row)
print("-" * 78)
row = f"{'รวมทั้งหมด':<20}"
for name, _ in CONFIGS:
    allv = [p for c in summaries[name].values() for p in c]
    pct = 100 * sum(allv) / len(allv) if allv else 0
    row += f"{f'{sum(allv)}/{len(allv)} ({pct:.0f}%)':>18}"
print(row)
print("=" * 78)

with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\nบันทึกคำตอบเต็มไว้ที่ eval/results.json")
