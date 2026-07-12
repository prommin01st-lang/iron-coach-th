"""
อัป LoRA adapter ขึ้น Hugging Face Hub

เตรียมก่อนรัน (ครั้งเดียว):
  1) สร้างบัญชี https://huggingface.co  แล้วสร้าง Access Token (write) ที่ Settings > Access Tokens
  2) ล็อกอิน:  .venv/bin/huggingface-cli login    (วาง token)

รัน:
  # อัป adapter ตัว exp04 (ค่าเริ่มต้น) ไปที่ repo ของคุณ
  .venv/bin/python upload_to_hf.py --repo <your-username>/fitness-coach-th

  # หรือเลือก adapter อื่น / อัปแบบ private
  .venv/bin/python upload_to_hf.py --repo <user>/fitness-coach-th \\
        --adapter experiments/05_antihalluc/output/final_adapter --private

  # อัปไฟล์ GGUF (สำหรับ Ollama/llama.cpp) ไป repo แยก
  .venv/bin/python upload_to_hf.py --repo <user>/fitness-coach-th-gguf --gguf /path/to/fitness-coach-f16.gguf
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="เช่น yourname/fitness-coach-th")
    ap.add_argument("--adapter", default=os.path.join(HERE, "experiments/04_refusal_thai/output/final_adapter"),
                    help="โฟลเดอร์ adapter ที่จะอัป (ค่าเริ่มต้น = exp04)")
    ap.add_argument("--gguf", default=None, help="อัปไฟล์ .gguf แทน adapter")
    ap.add_argument("--private", action="store_true", help="ทำ repo แบบ private")
    ap.add_argument("--card", default=os.path.join(HERE, "MODEL_CARD.md"), help="ไฟล์ model card (README บน HF)")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        sys.exit("ไม่พบ huggingface_hub — ติดตั้ง: .venv/bin/pip install huggingface_hub")

    api = HfApi()
    # ตรวจว่าล็อกอินแล้ว
    try:
        who = api.whoami()
        print(f">> logged in as: {who['name']}")
    except Exception:
        sys.exit("ยังไม่ได้ล็อกอิน — รัน: .venv/bin/huggingface-cli login ก่อน")

    print(f">> สร้าง/ตรวจ repo: {args.repo} (private={args.private})")
    create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")

    if args.gguf:
        if not os.path.isfile(args.gguf):
            sys.exit(f"ไม่พบไฟล์ GGUF: {args.gguf}")
        print(f">> อัป GGUF: {args.gguf}")
        api.upload_file(path_or_fileobj=args.gguf, path_in_repo=os.path.basename(args.gguf), repo_id=args.repo)
    else:
        if not os.path.isdir(args.adapter):
            sys.exit(f"ไม่พบโฟลเดอร์ adapter: {args.adapter}")
        print(f">> อัป adapter จาก: {args.adapter}")
        api.upload_folder(folder_path=args.adapter, repo_id=args.repo,
                          ignore_patterns=["checkpoint-*", "optimizer.pt", "*.log"])

    # อัป model card เป็น README.md
    if os.path.isfile(args.card):
        print(">> อัป model card (README.md)")
        api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md", repo_id=args.repo)

    print(f"\n✅ เสร็จ! ดูได้ที่: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
