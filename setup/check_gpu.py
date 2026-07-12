"""ทดสอบว่า environment พร้อมเทรนบน GPU หรือยัง — รันหลังติดตั้งเสร็จ"""

def main():
    print("=" * 50)
    print("  GPU / Environment Health Check")
    print("=" * 50)

    import torch
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA (torch)    : {torch.version.cuda}")
        print(f"GPU name        : {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute cap     : {cap[0]}.{cap[1]}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM total      : {total:.1f} GB")
        # ทดสอบคำนวณจริงบน GPU
        x = torch.randn(1000, 1000, device="cuda")
        y = (x @ x).sum().item()
        print(f"GPU matmul test : OK ({y:.1f})")
    else:
        print("!! GPU ใช้ไม่ได้ — จะเทรนได้ช้ามากบน CPU")

    # เช็ค libs หลัก
    print("-" * 50)
    for mod in ["transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"]:
        try:
            m = __import__(mod)
            print(f"{mod:14s}: {getattr(m, '__version__', 'ok')}")
        except Exception as e:
            print(f"{mod:14s}: !! MISSING ({e})")
    print("=" * 50)


if __name__ == "__main__":
    main()
