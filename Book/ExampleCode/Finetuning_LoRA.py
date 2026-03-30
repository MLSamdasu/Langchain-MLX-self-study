"""
Ch6.3~6.4 MLX LoRA 파인튜닝 실습
- HuggingFace 데이터셋 다운로드 → JSONL 변환 → mlx_lm.lora 파인튜닝
- 환경: MacBook Pro M4 Pro 48GB, Qwen3.5-9B-4bit
"""

import json
import os
import subprocess
from datasets import load_dataset


# ============================================================
# 1. HuggingFace 데이터셋 다운로드
# ============================================================

# 코드 생성용 데이터셋 (20K개)
ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

# 한국어 지시 따르기 데이터셋을 쓰고 싶으면 아래로 교체
# ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train")

print(f"데이터셋 크기: {len(ds)}개")
print(f"컬럼 구조: {ds.column_names}")
print(f"샘플 1개: {ds[0]}")


# ============================================================
# 2. Chat 형식 변환 함수
# ============================================================

def to_chat_format(example):
    """
    HuggingFace 원본 컬럼 → mlx_lm.lora Chat 형식으로 변환.
    데이터셋마다 컬럼명이 다르므로, 원본 구조 확인 후 매핑 수정 필요.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful coding assistant."
            },
            {
                "role": "user",
                "content": example["prompt"]
            },
            {
                "role": "assistant",
                "content": example["completion"]
            },
        ]
    }


# ============================================================
# 3. train/valid/test 분할 & JSONL 저장
# ============================================================

OUTPUT_DIR = "./finetune_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 80% / 10% / 10% 분할
n = len(ds)
train_ds = ds.select(range(0, int(n * 0.8)))
valid_ds = ds.select(range(int(n * 0.8), int(n * 0.9)))
test_ds = ds.select(range(int(n * 0.9), n))

for name, subset in [
    ("train", train_ds),
    ("valid", valid_ds),
    ("test", test_ds),
]:
    path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for ex in subset:
            # ensure_ascii=False → 한국어가 유니코드 이스케이프 안 되고 그대로 저장됨
            f.write(json.dumps(to_chat_format(ex), ensure_ascii=False) + "\n")
    print(f"{name}: {len(subset)}개 저장 → {path}")


# ============================================================
# 4. mlx_lm.lora 파인튜닝 실행
# ============================================================

# M4 Pro 48GB 권장 설정
FINETUNE_CMD = [
    "mlx_lm.lora",
    "--model", "mlx-community/Qwen3.5-9B-4bit",
    "--data", OUTPUT_DIR,
    "--train",
    "--batch-size", "2",        # VRAM 여유 있으면 4까지 가능
    "--lora-layers", "16",      # 도메인 지식 주입용 (출력 형식만이면 8)
    "--iters", "1000",          # 약 30분 소요 (9B 4-bit 기준)
    "--learning-rate", "1e-5",  # 작을수록 안정, 느리면 3e-5로
    "--adapter-path", "./adapters/code_review",
    "--val-batches", "25",      # 검증 배치 수
    "--save-every", "100",      # 100 iter마다 체크포인트 저장
]

print("\n" + "=" * 60)
print("파인튜닝 실행 명령어:")
print(" ".join(FINETUNE_CMD))
print("=" * 60)

# 실제 실행하려면 아래 주석 해제 (약 30분 소요)
# subprocess.run(FINETUNE_CMD)
