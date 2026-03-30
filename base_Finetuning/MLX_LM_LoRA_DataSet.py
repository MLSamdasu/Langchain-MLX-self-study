"""
실제 데이터 셋은 mlx_lm.lora에서 받아들이는 데이터셋과 다른 경우가 많음
HuggingFace에 올라와 있는 공개 데이터셋은 각자 고유의 컬럼명과 구조를 가지고 있음
이것을 다운로드 -> 변환 -> JSONL 저장 하는 방식임.
"""

from datasets import load_dataset
import json, os

# HuggingFace에서 한국어 QA 데이터셋 로드
ds = load_dataset("heegyu/kowikitext", split="train")

# 다른 방식 (코드 관련 데이터셋)
ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")


# Chat 형식으로 변환
# example은 데이터셋의 한 행(row)
# 원본 컬럼을 Chat형식 user, assistant로 매핑
# 이 함수를 데이터셋에 맞게 수정해야함
def to_chat_format(example):
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful codding assistant."},
            {"role": "user", "content": example["prompt"]},  # 원본 컬럼명에 맞춰야함
            {"role": "assistant", "content": example["completion"]},
        ]
    }


# JSONL로 저장
# Json과 다르게 한줄이 하나의 JSON객체이므로 한줄씩 사용 가능함
# 메모리에 한번에 데이터셋을 올리지 않아도 됨
# train, valid, test 분할
os.makedirs("./finetune_data", exist_ok=True)
# ds.select -> df.iloc[]과 같음
train_ds = ds.select(range(0, 16000))  # 80%
valid_ds = ds.select(range(16000, 18000))  # 10%
test_ds = ds.select(range(18000, 20000))  # 10%

for name, subset in [
    ("train", train_ds),
    ("valid", valid_ds),
    ("test", test_ds),
]:
    path = f"./finetune_data/{name}.jsonl"
    with open(path, "w") as f:
        for ex in subset:
            # ensure_ascii -> False로 해야 한글이 안깨짐
            f.write(
                json.dumps(to_chat_format(ex), ensure_ascii=False) + "\n"
            )  # '\n'이 JSONL을 만드는 핵심
    print(f"{name}: {len(subset)}개 저장")
