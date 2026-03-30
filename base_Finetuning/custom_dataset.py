import json

# 도메인 전문가의 QA 쌍으로 데이터셋 구축
custom_data = [
    {
        "messages": [
            {"role": "system", "content": "당신은 금융 전문 어시스턴트입니다."},
            {
                "role": "user",
                "content": "PER이란 무엇인가요?",
            },
            {"role": "assistant", "content": "PER은 주가를 주당 순이익으로 나눈 값임"},
        ]
    },
    # ... 이게 최소 200~500개는 있어야됨
]

# 8:1:1 비율이 기본, 분할 저장
n = len(custom_data)
splits = {
    "train": custom_data[: int(n * 0.8)],
    "valid": custom_data[int(n * 0.8) : int(n * 0.9)],
    "test": custom_data[int(n * 0.9) :],
}

for name, data in splits.items():
    with open(f"./finetune_data/{name}.jsonl", "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
