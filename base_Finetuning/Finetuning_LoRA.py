import json
import os
import subprocess  # 파이썬 안에서 터미널 명령어를 실행해줌
from datasets import load_dataset


# ============================================================
# HuggingFace 데이터셋 다운로드
# ========================================

# 코드 생성용 데이터셋
ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

# 한국어 지시 따르기 데이터셋을 쓰고 싶으면 교체
# ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train")

print(f"데이터셋 크기: {len(ds)}개")
print(f"컬럼 구조: {ds.column_names}")
print(f"샘플 1개: {ds[0]}")


# ========================================================
# Chat 형식 변환 함수
# ============================================================
def to_chat_format(example):
    """
    HuggingFace 원본 컬럼 → mlx_lm.lora Chat 형식으로 변환.
    데이터셋마다 컬럼명이 다르므로, 원본 구조 확인 후 매핑 수정 필요.
    """
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }


# ===========================================
# train/valid/test 분할 & JSONL 저장
# ===============================================
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
# mlx_lm.lora 파인튜닝 실행
# =====================================================
# 파이썬 리스트로 만들어서 실행함
FINETUNE_CMD = [
    "mlx_lm.lora",
    "--model",
    "mlx-community/Qwen3.5-9B-4bit",
    "--data",
    OUTPUT_DIR,
    "--train",
    "--batch-size",
    "2",  # VRAM 여유 있으면 증가 가능
    "--lora-layers",
    "16",  # 도메인 지식 주입용 (출력 형식만이면 8)
    "--iters",
    "1000",  # 약 30분 소요 (9B 4-bit 기준)
    "--learning-rate",
    "1e-5",  # 작을수록 안정, 느리면 3e-5로
    "--adapter-path",
    "./adapters/code_review",
    "--val-batches",
    "25",  # 검증 배치 수
    "--save-every",
    "100",  # 100 iter마다 체크포인트 저장
]

print("\n" + "=" * 60)
print("파인튜닝 실행 명령어:")
print(" ".join(FINETUNE_CMD))
print("=" * 60)

subprocess.run(FINETUNE_CMD)

# ============================================
# 파인튜닝 모델 테스트 (loss 평가)
# ==========================================
TEST_CMD = [
    "mlx_lm.lora",
    "--model",
    "mlx-community/Qwen3.5-9B-4bit",
    "--adapter-path",
    "./adapters/code_review",
    "--data",
    OUTPUT_DIR,
    "--test",
]
subprocess.run(TEST_CMD)

# ======================================================
# 어댑터 적용 추론 테스트 (전/후 비교)
# =============================================

PROMPT = "다음 코드를 리뷰해주세요: def add(a,b): return a+b"
# 파인튜닝 전 (어댑터 없이, 범용 모델)
GENERATE_BEFORE_CMD = [
    "mlx_lm.generate",
    "--model",
    "mlx-community/Qwen3.5-9B-4bit",
    "--prompt",
    PROMPT,
]
# . 파인튜닝 후 (어댑터 적용)
GENERATE_AFTER_CMD = [
    "mlx_lm.generate",
    "--model",
    "mlx-community/Qwen3.5-9B-4bit",
    "--adapter-path",
    "./adapters/code_review",
    "--prompt",
    PROMPT,
]
print("\n" + "=" * 60)
print("[파인튜닝 전] 추론 명령어:")
print(" ".join(GENERATE_BEFORE_CMD))
print("\n[파인튜닝 후] 추론 명령어:")
print(" ".join(GENERATE_AFTER_CMD))
print("=" * 60)

print("\n>>> 파인튜닝 전 <<<")
subprocess.run(GENERATE_BEFORE_CMD)
print("\n>>> 파인튜닝 후 <<<")
subprocess.run(GENERATE_AFTER_CMD)


# ==============================
# 파인튜닝된 모델을 서버로 배포
# ===================================================

# 방법 1: 어댑터 분리 서빙
# mlx_lm.server에 --adapter-path만 추가하면 됨
SERVER_CMD = [
    "mlx_lm.server",
    "--model",
    "mlx-community/Qwen3.5-9B-4bit",
    "--adapter-path",
    "./adapters/code_review",
    "--port",
    "8080",
]

# 방법 2: mlx_lm.fuse로 어댑터를 베이스 모델에 병합
# 병합하면 단일 모델이 되어 --adapter-path 없이 사용 가능
FUSE_CMD = [
    "mlx_lm.fuse",
    "--model",
    "mlx-community/Qwen3.5-9B-4bit",
    "--adapter-path",
    "./adapters/code_review",
    "--save-path",
    "./models/code_review_fused",
]

# 병합 후 서빙 (adapter-path 필요 없음)
FUSED_SERVER_CMD = [
    "mlx_lm.server",
    "--model",
    "./models/code_review_fused",
    "--port",
    "8080",
]

print("\n" + "=" * 60)
print("[6.6] 서버 배포 명령어:")
print("  어댑터 서빙:", " ".join(SERVER_CMD))
print("  모델 병합:", " ".join(FUSE_CMD))
print("  병합 모델 서빙:", " ".join(FUSED_SERVER_CMD))
print("=" * 60)

# subprocess.run(FUSE_CMD)        # 병합 먼저
# subprocess.run(FUSED_SERVER_CMD) # 병합 모델로 서버 실행


# ============================================================
# 6.7 LangChain 파이프라인에 파인튜닝 모델 통합
# ============================================================
# 서버가 띄워진 상태에서 LangChain으로 연결
# ChatOpenAI(base_url=...) 패턴 그대로 사용

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 파인튜닝 모델 (어댑터 적용 서버, 포트 8080)
llm_finetuned = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    model="qwen3.5-9b-code-review",  # 식별용 이름
    temperature=0.3,
)

# 파인튜닝 모델 전용 체인 (LCEL 파이프)
code_review_chain = (
    ChatPromptTemplate.from_messages([("user", "다음 코드를 리뷰해주세요:\n{code}")])
    | llm_finetuned
    | StrOutputParser()
)

# 범용 모델은 별도 포트 (8081)에서 서빙
llm_general = ChatOpenAI(
    base_url="http://localhost:8081/v1",
    api_key="not-needed",
    model="qwen3.5-9b-general",
)

# 사용 예시
# result = code_review_chain.invoke({"code": "def add(a,b): return a+b"})
# print(result)


# ================================================
# 멀티 특화모델 오케스트레이션
# ===================================================
# 여러 파인튜닝 모델을 라우터로 자동 분배
# 질문 유형에 따라 최적화된 모델이 자동 선택됨

from langchain_core.runnables import RunnableLambda

# 모델별 인스턴스 (각각 다른 파인튜닝 모델)
models = {
    "code": ChatOpenAI(  # 코드 리뷰 파인튜닝
        base_url="http://localhost:8080/v1", api_key="not-needed", temperature=0.3
    ),
    "qa": ChatOpenAI(  # QA 파인튜닝
        base_url="http://localhost:8081/v1", api_key="not-needed", temperature=0.5
    ),
    "general": ChatOpenAI(  # 범용
        base_url="http://localhost:8082/v1", api_key="not-needed", temperature=0.7
    ),
}

# 라우터: 경량 모델로 질문 카테고리 분류
router_prompt = ChatPromptTemplate.from_template(
    "질문의 카테고리를 하나만 답하세요: " "code, qa, general\n질문: {query}\n카테고리:"
)
router_chain = router_prompt | models["general"] | StrOutputParser()


# 동적 라우팅 함수
def route_to_model(inputs):
    """질문을 분류하고, 해당 특화 모델로 라우팅"""
    category = router_chain.invoke(inputs).strip().lower()
    for key in models:
        if key in category:
            return models[key]
    return models["general"]


# 오케스트레이션 체인
orchestrated = RunnableLambda(
    lambda x: {"model": route_to_model(x), "query": x["query"]}
) | RunnableLambda(
    lambda x: x["model"].invoke(
        ChatPromptTemplate.from_messages([("user", "{query}")]).format_messages(
            query=x["query"]
        )
    )
)

# 사용 예시
# result = orchestrated.invoke({"query": "이 함수를 리뷰해줘: def foo(): pass"})
# print(result.content)  # → code 모델이 자동 선택되어 응답
