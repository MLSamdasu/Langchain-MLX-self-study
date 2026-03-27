### 파인튜닝이 필요한 이유
- 프롬프트 엔지니어링의 한계를 넘기 위해 -> 해야하는 판단 기준이 매우 중요함

- 파인튜닝이 필요한 경우 
    1. 같은 출력 형식을 매번 프롬프트에 명시해야 할 때, 
        - 예를들어 {severity, description, suggestion}의 JSON 형식으로 매번 받고싶으면, 
        - 매번 프롬프팅하면 토큰낭비 + 가끔 형식 깨짐 -> 학습데이터 200~500개만 있어도 파인튜닝으로 일관성이 급격하게 올라감
    2. 도메인 용어를 계속 설명 해야할때,
        - "정처기","빅분기","SQLD","AICE"와 같이 한국 IT자격증 용어를 범용 모델이 정확히 이해하지 못하면,
        - 매번 system prompt에 용어 설명을 넣어야함
        - 파인튜닝하면 모델이 이 용어를 내재화함 
    3. latency가 중요할때, 파인튜닝된 모델은 짧은 프롬프트로도 원하는 결과를 냄
        - imput 토큰이 줄어들고, 첫 생성 시간이 빨라짐 
- 파인튜닝이 필요없는 경우
    1. RAG로 해결 가능한 "지식"문제 -> vectorDB 검색으로 충분
    2. few-shot 프롬프팅으로 해결되는 경우 -> 예시 2~3개로 충분한 경우 학습까지 안가도 됨
    3. 데이터가 200개 미만일때 -> 오히려 overfit 위험이 너무 높음 -> 성능 급격히 떨어짐 

### MLX LoRA/QLoRA 파인튜닝 개요
- MLX의 mlx_lm.lora 명령으로 Apple Silicon에서 직접 LoRA 파인튜닝 수행함. 
- LoRA는 원본 가중치를 동결하고, 작은 어댑터 행렬만 학습함.

> LoRA 파인튜닝 흐름
> Base Model -> Freeze Weights -> Add LoRA -> Train -> Merge/Save

- 9B 모델 기준 LoRA -> 18GB, QLoRA -> 8GB 필요함 


```txt
파인튜닝은 W를 W+ΔW로 업데이트 하는것임. 
풀 파인튜닝은 ΔW 전체(수십억 파라미터)를 학습하지만,
LoRA는 ΔW를 low-rank 분해함.

ΔW = A × B
여기서 A: (d × r), B: (r × d)
r = rank (보통 8~64)
d = 원본 차원 (수천)

Qwen3.5-9B의 어텐션 레이어 하나가 4096*4096 이면, ΔW는 16.7M 파라미터인데,
rank=16이라 하면, A(4096*16) + B(16 * 4096) = 131K 파라미터 이므로, 131K만 학습함
-> 약 99% 절약됨 
```
QLoRA는 한단계 더 베이스 모델을 4bit양자화 하고, 동결시킨후, LoRA 어댑터만 FP16으로 학습함 
> Qwen3.5-9B-4bit를 LoRA하면 사실상 QLoRA인것임 

```python
mlx_lm.lora \
  --model mlx-community/Qwen3.5-9B-4bit \
  --lora-layers 16 \    # 상위 16개 레이어에 LoRA 적용
  --batch-size 2 \
  --iters 1000
```
- lora-layers에서 왜 16?
    - 하위 레이어 -> 일반적인 언어 패턴(문법, 구문) -> 이미 충분히 학습됨
    - 상위 레이어 -> 태스크 특화 패턴(출력 형식, 도메인 지식) -> 여기만 바꾸면 됨 

- 실전에서 사용법 -> Qwen3.5-9B(48 layers) 기준으로, 레이어수는 직접 확인해야함 
    - 출력 형식만 바꾸고 싶다 -> 전체의 15~20% -> lora-layers 8 충분 
    - 도메인 지식까지 주입 -> 전체의 30~50% -> lora-layers 16~24
    - 언어 자체를 바꾸고 싶다 (영어 -> 한국어 특화) -> 전체의 60%+ -> lora-layers 32+
    - 레이어 수를 올릴수록 VRAM 사용량 증가, OOM 주의해야됨 

```txt
VRAM 추가량 ≈ lora_layers × hidden_dim × rank × 2 × 2bytes(FP16)

[Layer 0~25%]  하위 레이어
  → 토큰 임베딩, 기본 문법, 구문 패턴
  → 거의 모든 언어/도메인에 공통. 건드릴 필요 거의 없음

[Layer 25~60%] 중간 레이어
  → 의미 이해, 문맥 파악, 도메인 지식 인코딩
  → 도메인 특화할 때 여기를 건드림

[Layer 60~100%] 상위 레이어
  → 태스크 수행, 출력 형식 결정, 스타일
  → 출력 형식/스타일만 바꿀 때 여기만 건드려도 충분
```
#### 모델의 총 레이어수 확인 방법
- 모델마다 전부 다르니까 미리 직접 확인해야함.
- HuggingFace 모델 카드에서 config.json의 num_hidden_layers 필드 봐도 됨
- 아래는 코드로 직접 확인하는 법 

```python
from mlx_lm import load

model, tokenizer = load("mlx-community/Qwen3.5-9B-4bit")

# 트랜스포머 블록 수 확인
n_layers = len(model.model.layers)
print(f"총 레이어 수: {n_layers}")  # Qwen3.5-9B → 48

# 비율 기반으로 lora-layers 계산
target_ratio = 0.33  # 도메인 지식 주입용
lora_layers = int(n_layers * target_ratio)
print(f"추천 --lora-layers: {lora_layers}")  # → 16
```

### 데이터셋 형식(4가지)
#### 형식 1 -> Chat
- 가장 많이 사용
- 한줄이 하나의 학습 샘플
- 모델은 ssistant 역할의 응답을 생성하도록 학습됨 
```python
{"messages": [
    {"role": "system", "content": "당신은 코드 리뷰어입니다."},
    {"role": "user", "content": "이 함수를 리뷰해주세요: def add(a,b): return a+b"},
    {"role": "assistant", "content": "이 함수는 타입 힌트가 없고..."}
]}
```
- mlx_lm 내부에서 일어나는일 -> 직접 할 필요는 없음
- messages 리스트를 모델의 chat template에 맞게 자동 변환함 -> 순수한 messages만 넣어야됨 
```python
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
```
- Chat에서 
    - system역할은 선택사항임. 없어됨
    - user -> assistant 쌍이 여러번 반복되는 멀티턴 대화도 가능함 
    - 모든 샘플에서 system 내용이 동일하면, 프롬프트 엔지니어링으로 해결할 수 있는 문제라서
        - 파인튜닝대신 system prompt 고정으로 충분할 수 있음 

- 멀티턴 예시
```python
{"messages": [
    {"role": "system", "content": "당신은 코드 리뷰어입니다."},
    {"role": "user", "content": "이 함수를 리뷰해주세요: def add(a,b): return a+b"},
    {"role": "assistant", "content": "타입 힌트를 추가하는 게 좋겠습니다."},
    {"role": "user", "content": "타입 힌트를 어떻게 추가하죠?"},
    {"role": "assistant", "content": "def add(a: int, b: int) -> int: return a + b"}
]}
```

#### 형식2 -> Tools
- Agent의 Tool 호출을 학습시키기 위한 형식
```python
{"messages": [
    {"role": "system", "content": "당신은 도구를 사용할 수 있는 어시스턴트입니다."},
    {"role": "user", "content": "서울의 현재 날씨 알려줘"},
    {"role": "assistant", "tool_calls": [
        {"id": "call_1", "type": "function", 
         "function": {"name": "get_weather", "arguments": "{\"location\": \"서울\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "서울: 맑음, 18°C"},
    {"role": "assistant", "content": "서울의 현재 날씨는 맑음이고 기온은 18°C입니다."}
]}
```
- Tool 형식 사용 이유
    - 9B모델에서 Agent를 돌리면 Tool 호출 정확도가 낮은데, 
    - Tool 호출 데이터셋으로 파인튜닝하면 9B에서도 Tool 호출 정확도를 크게 올릴 수 있음 
    - 로컬 MLX 모델로 Agent를 돌릴 떄 실전적으로 가장 가치있는 형식임 

#### 형식3 -> Completions
```python
{"prompt": "Python으로 퀵소트를 구현하세요", "completion": "def quicksort(arr): ..."}
```
- Chat형식보다 단순함 (내부적으로 Completions형식이 Chat형식으로 변환되어 처리)
- 입력 -> 출력 쌍만 있음 
- 언제 사용?
    - 단순 번역 태스크 {"prompt": "영→한: Hello", "completion": "안녕하세요"}
    - 텍스트 변환: {"prompt": "요약: [긴 텍스트]", "completion": "[요약문]"}
    - system 역할이 불필요한 단순 매핑 태스크

#### 형식 4 -> Text (연속 텍스트)
```python
{"text": "LangChain은 LLM 애플리케이션 프레임워크로, LCEL이라는 파이프 문법을 사용하여..."}
```
- Text 형식을 써야하는 경우
    - 특정 도메인의 문체/용어를 모델에 주입하고 싶을 때 (예: 법률 판례문, 의학 논문)
    - 모델이 특정 언어의 텍스트를 더 자연스럽게 생성하도록 하고 싶을 때
    - instruction following 이전 단계로, 도메인 지식을 먼저 주입하는 2단계 학습의 1단계

- 2단계 학습 전략
    1. Text 형식으로 도메인 지식 주입 (continued pre-training)
        - 금융 보고서, IT 자격증 교재 등을 Text 형식으로 학습
        - 모델이 도메인 용어를 자연스럽게 이해하게 됨
    2. Chat 형식으로 instruction tuning
        - 도메인 QA 쌍으로 학습
        - 모델이 질문에 정확히 답하는 능력을 갖게 됨 
- 1단계에서 300~500개의 도메인 텍스트, 2단계에서 200~500개의 QA쌍이면 상당한 효과를 볼 수 있음 

#### 형식 5 -> mlx_lm은 4개를 지원하지만, mlx-lm-lora 확장 패키지는 DPO와 GRPO 형식도 지원함 
- DPO 형식 
- DPO는 이 답변이 저 답변보다 낫다 같은 선호도를 학습함
- 따라서 SFT(Supervised Fine-Tuning) 이후 추가로 적용하면 답변 품질이 더 좋아짐 
```python
{"prompt": "Python으로 정렬 알고리즘을 설명해줘", 
 "chosen": "퀵소트는 분할정복 방식으로... [좋은 답변]", 
 "rejected": "정렬은 데이터를 순서대로... [나쁜 답변]"}
```
