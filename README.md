
#### 패키지 설치
```bush
pip install mlx mlx-lm
pip install langchain langchain-core langchain-community

pip install chromadb 
pip install sentence-transformers
pip install langsmith
pip install fastapi uvicorn
pip install rich 

```
- tip
> langchain 1.0부터는 langchain-core가 핵심이고, langchain은 래퍼입니다.
> langchain-community에 MLX 등 커뮤니티 통합이 들어있습니다.
> chromadb는 로컬에서 바로 돌릴 수 있는 벡터 DB입니다

#### 랭체인 1.0 
- LangChain 1.0 핵심 모듈:
- langchain-core langchain langsmith 핵심 추상화 (Runnable, LCEL, Prompt, Parser)
- 레거시 Chain + 새 Agent 시스템
- langchain-community MLX, Ollama 등 커뮤니티 통합
- 모니터링 / 트레이싱 / 평가

#### 모델 설치
```bush
python -m mlx_lm.generate \
--model mlx-community/Qwen3.5-9B-4bit \
--prompt "안녕하세요, 자기소개 해주세요" \
--max-tokens 200
```

```bush
python -m mlx_lm.generate \
--model mlx-community/Llama-3.2-3B-Instruct-4bit \
--prompt "What is LangChain?" \
--max-tokens 200
```

```bush
python -m mlx_lm.generate \
--model mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit \
--prompt "다음 코드의 버그를 찾아주세요" \
--max-tokens 500
```