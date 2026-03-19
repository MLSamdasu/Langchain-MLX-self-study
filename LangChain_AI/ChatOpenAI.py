"""
from openai import OpenAI          ← 전화기 (OpenAI 공식)
  → 기본 전화번호: api.openai.com  ← GPT한테 전화 걸림

from langchain_openai import ChatOpenAI  ← 전화기 (LangChain 버전)
  → 기본 전화번호: api.openai.com       ← 역시 GPT한테 전화
  → base_url 바꾸면?                    ← 다른 번호로 전화 가능!

"""

# 1) 일반 OpenAI SDK — GPT API 직통
from openai import OpenAI

client = OpenAI(api_key="sk-xxx")
client.chat.completions.create(...)  # → api.openai.com → GPT

# 2) LangChain의 ChatOpenAI — 기본은 GPT
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")  # → api.openai.com → GPT

# 3) LangChain의 ChatOpenAI — base_url을 로컬로
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",  # 여기가 핵심
    api_key="not-needed",  # 로컬이니까 아무 값
    model="mlx-community/Qwen3.5-9B-4bit",
)
# → localhost:8080 → mlx_lm.server → Qwen 로컬 모델
