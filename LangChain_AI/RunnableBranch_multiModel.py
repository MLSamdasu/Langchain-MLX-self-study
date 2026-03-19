from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 모델별 LLM 인스턴스 (각각 다른 포트)
"""
# 미리 서버 열어둬야함. 
## 터미널 1: 경량 3B (포트 8080)
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080

## 터미널 2: 고성능 9B (포트 8081)
mlx_lm.server --model mlx-community/Qwen3.5-9B-4bit --port 8081
"""

llm_light = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    model="mlx-community/Llama-3.2-3B-Instruct-4bit"
)

llm_heavy = ChatOpenAI(
    base_url="http://localhost:8081/v1",
    api_key="X",
    model="mlx-community/Qwen3.5-9B-4bit"
)

# 분류 체인 -> 작업 난이도 판별
classifier_prompt = ChatPromptTemplate.from_template(
    "다음 질문의 난이도를 'simple' 또는 'complex'로 답하시오. "
    "질문 : {question}\n 난이도 : "
)
classifier = classifier_prompt | llm_light | StrOutputParser()

# 라우팅 체인 
def route(info):
    difficulty = classifier.invoke(info)
    if 'simple' in difficulty.lower():
        return llm_light
    return llm_heavy

