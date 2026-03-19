# LCEL + MLX 코드 리뷰 파이프라인 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# MLX 로컬 모델 
llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    model="mlx-community/Qwen3.5-9B-4bit",
    temperature=0.7,
)

# 코드 리뷰 파이프라인 
review_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 파이썬 코드 리뷰어 입니다."),
    ("user", "다음 코드를 리뷰하시오. {code}"),
])

# LCEL 파이프 체인 
review_chain= review_prompt | llm | StrOutputParser()

# 실행 
result = review_chain.invoke({
    "code":"def add(a,b): return a+b"
})

print(result)