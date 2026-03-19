# RunnableLambda - 커스텀 반환 

from langchain_core.runnables import RunnableLambda

# 사용자 정의 함수
def add_timestamp(text: str) -> str:
    from datetime import datetime
    return f"[{datetime.now():%Y-%m-%d %H:%M}] {text}"

# LCEL 체인에 끼워넣기 
chain = prompt | llm | StrOutputParser() | RunnableLambda(add_timestamp)

result = chain.invoke({"input" : "오늘 날씨 알려줘. "})

# -> "[2026-03-03 14:30] 오늘은 ..."