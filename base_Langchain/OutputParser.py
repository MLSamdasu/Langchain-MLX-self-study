"""
outputparser -> 출력 파싱 

- StrOutputParser -> 문자열 그대로 반환 
- JsonOutputParser -> JSON으로 파싱 -> dict로 반환 
- PydanticOutputParser -> Pydantic 모델로 구조화 
- RunnableLambda -> 내 함수가 반환하는 것 

"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# JSON 출력을 요청하는 프롬프트 
# Langchain은 기본적으로 {}를 변수로 인식함 -> {{}}로 써야됨 -> 껍데기 벗겨진다 생각하면됨 
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """ 
        다음 형식으로 JSON을 반환하라 : 
        {{
            "title" : "...",
            "summary" : "...",
            "keywords" : [...]
        }}
        """ 
    ),
    ("user", "{topic}에 대해 정리해라.")
])

# JSON 파서
parser = JsonOutputParsor()

chain = prompt | llm | parser 
# topic 빈칸에 LangChain LCEL 을 넣어서 invoke 
result = chain.invoke({"topic": "LangChain LCEL"})
