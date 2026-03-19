from langchain_core.prompts import ChatPromptTemplate
# 순수 문자열만 꺼내주는 파서 
from langchain_core.output_parsers import StrOutputParser


# 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트이다."),
    ("user", "{input}")
])

# LLM -> 예시
# llm = ChatMLX( . . . )

# 출력 파서
parser = StrOutputParser()

# LCEL 체인 : prompt -> llm -> parser 
chain = prompt | llm | parser 

# 실행 
result = chain.invoke({"input" : "랭체인이 뭐야? 3줄로"})
print(result)