"""
병렬 처리 ->
RunnableParallel -> 동시에 여러 체인 돌리기 

카페에서 바리스타는 커피 내리고, 다른 직원은 케이크 자르고, 
둘 다 끝나면 트레이에 같이 담아줌
"""

from langchain_core.runnables import RunnableParallel

# 요약 체인 
summary_chain = summary_prompt | llm | StrOutputParser()

# 키워드 체인 
keyword_chain = keyword_prompt | llm |  StrOutputParser()

# 병렬 실행
parallel = RunnableParallel(
    summary=summary_chain, # 요약 체인 
    keywords=keyword_chain # 키워드 체인
)

result = parallel.invoke({"text" : "긴 문서 내용..."})
print(result["summary"]) # 요약 결과
print(result["keywords"]) # 키워드 결과 

"""
                      ┌─ summary_chain ─→ "요약"    ─┐                                
  {"text": "..."} ──→│                               │──→ {"summary": "...",          
  "keywords": "..."}                                                                  
                      └─ keyword_chain ─→ "키워드"  ─┘                                
                                                                                      
  핵심: 하나의 입력이 두 갈래로 동시에 처리되고, 결과가 dict 하나로 합쳐짐.    
"""