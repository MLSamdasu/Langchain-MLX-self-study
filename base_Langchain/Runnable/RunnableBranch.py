"""
RunnableBranch -> 조건 분기 

  구조 = Python의 if/elif/else와 동일:                                                
  "코드" 포함? ──Yes──→ code_chain                                                    
        │No                                                                           
  "번역" 포함? ──Yes──→ translate_chain                                               
        │No
  general_chain (default)  
"""

from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    # (조건함수, 실행할 체인)
    (lambda x: "코드" in x["input"], code_chain), # 조건1
    (lambda x: "번역" in x["input"], translate_chain), # 조건2

    # default 체인 
    general_chain # default(else)
)

# "코드" 가 포함되면 "코드"가 존재하니까 code_chain 실행 
data = {"input" : "코드 리뷰해줘."}
result = branch.invoke(data)

