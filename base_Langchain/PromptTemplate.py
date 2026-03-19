from langchain_core.prompts import PromptTemplate

"""
PromptTemplate은 변수를 포함한 프롬프트 틀을 만들어둠니다.
{topic} 같은 변수를 넣으면, 나중에 실제 값으로 교체됩니다.
매번 문자열을 조립하는 대신, 템플릿 한 번 만들고 재사용합니다.
"""

# 기본 템플릿 
template = PromptTemplate.from_template(
    "{topic}에 대해 간단히 설명하라."
)

# 변수 대입
prompt = template.format(topic="파이썬")
print(prompt)