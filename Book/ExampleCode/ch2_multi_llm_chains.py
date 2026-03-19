"""
Ch2 멀티 LLM 병렬 파이프라인 - 체인 구성

이 파일의 역할:
    - 각 LLM용 프롬프트 + LLM + 파서 체인 3개를 만든다
    - 각 검사 LLM용 체인 3개를 만든다
    - RunnableParallel로 병렬 연결한다
    - RunnableLambda로 검사 단계를 직렬 연결한다

비유:
    이 파일은 '주방 동선표'다.
    - 어떤 요리사가 어떤 재료를 받아서 어떤 순서로 처리하는지 정의한다
    - 3명의 요리사가 동시에 일하도록 배치하고 (RunnableParallel)
    - 각 요리가 끝나면 검수관이 바로 확인하도록 연결한다 (| 파이프)

전체 파이프라인 구조:
                        ┌─ 요약 프롬프트 → 요약 LLM → 파서 → 요약 검사 LLM ─┐
    [입력 텍스트] ──→   ├─ 번역 프롬프트 → 번역 LLM → 파서 → 번역 검사 LLM ─┤  → [최종 결과 dict]
                        └─ 키워드 프롬프트 → 키워드 LLM → 파서 → 키워드 검사 LLM ─┘
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

# config 파일에서 mock LLM들을 가져온다
from ch2_multi_llm_config import (
    summarizer_llm, translator_llm, keyword_llm,
    summary_checker_llm, translation_checker_llm, keyword_checker_llm,
)


# ============================================================
# 1) 프롬프트 3개 - 각 LLM에게 지시할 프롬프트 템플릿
# ============================================================

# 요약용 프롬프트
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 텍스트 요약 전문가입니다. 핵심만 간결하게 요약하세요."),
    ("user", "다음 텍스트를 3줄로 요약해주세요:\n{text}"),
])

# 번역용 프롬프트
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 한영 번역 전문가입니다. 자연스러운 영어로 번역하세요."),
    ("user", "다음 텍스트를 영어로 번역해주세요:\n{text}"),
])

# 키워드 추출용 프롬프트
keyword_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 키워드 추출 전문가입니다. 핵심 키워드를 쉼표로 구분하여 나열하세요."),
    ("user", "다음 텍스트에서 핵심 키워드 5개를 추출해주세요:\n{text}"),
])


# ============================================================
# 2) 출력 파서 - LLM 응답을 문자열로 변환
# ============================================================

parser = StrOutputParser()


# ============================================================
# 3) 개별 작업 체인 구성 (프롬프트 → LLM → 파서)
#    - 교재 2.3절의 LCEL 체인 연결 방식: prompt | llm | parser
#    - 여기서 llm 자리에 mock RunnableLambda가 들어감
#    - Ch3에서 ChatMLX를 배우면 여기만 교체하면 됨!
# ============================================================

# RunnableLambda로 감싼 mock LLM은 '문자열'을 받아야 하므로,
# 프롬프트의 출력(ChatPromptValue)을 문자열로 변환하는 중간 단계가 필요하다.
# 이것도 RunnableLambda로 해결! (커스텀 변환의 전형적인 활용)
prompt_to_str = RunnableLambda(lambda prompt_value: prompt_value.to_string())

# 요약 체인: 프롬프트 → 문자열 변환 → 요약 LLM → 파서
summary_chain = summary_prompt | prompt_to_str | summarizer_llm | parser

# 번역 체인: 프롬프트 → 문자열 변환 → 번역 LLM → 파서
translate_chain = translate_prompt | prompt_to_str | translator_llm | parser

# 키워드 체인: 프롬프트 → 문자열 변환 → 키워드 LLM → 파서
keyword_chain = keyword_prompt | prompt_to_str | keyword_llm | parser


# ============================================================
# 4) 검사 체인 구성 (작업 결과 → 검사 LLM)
#    - 각 작업 체인 뒤에 검사 LLM을 파이프(|)로 연결
#    - 요리사가 요리를 만들면 → 바로 검수관이 확인하는 구조
# ============================================================

# 요약 + 요약 검사 (직렬 연결)
summary_with_check = summary_chain | summary_checker_llm

# 번역 + 번역 검사 (직렬 연결)
translate_with_check = translate_chain | translation_checker_llm

# 키워드 + 키워드 검사 (직렬 연결)
keyword_with_check = keyword_chain | keyword_checker_llm


# ============================================================
# 5) RunnableParallel로 3개 체인을 병렬 연결
#    - 교재 2.6절: 하나의 입력으로 요약 + 키워드추출을 동시에 수행
#    - 여기서는 3개를 동시에! 결과는 dict로 돌아옴
#    - {"summary": "...", "translation": "...", "keywords": "..."}
# ============================================================

# 작업만 병렬 실행 (검사 없이)
parallel_llm_only = RunnableParallel(
    summary=summary_chain,
    translation=translate_chain,
    keywords=keyword_chain,
)

# 작업 + 검사까지 병렬 실행 (최종 파이프라인)
parallel_with_check = RunnableParallel(
    summary=summary_with_check,
    translation=translate_with_check,
    keywords=keyword_with_check,
)
