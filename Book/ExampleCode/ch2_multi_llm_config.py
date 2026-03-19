"""
Ch2 멀티 LLM 병렬 파이프라인 - 설정 & Mock LLM 정의

이 파일의 역할:
    - RunnableLambda로 가짜(mock) LLM 3개를 정의한다 (요약기, 번역기, 키워드추출기)
    - 각 LLM의 출력을 검증하는 검사 LLM 3개를 정의한다
    - Ch3에서 ChatMLX를 배우면, 여기의 mock 함수만 실제 모델로 교체하면 된다

비유:
    식당 주방을 생각해보자.
    - 요리사 3명이 동시에 각자 다른 요리를 만든다 (= LLM 3개 병렬 실행)
    - 각 요리마다 맛 검수관이 있어서 "이거 괜찮아?" 확인한다 (= 검사 LLM 3개)
    - 이 파일은 요리사와 검수관의 '레시피(함수)'를 정의하는 곳이다
"""

from langchain_core.runnables import RunnableLambda


# ============================================================
# 헬퍼: 프롬프트 문자열에서 사용자 텍스트만 추출
#   - prompt_to_str을 거치면 "System: ...\nHuman: ..." 형태의 문자열이 들어옴
#   - 실제 LLM은 전체를 받지만, mock에서는 사용자 입력만 꺼내 시뮬레이션
# ============================================================

def _extract_user_text(prompt_str: str) -> str:
    """프롬프트 문자열에서 'Human:' 이후의 사용자 메시지만 추출한다"""
    if "Human:" in prompt_str:
        # "Human: 다음 텍스트를 3줄로 요약해주세요:\n실제 텍스트..." 에서 실제 텍스트 부분 추출
        human_part = prompt_str.split("Human:")[-1].strip()
        # "다음 텍스트를 ... :\n" 이후의 실제 텍스트를 꺼냄
        if "\n" in human_part:
            return human_part.split("\n", 1)[-1].strip()
        return human_part
    return prompt_str


# ============================================================
# 1) Mock LLM 함수 3개 - 실제 모델 대신 가짜 응답을 반환
# ============================================================

def mock_summarizer(input_text: str) -> str:
    """
    요약기 LLM (Mock)
    - 실제로는 Qwen3.5-9B 같은 모델이 긴 글을 요약해줌
    - 여기서는 앞 50자를 잘라서 '요약했다'고 시뮬레이션
    """
    text = _extract_user_text(input_text)
    truncated = text[:50] + "..." if len(text) > 50 else text
    return f"[요약 결과] {truncated}"


def mock_translator(input_text: str) -> str:
    """
    번역기 LLM (Mock)
    - 실제로는 한국어 -> 영어 번역을 수행
    - 여기서는 간단히 'Translated:' 접두사를 붙여서 시뮬레이션
    """
    text = _extract_user_text(input_text)
    return f"[번역 결과] Translated: {text[:40]}"


def mock_keyword_extractor(input_text: str) -> str:
    """
    키워드 추출기 LLM (Mock)
    - 실제로는 핵심 키워드를 뽑아주는 모델
    - 여기서는 공백 기준으로 단어를 쪼개서 앞 5개를 키워드로 반환
    """
    text = _extract_user_text(input_text)
    words = text.split()
    keywords = words[:5] if len(words) >= 5 else words
    return f"[키워드 결과] {', '.join(keywords)}"


# ============================================================
# 2) Mock 검사 LLM 함수 3개 - 각 LLM 출력을 검증
# ============================================================

def mock_summary_checker(summary: str) -> str:
    """
    요약 검사기 (Mock)
    - 요약 결과가 너무 짧거나 비었으면 FAIL
    - 정상이면 PASS + 글자 수 보고
    """
    if not summary or len(summary) < 10:
        return "[요약 검사] FAIL - 요약이 너무 짧습니다"
    return f"[요약 검사] PASS - {len(summary)}자, 품질 양호"


def mock_translation_checker(translation: str) -> str:
    """
    번역 검사기 (Mock)
    - 'Translated' 키워드가 포함되어 있는지 확인 (번역이 실제로 수행되었는지)
    - 포함되어 있으면 PASS
    """
    if "Translated" not in translation:
        return "[번역 검사] FAIL - 번역이 수행되지 않았습니다"
    return f"[번역 검사] PASS - 번역 정상 완료"


def mock_keyword_checker(keywords: str) -> str:
    """
    키워드 검사기 (Mock)
    - 키워드가 최소 1개 이상 추출되었는지 확인
    - 쉼표(,)로 구분된 키워드 개수를 센다
    """
    # '[키워드 결과] ' 이후의 실제 키워드 부분을 추출
    keyword_part = keywords.replace("[키워드 결과] ", "")
    keyword_list = [k.strip() for k in keyword_part.split(",") if k.strip()]
    if len(keyword_list) == 0:
        return "[키워드 검사] FAIL - 키워드가 추출되지 않았습니다"
    return f"[키워드 검사] PASS - {len(keyword_list)}개 키워드 추출됨"


# ============================================================
# 3) RunnableLambda로 감싸서 LCEL 호환 객체 만들기
#    - 일반 함수를 RunnableLambda로 감싸면 | (파이프) 연산자로 연결 가능
#    - invoke(), batch(), stream() 메서드도 자동으로 생김
# ============================================================

# 작업 LLM 3개 (요리사)
summarizer_llm = RunnableLambda(mock_summarizer)
translator_llm = RunnableLambda(mock_translator)
keyword_llm = RunnableLambda(mock_keyword_extractor)

# 검사 LLM 3개 (검수관)
summary_checker_llm = RunnableLambda(mock_summary_checker)
translation_checker_llm = RunnableLambda(mock_translation_checker)
keyword_checker_llm = RunnableLambda(mock_keyword_checker)
