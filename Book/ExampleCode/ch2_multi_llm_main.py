"""
Ch2 멀티 LLM 병렬 파이프라인 - 실행 파일

이 파일의 역할:
    - chains 파일에서 조립된 파이프라인을 import해서 실행한다
    - 결과를 보기 좋게 출력한다
    - 단계별로 실행해서 LCEL의 동작을 확인한다

실행 방법:
    cd Book/ExampleCode
    python ch2_multi_llm_main.py
"""

from ch2_multi_llm_chains import (
    summary_chain,
    translate_chain,
    keyword_chain,
    parallel_llm_only,
    parallel_with_check,
)


def print_separator(title: str) -> None:
    """구분선과 제목을 출력하는 헬퍼 함수"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    """멀티 LLM 병렬 파이프라인 실행"""

    # 테스트용 입력 텍스트
    input_data = {
        "text": (
            "LangChain은 대규모 언어 모델을 활용한 애플리케이션 개발 프레임워크입니다. "
            "LCEL이라는 표현 언어를 통해 프롬프트, 모델, 파서를 파이프 연산자로 연결할 수 있습니다. "
            "RunnableParallel을 사용하면 여러 체인을 동시에 실행할 수 있어서 처리 속도가 빨라집니다. "
            "Apple Silicon의 MLX 프레임워크와 결합하면 로컬에서도 강력한 AI 파이프라인을 구축할 수 있습니다."
        )
    }

    # ----------------------------------------------------------
    # Step 1: 개별 체인 하나씩 실행해보기
    # ----------------------------------------------------------
    print_separator("Step 1: 개별 체인 실행 (하나씩)")

    print("\n[요약 체인 실행]")
    summary_result = summary_chain.invoke(input_data)
    print(f"  결과: {summary_result}")

    print("\n[번역 체인 실행]")
    translate_result = translate_chain.invoke(input_data)
    print(f"  결과: {translate_result}")

    print("\n[키워드 체인 실행]")
    keyword_result = keyword_chain.invoke(input_data)
    print(f"  결과: {keyword_result}")

    # ----------------------------------------------------------
    # Step 2: RunnableParallel로 3개 동시 실행 (검사 없이)
    # ----------------------------------------------------------
    print_separator("Step 2: 병렬 실행 (검사 없이)")

    # invoke 한 번으로 3개 체인이 동시에 실행된다!
    # 결과는 dict로 돌아옴: {"summary": "...", "translation": "...", "keywords": "..."}
    parallel_result = parallel_llm_only.invoke(input_data)

    for key, value in parallel_result.items():
        print(f"\n  [{key}]")
        print(f"    {value}")

    # ----------------------------------------------------------
    # Step 3: 작업 + 검사까지 병렬 실행 (최종 파이프라인)
    # ----------------------------------------------------------
    print_separator("Step 3: 병렬 실행 + 검사 (최종 파이프라인)")

    # 이 한 줄로 6개의 작업이 실행된다:
    # 요약 → 요약검사, 번역 → 번역검사, 키워드 → 키워드검사 (3쌍이 병렬)
    final_result = parallel_with_check.invoke(input_data)

    for key, value in final_result.items():
        print(f"\n  [{key}]")
        print(f"    {value}")

    # ----------------------------------------------------------
    # Step 4: batch()로 여러 입력 한꺼번에 처리
    # ----------------------------------------------------------
    print_separator("Step 4: batch() - 여러 입력 동시 처리")

    # batch()는 여러 입력을 리스트로 받아서 한꺼번에 처리한다 (교재 2.4절)
    batch_inputs = [
        {"text": "파이썬은 간결하고 읽기 쉬운 프로그래밍 언어입니다."},
        {"text": "MLX는 Apple Silicon에 최적화된 머신러닝 프레임워크입니다."},
        {"text": "RAG는 검색 증강 생성으로 외부 문서를 활용해 답변합니다."},
    ]

    batch_results = parallel_with_check.batch(batch_inputs)

    for i, result in enumerate(batch_results):
        print(f"\n  --- 입력 {i + 1} ---")
        for key, value in result.items():
            print(f"    [{key}] {value}")

    # ----------------------------------------------------------
    # 정리
    # ----------------------------------------------------------
    print_separator("파이프라인 구조 요약")
    print("""
    이 예제에서 배운 것:

    1. RunnableLambda  - 일반 함수를 LCEL 호환 객체로 변환
    2. RunnableParallel - 여러 체인을 동시에 실행
    3. | (파이프)       - 체인을 직렬로 연결
    4. invoke()        - 단건 실행
    5. batch()         - 여러 건 동시 실행

    전체 흐름:
                            ┌─ 요약 프롬프트 → 요약 LLM → 파서 → 요약 검사 ─┐
    [입력 텍스트] ──→       ├─ 번역 프롬프트 → 번역 LLM → 파서 → 번역 검사 ─┤ → [결과 dict]
                            └─ 키워드 프롬프트 → 키워드 LLM → 파서 → 키워드 검사 ─┘

    Ch3에서 ChatMLX를 배우면:
    - config.py의 mock 함수 대신 실제 MLX 모델로 교체
    - chains.py의 prompt_to_str 제거하고 바로 llm 연결
    - main.py는 그대로 사용 가능!
    """)


if __name__ == "__main__":
    main()
