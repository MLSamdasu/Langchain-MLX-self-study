# pydantic 기반 structured tool
# 복잡한 입력을 받는 도구는 Pydantic 모델로 입력 스키마를 정의함

# =============================================================
# step1 -> Pydantic으로 입력 스키마 정의
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class FileReadInput(BaseModel):
    path: str = Field(description="읽을 파일 경로")
    encoding: str = Field(default="utf-8", description="파일 인코딩")


# Field -> LLM이 각 파라미터의 의미를 이해
# default 기본값 utf-8


# =========================================================
# step2 -> 실제 함수 작성
def read_file(path: str, encoding: str = "utf-8") -> str:
    """로컬 파일을 읽어 내용을 반환합니다."""
    try:
        with open(path, encoding=encoding) as f:
            return f.read()[:2000]  # 최대 2000자 안전장치
    except Exception as e:
        return f"파일 읽기 실패: {e}"


# =====================================================
# step3 -> StructuredTool.from_function로 조합
file_reader = StructuredTool.from_function(
    func=read_file,  # 실행할 함수
    name="file_reader",  # Tool 이름
    description="로컬 파일을 읽어 내용을 반환",  # tool 설명
    args_schema=FileReadInput,  # 입력 스키마
)
