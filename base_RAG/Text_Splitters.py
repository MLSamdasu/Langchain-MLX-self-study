from base_RAG.Document_Loader import pdf_docs
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    # chunk_size -> 권장 300-1000 -> 청크 최대 크기, 작을수록 정밀, 클수록 맥락 풍부
    chunk_size=500,
    # chunk_overlap -> 권장 50-100 -> 인접 청크 간 겹침. 문맥 단절 방지
    chunk_overlap = 50,
    
    # separators -> [\n\n, \n, . ...] -> 분할 우선순위, 문단 > 줄 > 문장 순서 
    # 1순위 "\n\n" → 빈 줄 (문단 경계)에서 먼저 자름                                                  
    # 2순위 "\n"   → 문단이 너무 크면 줄바꿈에서 자름                                                 
    # 3순위 ". "   → 줄이 너무 길면 문장 끝에서 자름                                                  
    # 4순위 " "    → 문장이 너무 길면 단어 사이에서 자름                                              
    # 5순위 ""     → 최후의 수단: 글자 하나씩 자름   
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = splitter.split_documents(pdf_docs)
print(f"원본: {len(pdf_docs)}개 -> 청크: {len(chunks)}개")
print(f"청크 예시: {chunks[0].page_content[:100]}...")