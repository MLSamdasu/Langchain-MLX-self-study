### 패키지 설치

```bush

# 벡터 DB & 임베딩
pip install chromadb faiss-cpu
pip install sentence-transformers

# 문서 로더 & 텍스트 분할 
pip install langchain-community
pip install langchain-chroma

# pdf + markdown,html,word 
pip install pypdf unstructured

```

### 순서
1. 문서 로딩 (Document Loaders)
2. 텍스트 분할 (Text Splitters)
3. 임베딩 (Embeddings)
4. 벡터 DB 저장 ( Chroma DB, FAISS)
5. 검색 (similarity_search)
6. LLM 답변 