# PDF 파일을 읽고 질문에 답하는 완전한 시스템 만들기
# 두 파트로 나눠서 보기 쉽게 정리

# ==============================================
# part 1 : 인덱싱
from cryptography.x509 import base
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 1. 문서 로딩 & 2. 청크 분할
docs = PyPDFLoader("target.pdf").load()
chunks = RecursiveCharacterTextSplitter(
    chunks_size=500, chunk_overlab=50
).split_documents(docs)

# 3. 임베딩 & 벡터 DB
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large", model_kwargs={"device": "mps"}
)

# chroma 사용했지만 FAISS로 수정해서 사용해도됨
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./pdf_db")

# =====================================
# part 2 : RAG 체인 & QA 루프
llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="X",
    model="mlx-community/Qwen3.5-9B-4bit",
)

prompt = ChatPromptTemplate.from_template(
    "컨텍스트 : \n{context}\n\n" "질문 : {question} \n답변 : "
)

retriever = vectorstore.as_retriever(k=4)
fmt = lambda docs: "\n---\n".join(d.page_content for d in docs)
chain = (
    RunnableParallel(context=retriever | fmt, question=RunnablePassthrough())
    | prompt
    | llm
    | StrOutputParser()
)

# 대화형 QA 루프
while True:
    q = input("\n질문>")
    if q in ("q", "quit"):
        break
    for chunk in chain.stream(q):
        print(chunk, end="", flush=True)
