import os
import asyncio
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# [설정] ingest_db.py와 동일한 설정 유지
PERSIST_DIRECTORY = "./chroma_advanced_db"
embeddings = OllamaEmbeddings(model="bge-m3")
reranker_model = HuggingFaceCrossEncoder(model_name="dragonkue/bge-reranker-v2-m3-ko")

# LLM 설정
judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
drafter_models = {
    "scout": ChatGroq(model="llama-3.1-8b-instant", temperature=0.7),
    "qwen": ChatGroq(model="qwen/qwen3-32b", temperature=0.7) # API ID 확인 필요
}

def get_search_pipeline(vector_store):
    # 1. 문서 전체 텍스트 추출 (BM25용)
    all_docs = vector_store.get()["documents"]
    
    # 2. Base Retrievers
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    bm25_retriever = BM25Retriever.from_texts(all_docs)
    bm25_retriever.k = 15
    
    # 3. Hybrid Ensemble (BM25 0.4 : Vector 0.6)
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    
    # 4. Reranking
    compressor = CrossEncoderReranker(model=reranker_model, top_n=5)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble
    )
    return final_retriever

async def generate_final_answer(query, context):
    """MoA 구조로 최종 답변 생성 (필터링 강화 버전)"""
    
    # [수정] 1차 드래프트 모델들을 위한 프롬프트 - 여기서부터 입을 막아야 합니다.
    draft_prompt_template = """
    당신은 앱 리뷰 데이터 분석가입니다.
    
    [가이드라인]
    1. [질문]이 제공된 [문맥]과 조금이라도 관련이 있는지 확인하세요.
    2. 만약 질문이 '배고파', '졸려', '인사' 등 앱 분석과 무관한 내용이라면, 구구절절 설명하지 말고 반드시 딱 한 문장 "NO_RELATION"이라고만 답하세요.
    3. 만약 질문 대상 앱이 '넷플릭스', '왓챠', '티빙', '디즈니+', '쿠팡플레이', 'Prime Video', '애플tv' 가 아닐 경우 "NOT_APP"이라고만 답하세요.
    4. 관련이 있다면, 문맥에 기반하여 전문적인 분석 초안을 작성하세요.

    [문맥]: {context}
    [질문]: {query}
    
    답변:
    """
    draft_prompt = ChatPromptTemplate.from_template(draft_prompt_template)
    
    # 1. 드래프트 모델 병렬 실행
    tasks = []
    model_labels = list(drafter_models.keys())
    for name, model in drafter_models.items():
        chain = draft_prompt | model | StrOutputParser()
        tasks.append(chain.ainvoke({"context": context, "query": query}))
    
    results = await asyncio.gather(*tasks)
    
    # [추가] 모든 드래프트 모델이 거절했는지 확인 (성능 최적화)
    if all("NO_RELATION" in res for res in results):
        return "죄송합니다. 저는 앱 리뷰 분석 전문가로서 해당 질문에 답변을 드릴 수 없습니다. 앱의 성능, 사용자 반응, 버전 비교 등에 대해 질문해 주세요!"
    
    if all("NOT_APP" in res for res in results):
        return "죄송합니다. 저는 OTT 리뷰 분석 전문가로서 해당 질문에 답변을 드릴 수 없습니다. OTT 앱에 관해 질문해 주세요!"
    
    # 2. 결과 종합용 프롬프트 (Judge LLM)
    candidates = ""
    for name, res in zip(model_labels, results):
        candidates += f"\n\n--- [분석가: {name}] ---\n{res}"
    
    final_prompt = f"""
    당신은 앱 리뷰 분석 전문가입니다. 아래 의견들을 종합하여 답변하세요.
    
    [참조 데이터]
    {context}
    
    [분석가 의견 모음]
    {candidates}
    
    [최종 답변 규칙]
    - 만약 분석가들의 의견이 "NO_RELATION"이거나 질문이 일상 대화라면, 억지로 앱과 연결하지 마세요.
    - 무관한 질문에는 "죄송합니다. 저는 앱 분석 전문가입니다. 앱 관련 질문을 해주세요."라고만 답하세요.
    - 관련 있는 질문에는 친절한 구어체(~해요)로 핵심만 짚어주세요.
    
    질문: {query}
    """
    
    final_answer = await judge_llm.ainvoke(final_prompt)
    return final_answer.content

async def main():
    # 1. 저장된 벡터 DB 로드
    print("📂 벡터 DB 로드 중...")
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    
    # 2. 검색 파이프라인 구축
    retriever = get_search_pipeline(vector_store)
    
    print("\n✅ 준비 완료! 질문을 입력하세요 (종료하려면 'exit' 입력)")
    
    while True:
        query = input("\n[질문]: ")
        if query.lower() in ['exit', 'quit', 'q', '종료']:
            break
            
        print("🔍 검색 및 분석 중...")
        
        # A. 관련 문서 검색
        retrieved_docs = retriever.invoke(query)
        
        # B. 문맥 구성 (메타데이터 포함)
        context_list = []
        for d in retrieved_docs:
            source_info = f"[앱: {d.metadata.get('app_name')}, 버전: {d.metadata.get('version')}, 날짜: {d.metadata.get('date')}]"
            context_list.append(f"{source_info}\n내용: {d.page_content}")
        
        context = "\n\n".join(context_list)
        
        # C. 최종 답변 생성
        answer = await generate_final_answer(query, context)
        
        print("\n" + "="*50)
        print("🤖 [AI 답변]:")
        print(answer)
        print("="*50)

if __name__ == "__main__":
    asyncio.run(main())