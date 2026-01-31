import os
import asyncio
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

from django.conf import settings
from dotenv import load_dotenv

# =======================================================
# [Fix] Import: 존재하는 패키지만 확실하게 가져오기
# =======================================================
# 1. LangChain Core (설치된 1.2.7 버전 활용)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 2. Embeddings & GenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

# 3. Community Modules (BM25, CrossEncoder)
# 만약 경로 문제 발생 시 직접적인 대체 로직을 위해 try-import
try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    BM25Retriever = None # 없을 경우 직접 구현 로직 사용 대비

try:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
except ImportError:
    HuggingFaceCrossEncoder = None

# =======================================================
# [설정] 환경 변수 및 모델 로드
# =======================================================
load_dotenv()

PERSIST_DIRECTORY = "./RAG/chromadb_report_L1_to_L4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📂 [System] AI 서비스 초기화 (Direct Logic Mode)")
print(f"⚡ [Device] {DEVICE.upper()}")

# 1. 임베딩 모델
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. 리랭커 모델 (CrossEncoder)
# Import 에러가 나더라도 sentence_transformers를 직접 사용하여 구현 가능
reranker_model = None
try:
    if HuggingFaceCrossEncoder:
        reranker_model = HuggingFaceCrossEncoder(
            model_name="dragonkue/bge-reranker-v2-m3-ko",
            model_kwargs={'device': DEVICE}
        )
    else:
        # Fallback: sentence_transformers 직접 사용
        from sentence_transformers import CrossEncoder
        reranker_model = CrossEncoder("dragonkue/bge-reranker-v2-m3-ko", device=DEVICE)
except Exception as e:
    print(f"⚠️ Reranker 초기화 실패: {e}")

# 3. LLM 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-3.0-flash", 
    temperature=0.2,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    streaming=True
)

# 4. VectorStore 연결
vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# =======================================================
# [Logic] Direct Hybrid Search Implementation
# 라이브러리(EnsembleRetriever 등) 없이 직접 로직 구현
# =======================================================

class DirectSearchEngine:
    _docs_cache: Dict[str, List[Document]] = {}
    _is_loaded = False

    @classmethod
    def load_cache(cls):
        if cls._is_loaded: return
        
        # [디버깅 1] 경로 확인
        abs_path = os.path.abspath(PERSIST_DIRECTORY)
        print(f"🧐 [Debug] DB 경로 확인: {abs_path}")
        if not os.path.exists(PERSIST_DIRECTORY):
            print(f"❌ [Critical] 해당 경로에 폴더가 없습니다! 경로를 확인하세요.")
            return

        print("⏳ [System] 문서 데이터 로딩 및 캐싱 시도...")
        try:
            # DB 연결 테스트
            data = vector_store.get()
            doc_count = len(data['documents']) if data['documents'] else 0
            
            print(f"📊 [Debug] DB에서 발견된 문서 수: {doc_count}개")

            if doc_count == 0:
                print("⚠️ [Warning] DB가 비어있습니다. 임베딩(Ingest)이 제대로 안 됐거나 경로가 틀렸습니다.")
                return
            
            for text, meta in zip(data['documents'], data['metadatas']):
                app = meta.get('app_name', 'Unknown')
                doc = Document(page_content=text, metadata=meta)
                if app not in cls._docs_cache:
                    cls._docs_cache[app] = []
                cls._docs_cache[app].append(doc)
            
            cls._is_loaded = True
            apps_found = list(cls._docs_cache.keys())
            print(f"✅ [System] 캐싱 완료. 발견된 앱 목록: {apps_found}")

        except Exception as e:
            print(f"❌ [Error] 캐싱 중 에러 발생: {e}")

    @classmethod
    def get_bm25_retriever(cls, target_apps: List[str]):
        docs = []
        for app in target_apps:
            docs.extend(cls._docs_cache.get(app, []))
        
        if not docs: 
            print(f"⚠️ [Debug] '{target_apps}'에 대한 캐시 문서가 없음 (BM25 생성 불가)")
            return None
        
        if BM25Retriever:
            return BM25Retriever.from_documents(docs)
        return None

    @staticmethod
    def rerank_documents(query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
        if not docs or not reranker_model: return docs[:top_n]
        pairs = [[query, doc.page_content] for doc in docs]
        try:
            if hasattr(reranker_model, 'predict'):
                scores = reranker_model.predict(pairs)
            elif hasattr(reranker_model, 'score'):
                scores = reranker_model.score(pairs)
            else:
                return docs[:top_n]
            
            scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:top_n]]
        except Exception as e:
            print(f"⚠️ Reranking Error: {e}")
            return docs[:top_n]

# 서버 시작 시 로드
DirectSearchEngine.load_cache()


# =======================================================
# [유틸] 질문 의도 분석
# =======================================================
async def analyze_query_intent(query: str) -> Dict:
    query_lower = query.lower()
    intent_filter = {} 
    if any(k in query_lower for k in ["종합", "전체", "총평", "브랜드"]):
        intent_filter["an_level"] = {"$in": ["L3", "L4"]}
    elif any(k in query_lower for k in ["분기", "트렌드", "동향", "추세", "비교"]):
        intent_filter["an_level"] = "L2"
    elif any(k in query_lower for k in ["버그", "오류", "안돼", "실행", "업데이트", "버전"]):
        intent_filter["an_level"] = "L1"
    return intent_filter


# =======================================================
# [Main] 응답 생성기
# =======================================================
async def analyze_query_intent(query: str) -> Dict:
    query_lower = query.lower()
    intent_filter = {} 
    if any(k in query_lower for k in ["종합", "전체", "총평", "브랜드"]):
        intent_filter["an_level"] = {"$in": ["L3", "L4"]}
    elif any(k in query_lower for k in ["분기", "트렌드", "동향", "추세", "비교"]):
        intent_filter["an_level"] = "L2"
    elif any(k in query_lower for k in ["버그", "오류", "안돼", "실행", "업데이트", "버전"]):
        intent_filter["an_level"] = "L1"
    return intent_filter


async def generate_chat_response(query, valid_apps, current_app_name=None):
    if not DirectSearchEngine._is_loaded:
        DirectSearchEngine.load_cache()

    # [디버깅 2] 입력값 확인
    print(f"📥 [Input] 질문: {query} | 현재앱: {current_app_name} | 권한앱: {valid_apps}")

    target_apps = []
    context_keywords = ["내 앱", "이 앱", "여기", "우리 앱", "요약", "분석"]
    if current_app_name and (any(k in query for k in context_keywords) or not any(app in query for app in valid_apps if app != current_app_name)):
        target_apps = [current_app_name]
        search_query = f"{current_app_name} {query}"
    else:
        target_apps = valid_apps
        search_query = query
    
    display_query = query 

    if not target_apps:
        yield "분석할 앱 권한이 없습니다."
        return

    level_filter = await analyze_query_intent(search_query)
    
    # 필터 조건 구성
    app_condition = {}
    if len(target_apps) == 1:
        app_condition = {"app_name": target_apps[0]}
    else:
        app_condition = {"app_name": {"$in": target_apps}}
    
    chroma_filter = {}
    if level_filter:
        chroma_filter = {"$and": [app_condition, level_filter]}
    else:
        chroma_filter = app_condition

    # [디버깅 3] 실제 검색에 사용되는 필터 확인
    print(f"🔎 [Search] 검색 쿼리: '{search_query}'")
    print(f"🔎 [Search] 필터 조건: {chroma_filter}")

    final_docs = []
    
    try:
        # Step 1: Vector Search
        vector_docs = await asyncio.to_thread(
            vector_store.similarity_search,
            query=search_query,
            k=20,
            filter=chroma_filter
        )
        print(f"  └ [Result] 벡터 검색 결과: {len(vector_docs)}개")
        
        # Step 2: BM25 Search
        bm25_docs = []
        bm25_retriever = DirectSearchEngine.get_bm25_retriever(target_apps)
        if bm25_retriever:
            bm25_retriever.k = 20
            bm25_docs = await asyncio.to_thread(bm25_retriever.invoke, search_query)
        print(f"  └ [Result] BM25 검색 결과: {len(bm25_docs)}개")
        
        # Step 3: Ensemble
        seen_contents = set()
        combined_docs = []
        for doc in bm25_docs + vector_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                combined_docs.append(doc)
        
        # Step 4: Reranking
        final_docs = DirectSearchEngine.rerank_documents(search_query, combined_docs, top_n=5)
        print(f"  └ [Result] 최종 리랭킹 결과: {len(final_docs)}개")

    except Exception as e:
        print(f"❌ Search Pipeline Error: {e}")
        yield f"검색 중 시스템 오류가 발생했습니다: {e}"
        return

    if not final_docs:
        yield f"🔍 **'{display_query}'** 관련 리포트를 찾지 못했습니다.\n(DB 경로와 앱 이름을 확인해주세요)"
        return

    # 컨텍스트 구성
    context_text = ""
    used_sources = []
    
    for i, doc in enumerate(final_docs, 1):
        meta = doc.metadata
        level = meta.get('an_level', 'Report')
        ver = meta.get('version', 'Unknown')
        source_name = f"[{meta.get('app_name')}] {level} (v{ver})"
        if meta.get('quarter_id'): source_name += f" - {meta.get('quarter_id')}"
        used_sources.append(source_name)
        context_text += f"\n[문서 {i}] 출처: {source_name} | 내용: {doc.page_content}"

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    prompt_template = """당신은 앱 데이터 분석 전문가 'KIWI AI'입니다.
제공된 [참고 문서]를 바탕으로 질문에 대해 전문적으로 답변하십시오.

[분석 환경]
- 시간: {current_time}
- 앱: {target_apps_str}

[참고 문서]
{context}

[질문]
{query}

[답변 가이드]
1. 문서에 있는 사실(Data)에 기반하여 구체적으로 답변하세요.
2. 버그나 문제점은 해결 여부나 특정 버전을 언급하세요.
3. 문서에 없는 내용은 추측하지 말고 없다고 말하세요.
4. 중요 키워드는 **굵게** 표시하세요.
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()

    try:
        async for chunk in chain.astream({
            "context": context_text,
            "query": display_query,
            "current_time": current_time,
            "target_apps_str": ", ".join(target_apps)
        }):
            yield chunk
        
        unique_sources = sorted(list(set(used_sources)))
        yield "\n\n---\n**📚 참고 리포트:**\n" + "\n".join([f"- {s}" for s in unique_sources])

    except Exception as e:
        print(f"❌ Generation Error: {e}")
        yield "답변 생성 중 오류가 발생했습니다."