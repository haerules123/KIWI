import os
import asyncio
from typing import List
import json
import re

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# 로컬 NLP 라이브러리 (가정)
from keybert import KeyBERT
from konlpy.tag import Okt
# from transformers import pipeline # KoBERT 요약용

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==========================================
# [설정] 모델 및 경로 정의
# ==========================================
PERSIST_DIRECTORY = "./chroma_advanced_db"

# 1. 임베딩 (Local Ollama)
# 'ollama pull bge-m3' 선행 필요
embeddings = OllamaEmbeddings(model="bge-m3") 

# 2. 리랭커 (HuggingFace Local)
reranker_model = HuggingFaceCrossEncoder(model_name="dragonkue/bge-reranker-v2-m3-ko")

# 3. LLM 설정 (Groq)
# (1) 메타데이터 판정 및 최종 종합용 (심판)
judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)

# (2) 답변 생성용 3대장 (Draft Models)
# 주의: Groq에서 실제 지원하는 모델 ID여야 함. (예시 ID 사용)
drafter_models = {
    "scout": ChatGroq(model="llama-3.1-8b-instant", temperature=0.7),      # Llama-4 대체용 (임시)
    "qwen": ChatGroq(model="qwen/qwen3-32b", temperature=0.7)                # Qwen3 대체용 (임시)
}

# ==========================================
# [Phase 1] 메타데이터 앙상블 추출기
# ==========================================
class MetadataEnsemble:
    def __init__(self):
        self.kw_model = KeyBERT()
        self.okt = Okt()
        # self.summ_model = pipeline("summarization", model="kobert-base...") # KoBERT 로드 가정
    
    def _extract_local_keywords(self, text):
        """KeyBERT + KoNLPy 결합"""
        nouns = " ".join(self.okt.nouns(text)) # 명사만 추출해서 후보군 압축
        keywords = self.kw_model.extract_keywords(nouns, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=5)
        return [k[0] for k in keywords]

    async def generate_metadata(self, text):
        """
        로컬 vs LLM 결과를 비교하여 Judge가 더 나은 것을 선택
        """
        # 1. 로컬 추출
        local_keywords = self._extract_local_keywords(text)
        
        # 2. LLM 추출 (Groq)
        llm_prompt = f"""
        당신은 앱 리뷰 분석 전문가입니다. 다음 텍스트에서 핵심 정보를 JSON 형식으로 추출하세요.
        
        [텍스트]: {text}
        
        [응답 형식]:
        {{
            "keywords": ["키워드1", "키워드2", ...],
            "summary": "한 줄 요약",
            "sentiment": "긍정/부정/중립 중 선택",
            "features": ["언급된 기능1", "언급된 기능2"]
        }}
        """
        try :
            llm_res = await drafter_models["scout"].ainvoke(llm_prompt)
            json_str = re.search(r'\{.*\}', llm_res.content, re.DOTALL).group()
            llm_data = json.loads(json_str)
        except:
            # 실패 시 기본값
            llm_data = {"keywords": [], "summary": text[:50], "sentiment": "중립", "features": []}

        # 3. 데이터 결합 (Ensemble)
        # 로컬의 정확한 단어 + LLM의 문맥 단어를 합집합(set)으로 처리
        final_keywords = list(set(local_keywords) | set(llm_data.get("keywords", [])))
        
        # 4. 최종 결과 반환
        metadata = {
            "keywords": ", ".join(final_keywords),
            "summary": llm_data.get("summary", text[:100]),
            "sentiment": llm_data.get("sentiment", "알 수 없음"),
            "features": ", ".join(llm_data.get("features", []))
        }
        
        return metadata

# ==========================================
# [Phase 2] 데이터 적재 (Ingestion)
# ==========================================
async def ingest_markdown_reports(markdown_reports: list):
    """
    markdown_reports: [{"text": "마크다운내용", "version": "v1.0.2"}, ...] 형태의 리스트
    """
    extractor = MetadataEnsemble()
    all_processed_chunks = []

    # 1. 마크다운 헤더 스플리터 설정
    headers_to_split_on = [
        ("#", "AppTitle"),
        ("##", "Section"),
        ("###", "SubSection"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # 내용이 너무 길 경우를 대비한 2차 스플리터
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    print(f"🚀 총 {len(markdown_reports)}개의 보고서 처리 시작...")

    for report in markdown_reports:
        # A. 구조적 분할 (헤더 기반)
        header_splits = md_splitter.split_text(report["text"])
        
        # B. 세부 분할 및 메타데이터 강화
        for doc in header_splits:
            # 너무 길면 쪼갬 (기존 metadata는 유지됨)
            sub_chunks = text_splitter.split_documents([doc])
            
            for chunk in sub_chunks:
                # 기본 정보 태깅
                chunk.metadata["version"] = report["version"]
                
                # C. [핵심] Phase 1의 앙상블 메타데이터 추출 활용
                # 쪼개진 텍스트(chunk.page_content)에 대해 키워드/요약 생성
                metadata_result = await extractor.generate_metadata(chunk.page_content)
                chunk.metadata["keywords"] = metadata_result["keywords"]
                chunk.metadata["summary"] = metadata_result["summary"]
                chunk.metadata["sentiment"] = metadata_result["sentiment"] # 추가된 필드 활용
                chunk.metadata["features"] = metadata_result["features"]   # 추가된 필드 활용
                all_processed_chunks.append(chunk)

    # 2. 벡터 DB 저장 (임베딩 수행)
    vector_store = Chroma.from_documents(
        documents=all_processed_chunks,
        embedding=embeddings, # bge-m3 사용
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f"💾 {len(all_processed_chunks)}개의 청크가 벡터 DB에 저장되었습니다.")
    return vector_store

# ==========================================
# [Phase 3] 검색 및 리랭킹
# ==========================================
def get_search_pipeline(vector_store, doc_texts):
    # 1. Base Retrievers
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    bm25_retriever = BM25Retriever.from_texts(doc_texts)
    bm25_retriever.k = 20
    
    # 2. Hybrid (Ensemble)
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    
    # 3. Reranking (Cross Encoder)
    compressor = CrossEncoderReranker(model=reranker_model, top_n=5)
    
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble
    )
    return final_retriever

# ==========================================
# [Phase 4] MoA (Mixture of Agents) 생성
# ==========================================
async def generate_final_answer(query, context):
    """
    3개의 모델이 초안을 작성하고, Llama-3.3-70B가 최종 종합
    """
    print("\n🤖 [MoA] 3개 모델이 동시 분석 중...")
    
    prompt_template = """
    [문맥]: {context}
    [질문]: {query}
    
    위 내용을 바탕으로 질문에 대한 전문적인 답변을 작성하세요.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # 1. 3개 모델 병렬 실행 (Async)
    tasks = []
    model_names = ["llama-4-scout", "qwen3-32b"]
    
    for name, model in drafter_models.items():
        chain = prompt | model | StrOutputParser()
        tasks.append(chain.ainvoke({"context": context, "query": query}))
        
    results = await asyncio.gather(*tasks)
    
    # 2. 답변 모음
    candidates = ""
    for name, res in zip(model_names, results):
        candidates += f"\n\n--- [의견: {name}] ---\n{res}"
        
    print("👨‍⚖️ [Judge] Llama-3.3-70B가 최종 판결 중...")
    
    # 3. 최종 종합 (Synthesizer)
    final_prompt = f"""
    당신은 앱 리뷰 분석 전문가이자 친절한 상담원입니다. 
    아래 3명의 분석가 의견을 종합하여 사용자에게 최종 답변을 제공하세요.
    
    [질문]: {query}
    
    [검색된 팩트 데이터]
    {context}
    
    [분석가 초안 모음]
    {candidates}
    
    [답변 가이드라인]
    1. 말투는 "~해요", "~입니다"와 같은 친절한 구어체를 사용하세요.
    2. 보고서 형식이 아닌, 질문에 대해 핵심만 짚어주는 '채팅 답변' 형태로 작성하세요.
    3. 팩트 데이터에 근거하되, 불필요하게 긴 서론이나 결론은 생략하고 바로 본론을 말씀하세요.
    4. 중요한 수치나 버전 정보가 있다면 빠뜨리지 마세요.
    """
    
    final_chain = judge_llm | StrOutputParser()
    final_answer = await final_chain.ainvoke(final_prompt)
    
    return final_answer

# ==========================================
# [메인 실행 로직]
# ==========================================
async def main():
    # 실험용 가상 마크다운 보고서 데이터
    markdown_reports = [
        {
            "version": "v1.0.0",
            "text": """# 📱 [넷플릭스] 버전별 심층 분석 보고서
## 1. 📑 보고서 개요
| 항목 | 내용 |
| :--- | :--- |
| **분석 대상 버전** | v1.0.0 |
| **사용자 평점** | 3.5 / 5.0 |

## 2. 📊 종합 요약
### 2.1 총평
초기 런칭 버전으로 콘텐츠 양은 만족스러우나, 앱 실행 속도가 느리고 UI 내비게이션이 복잡하다는 의견이 많음.

## 3. 🚨 상세 이슈 분석
### 3.1 재생 끊김 (언급량: 상, 부정 비율: 45%)
**💬 대표 VOC**
> 💥 **Problem**: "영상을 보다가 자꾸 버퍼링이 걸려서 몰입도가 떨어져요."
**🔧 개선 가이드라인**
- 서버 캐싱 로직 최적화 필요.
"""
        },
        {
            "version": "v2.0.0",
            "text": """# 📱 [넷플릭스] 버전별 심층 분석 보고서
## 1. 📑 보고서 개요
| 항목 | 내용 |
| :--- | :--- |
| **분석 대상 버전** | v2.0.0 |
| **사용자 평점** | 4.5 / 5.0 |

## 2. 📊 종합 요약
### 2.1 총평
UI 개편을 통해 사용성이 크게 개선됨. 특히 v1.0.0에서 지적된 버퍼링 문제가 거의 해결되어 긍정적 반응이 지배적임.

## 3. 🚨 상세 이슈 분석
### 3.1 자막 가독성 (언급량: 중, 부정 비율: 20%)
**💬 대표 VOC**
> 💥 **Problem**: "자막 크기가 너무 작아서 태블릿으로 볼 때 불편해요."
**🔧 개선 가이드라인**
- 자막 크기 및 배경색 커스텀 설정 기능 추가 검토.
"""
        }
    ]
    
    # 1. 문서 적재
    vector_store = await ingest_markdown_reports(markdown_reports)
    
    # 2. 검색 파이프라인 구성
    all_docs = vector_store.get()["documents"]
    retriever = get_search_pipeline(vector_store, all_docs)
    
    # 3. 사용자 질문: 버전별 비교 질문
    query = "넷플릭스 v1.0.0과 v2.0.0을 비교했을 때, 사용자 평점이 어떻게 달라졌나요?"
    
    # 4. 검색 및 답변 생성
    retrieved_docs = retriever.invoke(query)
    context = "\n".join([f"- {d.page_content} (출처: {d.metadata.get('version')})" for d in retrieved_docs])
    
    final_answer = await generate_final_answer(query, context)
    
    print("\n" + "="*50)
    print("📝 [최종 분석 리포트]")
    print("="*50)
    print(final_answer)

if __name__ == "__main__":
    asyncio.run(main())