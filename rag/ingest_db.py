import mysql.connector
import asyncio
import os
from datetime import datetime
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from keybert import KeyBERT
from konlpy.tag import Okt
from tqdm import tqdm
import json
import re

from dotenv import load_dotenv 
load_dotenv()

# [설정]
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PERSIST_DIRECTORY = "./chroma_advanced_db"

embeddings = OllamaEmbeddings(model="bge-m3") 
reranker_model = HuggingFaceCrossEncoder(model_name="dragonkue/bge-reranker-v2-m3-ko")

judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
drafter_models = {
    "scout": ChatGroq(model="llama-3.1-8b-instant", temperature=0.7),
    "qwen": ChatGroq(model="qwen/qwen3-32b", temperature=0.7)
}

# DB 설정
DB_CONFIG = {
    'host': os.getenv('host'),
    'user': os.getenv('user'),
    'password': os.getenv('passwd'),
    'database': os.getenv('dbname')
}

class MetadataEnsemble:
    def __init__(self):
        self.kw_model = KeyBERT()
        self.okt = Okt()
    
    def _extract_local_keywords(self, text):
        nouns = " ".join(self.okt.nouns(text))
        # 텍스트가 너무 짧으면 키워드 추출 시 에러가 날 수 있으므로 예외처리
        if not nouns.strip():
            return []
        try:
            keywords = self.kw_model.extract_keywords(nouns, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=5)
            return [k[0] for k in keywords]
        except:
            return []

    async def generate_metadata(self, text):
        # 1. 로컬 추출
        local_keywords = self._extract_local_keywords(text)
        
        # 2. LLM 추출 (Groq)
        llm_prompt = f"""
        당신은 앱 리뷰 분석 전문가입니다. 다음 텍스트에서 핵심 정보를 JSON 형식으로 추출하세요.
        
        [텍스트]: {text}
        
        [응답 형식]:
        {{
            "keywords": ["키워드1", "키워드2"],
            "summary": "한 줄 요약",
            "sentiment": "긍정/부정/중립",
            "features": ["기능1", "기능2"]
        }}
        """
        try:
            llm_res = await drafter_models["scout"].ainvoke(llm_prompt)
            json_match = re.search(r'\{.*\}', llm_res.content, re.DOTALL)
            if json_match:
                llm_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")
        except Exception as e:
            llm_data = {"keywords": [], "summary": text[:50], "sentiment": "중립", "features": []}

        # 데이터 정제 (기존 로직 유지)
        raw_llm_keywords = llm_data.get("keywords", [])
        clean_llm_keywords = []
        if isinstance(raw_llm_keywords, list):
            for k in raw_llm_keywords:
                if isinstance(k, str): clean_llm_keywords.append(k)
                elif isinstance(k, dict): clean_llm_keywords.extend([str(v) for v in k.values()])
                else: clean_llm_keywords.append(str(k))
        
        raw_features = llm_data.get("features", [])
        clean_features = []
        if isinstance(raw_features, list):
            for f in raw_features:
                if isinstance(f, str): clean_features.append(f)
                elif isinstance(f, dict): clean_features.extend([str(v) for v in f.values()])
                else: clean_features.append(str(f))

        final_keywords = list(set(local_keywords) | set(clean_llm_keywords))
        
        metadata = {
            "keywords": ", ".join([str(k) for k in final_keywords]),
            "summary": str(llm_data.get("summary", text[:100])),
            "sentiment": str(llm_data.get("sentiment", "알 수 없음")),
            "features": ", ".join([str(f) for f in clean_features])
        }
        
        return metadata

async def fetch_new_reports_from_db():
    """아직 처리되지 않은(an_vectorized_at IS NULL) 보고서 목록 가져오기"""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    
    query = """
    SELECT 
        an.an_idx,
        a.a_name as app_name,
        v.v_version as version,
        an.an_text as report_markdown,
        MIN(r.r_date) as latest_review_date
    FROM analytics an
    JOIN version v ON an.v_idx = v.v_idx
    JOIN app a ON v.a_idx = a.a_idx
    JOIN review r ON v.v_idx = r.v_idx
    WHERE an.an_vectorized_at IS NULL
    GROUP BY an.an_idx;
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    cursor.close()
    conn.close()
    return rows

def update_single_report_timestamp(an_idx):
    """
    [변경] 단일 보고서에 대해 처리 완료 시간(TimeStamp)을 DB에 업데이트
    (async 함수 내에서 호출되지만, mysql.connector는 동기식이므로 일반 함수로 작성)
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        query = "UPDATE analytics SET an_vectorized_at = %s WHERE an_idx = %s"
        cursor.execute(query, (now, an_idx))
        conn.commit()
        # print(f"  └ [DB] ID {an_idx} 업데이트 완료") # 로그가 너무 많으면 주석 처리
    except Exception as e:
        print(f"  └ ❌ [DB] ID {an_idx} 업데이트 실패: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

async def ingest_db_to_vector():
    # 1. DB에서 처리 대상 데이터 로드
    db_reports = await fetch_new_reports_from_db()
    
    if not db_reports:
        print("🎉 모든 보고서가 이미 처리되었습니다. (신규 데이터 없음)")
        return

    print(f"📦 신규 보고서 {len(db_reports)}개를 순차 처리합니다.")

    # 2. Chroma DB 초기화 (루프 밖에서 한 번만 로드)
    # persist_directory가 설정되어 있으므로, 데이터는 파일 시스템에 저장됩니다.
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    extractor = MetadataEnsemble()
    
    # 3. 스플리터 설정
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "report_title"), ("##", "category"), ("###", "sub_category")
    ])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    # 4. 보고서 1개씩 처리 -> 저장 -> DB업데이트
    for row in tqdm(db_reports, desc="Processing Reports"):
        an_idx = row['an_idx']
        
        try:
            # --- 날짜 및 기본 정보 처리 ---
            dt = row['latest_review_date']
            if dt is None:
                print(f"⚠️ 날짜 정보 없음: ID {an_idx} 스킵")
                continue
                
            year = str(dt.year)
            month = f"{dt.month:02d}"
            quarter = f"{(dt.month-1)//3 + 1}Q"
            full_date = dt.strftime('%Y-%m-%d')

            # --- 텍스트 분할 및 메타데이터 생성 ---
            current_report_chunks = []
            header_splits = md_splitter.split_text(row['report_markdown'])
            
            for doc in header_splits:
                sub_chunks = text_splitter.split_documents([doc])
                for chunk in sub_chunks:
                    # 기본 메타데이터
                    chunk.metadata.update({
                        "source_an_idx": an_idx,
                        "app_name": row['app_name'],
                        "version": row['version'],
                        "year": year,
                        "month": month,
                        "quarter": quarter,
                        "date": full_date
                    })
                    
                    # AI 메타데이터 (비동기)
                    meta_analysis = await extractor.generate_metadata(chunk.page_content)
                    chunk.metadata.update(meta_analysis)
                    
                    current_report_chunks.append(chunk)
            
            # --- [중요] 1개 보고서 처리 끝날 때마다 벡터 DB에 즉시 저장 ---
            if current_report_chunks:
                vector_store.add_documents(current_report_chunks)
                
                # --- [중요] MySQL DB에 즉시 업데이트 ---
                update_single_report_timestamp(an_idx)
            else:
                print(f"⚠️ ID {an_idx}: 생성된 청크가 없습니다.")

        except Exception as e:
            print(f"❌ 보고서 처리 중 치명적 에러 (ID: {an_idx}): {e}")
            # 에러 발생 시 해당 건은 넘어가고 다음 건을 처리 (DB 업데이트 안 함)
            continue

    print("✅ 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    asyncio.run(ingest_db_to_vector())