import mysql.connector
import asyncio
import os
import torch
import gc
import json
import re
from datetime import datetime
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt

from dotenv import load_dotenv 
load_dotenv()

# [설정]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PERSIST_DIRECTORY = "./chromadb_report"

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ 하드웨어 가속: {device.upper()}")

# 임베딩 모델 (전역)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
)

# LLM 설정
llm_drafter = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.3,
    google_api_key=GEMINI_API_KEY,
    timeout=60
)

# DB 설정
DB_CONFIG = {
    'host': os.getenv('host'),
    'user': os.getenv('user'),
    'password': os.getenv('passwd'),
    'database': os.getenv('dbname')
}

# (MetadataEnsemble 클래스는 기존과 동일하므로 생략 가능, 그대로 두시면 됩니다)
class MetadataEnsemble:
    def __init__(self, shared_embeddings):
        self.okt = Okt()
        print("  └ KeyBERT 초기화 중...")
        try:
            self.kw_model = KeyBERT(model=shared_embeddings._client)
        except AttributeError:
            kw_sentence_model = SentenceTransformer("BAAI/bge-m3", device=device)
            self.kw_model = KeyBERT(model=kw_sentence_model)
    
    def _extract_keywords(self, text, top_n=15):
        try:
            nouns = " ".join(self.okt.nouns(text))
            if not nouns.strip(): return []
            keywords = self.kw_model.extract_keywords(nouns, keyphrase_ngram_range=(1, 2), top_n=top_n)
            return [k[0] for k in keywords]
        except: return []

    async def generate_report_metadata(self, report_text, app_name, version):
        # (기존 로직 동일)
        keywords = self._extract_keywords(report_text, top_n=15)
        llm_prompt = f"""당신은 앱 리뷰 분석 전문가입니다.
다음은 [{app_name}] v{version}의 사용자 리뷰 분석 보고서 전문입니다.
보고서를 분석하여 **반드시 유효한 JSON 형식으로만** 응답하세요.
[보고서]: {report_text[:3000]}... (생략)
[출력 형식]: {{ "sentiment": "긍정/부정/중립", "features": ["기능1", "기능2"] }}"""
        try:
            response = await asyncio.wait_for(llm_drafter.ainvoke(llm_prompt), timeout=60.0)
            content = response.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match: return None
            llm_data = json.loads(json_match.group())
            return {
                "keywords": ", ".join(keywords[:15]),
                "sentiment": str(llm_data.get("sentiment", "중립")),
                "features": ", ".join([str(f) for f in llm_data.get("features", [])][:5])
            }
        except: return None

# ==============================================================================
# 🚀 수정된 DB Fetch 함수 (L2~L4 포함)
# ==============================================================================
async def fetch_new_reports_from_db():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    
    # 1. review 테이블 조인을 LEFT JOIN으로 변경
    # 2. an_level 추가 조회
    query = """
    SELECT 
        an.an_idx,
        an.an_level,
        a.a_name as app_name,
        a.ag_idx,
        ag.ag_name,
        v.v_version as version,
        an.an_text as report_markdown,
        MIN(r.r_date) as latest_review_date
    FROM analytics an
    JOIN version v ON an.v_idx = v.v_idx
    JOIN app a ON v.a_idx = a.a_idx
    JOIN app_genre ag ON a.ag_idx = ag.ag_idx
    LEFT JOIN review r ON v.v_idx = r.v_idx  -- 🔥 LEFT JOIN으로 변경
    WHERE an.an_vectorized_at IS NULL
    GROUP BY an.an_idx;
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def update_single_report_timestamp(an_idx):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        cursor.execute("UPDATE analytics SET an_vectorized_at = %s WHERE an_idx = %s", (now, an_idx))
        conn.commit()
    except: conn.rollback()
    finally:
        cursor.close()
        conn.close()

def clear_memory_cache():
    if device == "cuda": torch.cuda.empty_cache()
    gc.collect()

# ==============================================================================
# 🚀 메인 처리 로직 (날짜 파싱 로직 추가)
# ==============================================================================
async def ingest_db_to_vector():
    db_reports = await fetch_new_reports_from_db()
    
    if not db_reports:
        print("🎉 모든 보고서가 이미 처리되었습니다.")
        return

    print(f"📦 신규 보고서 {len(db_reports)}개를 순차 처리합니다.")

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    extractor = MetadataEnsemble(shared_embeddings=embeddings)
    
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "report_title"), ("##", "category"), ("###", "sub_category")
    ])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    success_count = 0
    failed_count = 0

    for idx, row in enumerate(tqdm(db_reports, desc="Processing Reports"), 1):
        an_idx = row['an_idx']
        an_level = row['an_level']
        version_str = row['version']
        
        try:
            print(f"\n  📄 [{an_level}] ID {an_idx} ({row['app_name']} v{version_str}) 처리 중...")
            
            # 🔥 [핵심 수정] 날짜 및 메타데이터 계산 로직
            meta_year = 0
            meta_month = 0
            meta_quarter = 0
            meta_quarter_id = "Unknown"
            meta_date = "Unknown"

            # Case 1: L1 (실제 리뷰 날짜 기반)
            if an_level == 'L1':
                dt = row['latest_review_date']
                if dt:
                    meta_year = int(dt.year)
                    meta_month = int(dt.month)
                    meta_quarter = (meta_month - 1) // 3 + 1
                    meta_quarter_id = f"{meta_year}-Q{meta_quarter}"
                    meta_date = dt.strftime('%Y-%m-%d')
                else:
                    # L1인데 날짜가 없으면 문제
                    print("    ⚠️ L1인데 날짜 정보 없음 (SKIP)")
                    failed_count += 1
                    continue

            # Case 2: L2 (분기 리포트, 예: '2024-Q1')
            elif an_level == 'L2':
                try:
                    # version_str = '2024-Q1'
                    y_str, q_str = version_str.split('-Q')
                    meta_year = int(y_str)
                    meta_quarter = int(q_str)
                    meta_month = (meta_quarter - 1) * 3 + 1
                    meta_quarter_id = version_str
                    meta_date = f"{meta_year}-{meta_month:02d}-01" # 해당 분기 첫날로 설정
                except:
                    print(f"    ⚠️ L2 버전 형식 오류 ({version_str})")
                    failed_count += 1
                    continue

            # Case 3: L3 (연간 리포트, 예: '2024')
            elif an_level == 'L3':
                try:
                    meta_year = int(version_str)
                    meta_quarter = 0 # 연간은 분기 없음
                    meta_quarter_id = f"{meta_year}-ALL"
                    meta_date = f"{meta_year}-01-01"
                except:
                    print(f"    ⚠️ L3 버전 형식 오류 ({version_str})")
                    failed_count += 1
                    continue

            # Case 4: L4 (종합 리포트, 예: 'TOTAL')
            elif an_level == 'L4':
                meta_year = 9999
                meta_quarter_id = "TOTAL"
                meta_date = datetime.now().strftime('%Y-%m-%d') # 처리 시점 날짜

            # LLM 메타데이터 추출
            report_metadata = await extractor.generate_report_metadata(
                report_text=row['report_markdown'],
                app_name=row['app_name'],
                version=version_str
            )
            
            if report_metadata is None:
                print(f"    ⚠️ 메타데이터 추출 실패 (SKIP)")
                failed_count += 1
                continue

            # 텍스트 분할 및 청크 생성
            current_report_chunks = []
            header_splits = md_splitter.split_text(row['report_markdown'])
            
            for doc in header_splits:
                sub_chunks = text_splitter.split_documents([doc])
                for chunk in sub_chunks:
                    # 공통 메타데이터
                    chunk.metadata.update({
                        "source_an_idx": an_idx,
                        "app_name": row['app_name'],
                        "version": version_str,
                        "an_level": an_level,  # 레벨 정보 추가
                        "year": meta_year,
                        "month": meta_month,
                        "quarter": meta_quarter,
                        "quarter_id": meta_quarter_id,
                        "date": meta_date,
                        "doc_level": an_level, # doc_level도 an_level로 맞춤
                        "genre": row.get('ag_name', 'Unknown')
                    })
                    # LLM 메타데이터 병합
                    chunk.metadata.update(report_metadata)
                    current_report_chunks.append(chunk)
            
            if current_report_chunks:
                vector_store.add_documents(current_report_chunks)
                update_single_report_timestamp(an_idx)
                print(f"  ✅ 완료 ({len(current_report_chunks)} 청크)")
                success_count += 1
            else:
                print("    ⚠️ 생성된 청크 없음")
                failed_count += 1

            if idx % 20 == 0: clear_memory_cache()

        except Exception as e:
            print(f"\n❌ 에러 (ID: {an_idx}): {e}")
            failed_count += 1
            continue

    print("\n" + "="*60)
    print(f"✅ 처리 완료: {success_count}개")
    print(f"❌ 실패/스킵: {failed_count}개")
    print("="*60)
    clear_memory_cache()

if __name__ == "__main__":
    asyncio.run(ingest_db_to_vector())