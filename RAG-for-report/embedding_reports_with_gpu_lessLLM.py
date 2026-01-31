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

# Reranker
reranker_model = HuggingFaceCrossEncoder(
    model_name="dragonkue/bge-reranker-v2-m3-ko",
    model_kwargs={'device': device}
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

class MetadataEnsemble:
    def __init__(self, shared_embeddings):
        """
        shared_embeddings: 전역 임베딩 모델 재사용
        """
        self.okt = Okt()
        print("  └ KeyBERT 초기화 중...")
        # HuggingFaceEmbeddings의 내부 모델에 접근
        try:
            # _client 속성 시도
            self.kw_model = KeyBERT(model=shared_embeddings._client)
        except AttributeError:
            # 실패 시 직접 로드
            print("    (별도 SentenceTransformer 로드)")
            kw_sentence_model = SentenceTransformer("BAAI/bge-m3", device=device)
            self.kw_model = KeyBERT(model=kw_sentence_model)
    
    def _extract_keywords(self, text, top_n=15):
        """키워드 추출 (KeyBERT + Okt)"""
        try:
            nouns = " ".join(self.okt.nouns(text))
            if not nouns.strip():
                return []
            
            keywords = self.kw_model.extract_keywords(
                nouns, 
                keyphrase_ngram_range=(1, 2), 
                stop_words=None, 
                top_n=top_n
            )
            return [k[0] for k in keywords]
        except Exception as e:
            print(f"      ⚠️ KeyBERT 실패: {e}")
            return []

    async def generate_report_metadata(self, report_text, app_name, version):
        """
        보고서 전체(10000자 이하)에 대해 LLM 1회 호출
        성공 시에만 메타데이터 반환, 실패 시 None 반환
        """
        print(f"    🔍 KeyBERT 키워드 추출...", end=" ", flush=True)
        keywords = self._extract_keywords(report_text, top_n=15)
        print(f"{len(keywords)}개 완료")
        
        # LLM 프롬프트 (보고서 전체 입력)
        llm_prompt = f"""당신은 앱 리뷰 분석 전문가입니다.
다음은 [{app_name}] v{version}의 사용자 리뷰 분석 보고서 전문입니다.

보고서를 분석하여 **반드시 유효한 JSON 형식으로만** 응답하세요.

[보고서]:
{report_text}

[출력 형식 - JSON만 출력, 다른 텍스트 절대 금지]:
{{
    "sentiment": "긍정/부정/중립",
    "features": ["주요기능1", "주요기능2", "주요기능3"]
}}
"""
        
        print(f"    🤖 LLM 메타데이터 분석...", end=" ", flush=True)
        try:
            # LLM 호출 (타임아웃 60초)
            response = await asyncio.wait_for(
                llm_drafter.ainvoke(llm_prompt),
                timeout=60.0
            )
            
            # JSON 파싱
            content = response.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if not json_match:
                print("❌ JSON 형식 없음")
                return None
            
            llm_data = json.loads(json_match.group())
            
            # 필수 필드 검증
            if "sentiment" not in llm_data or "features" not in llm_data:
                print("❌ 필수 필드 누락")
                return None
            
            # features가 리스트인지 확인
            if not isinstance(llm_data["features"], list):
                print("❌ features가 리스트 아님")
                return None
            
            print("✅ 완료")
            
            # 데이터 정제
            sentiment = str(llm_data["sentiment"])
            features = [str(f) for f in llm_data["features"] if f]
            
            metadata = {
                "keywords": ", ".join(keywords[:15]),
                "sentiment": sentiment,
                "features": ", ".join(features[:5]) if features else "없음"
            }
            
            return metadata
            
        except asyncio.TimeoutError:
            print("❌ 타임아웃")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 실패: {str(e)[:30]}")
            return None
        except Exception as e:
            print(f"❌ LLM 오류: {str(e)[:40]}")
            return None

async def fetch_new_reports_from_db():
    """아직 처리되지 않은 보고서 목록 가져오기"""
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
    """DB 타임스탬프 업데이트"""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        query = "UPDATE analytics SET an_vectorized_at = %s WHERE an_idx = %s"
        cursor.execute(query, (now, an_idx))
        conn.commit()
    except Exception as e:
        print(f"      ❌ DB 업데이트 실패: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def clear_memory_cache():
    """메모리 캐시 정리"""
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

async def ingest_db_to_vector():
    db_reports = await fetch_new_reports_from_db()
    
    if not db_reports:
        print("🎉 모든 보고서가 이미 처리되었습니다.")
        return

    print(f"📦 신규 보고서 {len(db_reports)}개를 순차 처리합니다.")

    # ChromaDB 초기화
    print("🔧 ChromaDB 초기화 중...", end=" ", flush=True)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("완료")

    # MetadataEnsemble 초기화
    extractor = MetadataEnsemble(shared_embeddings=embeddings)
    
    # 스플리터 설정
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "report_title"), ("##", "category"), ("###", "sub_category")
    ])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    # 통계
    success_count = 0
    failed_count = 0

    # 보고서 처리 루프
    for idx, row in enumerate(tqdm(db_reports, desc="Processing Reports"), 1):
        an_idx = row['an_idx']
        
        try:
            print(f"\n  📄 보고서 ID {an_idx} 처리 시작...")
            
            # 날짜 처리
            dt = row['latest_review_date']
            if dt is None:
                print(f"    ⚠️ 날짜 정보 없음: 스킵")
                failed_count += 1
                continue
            
            year_int = int(dt.year)
            month_int = int(dt.month)
            quarter_int = (month_int - 1) // 3 + 1
            quarter_id = f"{year_int}-Q{quarter_int}"
            full_date = dt.strftime('%Y-%m-%d')

            # ★★★ LLM 메타데이터 추출 (1회 호출) ★★★
            report_metadata = await extractor.generate_report_metadata(
                report_text=row['report_markdown'],
                app_name=row['app_name'],
                version=row['version']
            )
            
            # LLM 실패 시 보고서 전체 스킵
            if report_metadata is None:
                print(f"    ⚠️ 메타데이터 추출 실패 → 이 보고서는 건너뜀 (다음 실행 시 재시도)")
                failed_count += 1
                continue

            # 텍스트 분할
            print(f"    🔪 텍스트 분할...", end=" ", flush=True)
            current_report_chunks = []
            header_splits = md_splitter.split_text(row['report_markdown'])
            
            chunk_count = 0
            for doc in header_splits:
                sub_chunks = text_splitter.split_documents([doc])
                for chunk in sub_chunks:
                    chunk_count += 1
                    
                    # 기본 메타데이터
                    chunk.metadata.update({
                        "source_an_idx": an_idx,
                        "app_name": row['app_name'],
                        "version": row['version'],
                        "year": year_int,
                        "month": month_int,
                        "quarter": quarter_int,
                        "quarter_id": quarter_id,
                        "date": full_date,
                        "doc_level": "version",
                        "genre": row.get('ag_name', 'Unknown')
                    })
                    
                    # LLM 메타데이터 추가
                    chunk.metadata.update(report_metadata)
                    
                    current_report_chunks.append(chunk)
            
            print(f"{chunk_count}개 청크 생성")
            
            # ChromaDB 저장
            if current_report_chunks:
                print(f"    💾 ChromaDB 저장...", end=" ", flush=True)
                vector_store.add_documents(current_report_chunks)
                print("완료")
                
                print(f"    🗃️ DB 타임스탬프 업데이트...", end=" ", flush=True)
                update_single_report_timestamp(an_idx)
                print("완료")
                
                print(f"  ✅ ID {an_idx} 완료 ({chunk_count}개 청크)")
                success_count += 1
            else:
                print(f"    ⚠️ 청크 없음")
                failed_count += 1

            # 메모리 정리 (20개마다)
            if idx % 20 == 0:
                print(f"  🧹 메모리 정리...")
                clear_memory_cache()
                print(f"  📊 진행 상황: 성공 {success_count}개, 실패 {failed_count}개")

        except Exception as e:
            print(f"\n❌ 예상치 못한 에러 (ID: {an_idx}): {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            clear_memory_cache()
            continue

    # 최종 통계
    print("\n" + "="*60)
    print(f"✅ 처리 완료: {success_count}개")
    print(f"❌ 실패/스킵: {failed_count}개")
    print(f"📊 전체: {success_count + failed_count}개")
    print("="*60)
    
    if failed_count > 0:
        print(f"\n💡 실패한 {failed_count}개 보고서는 다음 실행 시 재시도됩니다.")
    
    print("\n🧹 최종 메모리 정리...")
    clear_memory_cache()
    print("✅ 모든 작업 완료!")

if __name__ == "__main__":
    try:
        asyncio.run(ingest_db_to_vector())
    except KeyboardInterrupt:
        print("\n🛑 사용자 중단")
        clear_memory_cache()
    except Exception as e:
        print(f"\n❌ 치명적 오류: {e}")
        import traceback
        traceback.print_exc()
        clear_memory_cache()