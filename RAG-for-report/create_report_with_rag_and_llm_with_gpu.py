import os
import sys
import gc
import ctypes
import json
import time
import warnings
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import mysql.connector
from mysql.connector.errors import OperationalError
import torch
import google.generativeai as genai

# ==============================================================================
# 🛑 [CRITICAL] 시스템 설정 (변경 없음)
# ==============================================================================
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["GRPC_POLL_STRATEGY"] = "epoll1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["DUCKDB_THREADS"] = "1"
os.environ["CHROMA_OTEL_COLLECTION_ENDPOINT"] = ""

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# ==============================================================================

from utils.logger import setup_logger

# Logger Setup
logger = setup_logger("rag_report_gpu")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# torch 설정
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

load_dotenv()

# [⚙️ 설정값]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PERSIST_DIRECTORY = "./chromadb_review_line"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "Dongjin-kr/ko-reranker"

TARGET_GENRE = "엔터테인먼트"  
MIN_REVIEW_LINES = 10  # SQL 필터링 기준

MAX_BATCH_SIZE = 2
RERANK_BATCH_LIMIT = 10
SLEEP_BETWEEN_BATCH = 3
SLEEP_AFTER_ERROR = 5

DB_CONFIG = {
    'host': os.getenv('host'),
    'user': os.getenv('user'),
    'password': os.getenv('passwd'),
    'database': os.getenv('dbname'),
    'autocommit': False,
    'connection_timeout': 10,
    'pool_size': 1,
    'pool_reset_session': True
}

# 전역 변수
device = "cpu"
reranker = None
vector_store = None
embeddings = None

def initialize_runtime():
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️ [SYSTEM] 실행 장치: {device.upper()}")
    try:
        if device == "cuda":
            try:
                print(f"   └ GPU: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass

        logger.info(f"📂 [DEBUG] Vector DB: {os.path.abspath(PERSIST_DIRECTORY)}")
        if not os.path.exists(PERSIST_DIRECTORY):
            logger.critical("🚨 [CRITICAL] DB 폴더가 없습니다!")
            return False

        try:
            genai.configure(api_key=GEMINI_API_KEY)
        except Exception:
            pass

        return True
    except Exception as e:
        logger.error(f"❌ 초기화 실패: {e}")
        return False

def get_db_connection():
    max_retries = 5
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            return conn
        except (OperationalError, Exception) as e:
            logger.warning(f"⚠️ DB 연결 실패 ({attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"❌ DB 연결 최종 실패: {e}")
                return None

def close_db_safely(conn, cursor=None):
    try:
        if cursor: cursor.close()
    except: pass
    try:
        if conn and conn.is_connected(): conn.close()
    except: pass

# ==============================================================================
# 🚀 [OPTIMIZED] 최적화된 SQL 쿼리 함수
# ==============================================================================
def get_target_versions(genre, min_lines):
    conn = get_db_connection()
    if not conn:
        return []
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        # 1. analytics 테이블과 JOIN하여 이미 존재하는 리포트 제외 (an.v_idx IS NULL)
        # 2. review_line 수를 카운트하여 min_lines 이상인 것만 가져옴 (HAVING 절)
        # 3. 날짜 필터링이 필요하다면 WHERE 절의 주석을 해제하고 실제 컬럼명 사용
        query = """
            SELECT v.v_idx, v.v_version, a.a_name, ag.ag_name, COUNT(rl.rl_idx) as line_count
            FROM version v
            JOIN app a ON v.a_idx = a.a_idx
            JOIN app_genre ag ON a.ag_idx = ag.ag_idx
            LEFT JOIN analytics an ON v.v_idx = an.v_idx
            JOIN review r ON v.v_idx = r.v_idx
            JOIN review_line rl ON r.r_idx = rl.r_idx
            WHERE ag.ag_name = %s
              AND an.v_idx IS NULL
              -- AND v.created_at >= '2023-01-01'  -- ⚠️ [날짜 필터] 실제 DB 컬럼명 확인 후 주석 해제
            GROUP BY v.v_idx, v.v_version, a.a_name, ag.ag_name
            HAVING COUNT(rl.rl_idx) >= %s
            ORDER BY a.a_name ASC, v.v_idx DESC
        """
        cursor.execute(query, (genre, min_lines))
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"❌ 버전 조회 실패: {e}")
        return []
    finally:
        close_db_safely(conn, cursor)

def save_report_to_db(v_idx, report_text):
    conn = get_db_connection()
    if not conn:
        return False
    cursor = None
    try:
        cursor = conn.cursor()
        query = "INSERT INTO analytics (an_text, an_vectorized_at, v_idx) VALUES (%s, NULL, %s)"
        cursor.execute(query, (report_text, v_idx))
        conn.commit()
        return True
    except Exception as e:
        print(f"❌ DB 저장 실패: {e}")
        if conn:
            try: conn.rollback()
            except: pass
        return False
    finally:
        close_db_safely(conn, cursor)

def get_version_statistics(app_name, version):
    conn = get_db_connection()
    if not conn:
        return None, []
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT COUNT(DISTINCT r.r_idx) as total, AVG(r.r_score) as avg_rating
            FROM review r
            JOIN version v ON r.v_idx = v.v_idx
            JOIN app a ON v.a_idx = a.a_idx
            WHERE a.a_name = %s AND v.v_version = %s
        """, (app_name, version))
        base_stats = cursor.fetchone()
        
        cursor.execute("""
            SELECT 
                at.at_type as aspect,
                SUM(CASE WHEN et.et_type = '긍정' THEN 1 ELSE 0 END) as pos_count,
                SUM(CASE WHEN et.et_type = '부정' THEN 1 ELSE 0 END) as neg_count,
                COUNT(*) as total_count
            FROM review_line rl
            JOIN analysis an ON rl.rl_idx = an.rl_idx
            JOIN aspect_type at ON an.at_idx = at.at_idx
            JOIN emotion_type et ON an.et_idx = et.et_idx
            JOIN review r ON rl.r_idx = r.r_idx
            JOIN version v ON r.v_idx = v.v_idx
            JOIN app a ON v.a_idx = a.a_idx
            WHERE a.a_name = %s AND v.v_version = %s
            GROUP BY at.at_type
            ORDER BY total_count DESC
        """, (app_name, version))
        aspect_stats = cursor.fetchall()
        return base_stats, aspect_stats
    except Exception as e:
        print(f"❌ 통계 조회 실패: {e}")
        return None, []
    finally:
        close_db_safely(conn, cursor)

def get_rag_contexts(app_name, version, aspect, sentiment, global_used_texts, 
                     vector_store_instance, reranker_instance, top_k=15):
    query_text = f"{app_name} {aspect} {sentiment}"
    results = []

    for attempt in range(3):
        try:
            results = vector_store_instance.similarity_search(
                query=query_text, k=20,
                filter={"$and": [{"app_name": app_name}, {"version": version}]}
            )
            break
        except Exception as e:
            if attempt == 2: return []
            time.sleep(1)

    if not results: return []

    doc_texts = [doc.page_content for doc in results]
    valid_indices = [i for i, t in enumerate(doc_texts) if len(t) > 2]
    filtered_results = [results[i] for i in valid_indices]
    filtered_texts = [doc_texts[i] for i in valid_indices]

    if len(filtered_texts) > RERANK_BATCH_LIMIT:
        filtered_results = filtered_results[:RERANK_BATCH_LIMIT]
        filtered_texts = filtered_texts[:RERANK_BATCH_LIMIT]

    if not filtered_texts: return []

    rerank_query = f"{aspect} 관련 {sentiment} 의견"
    pairs = [[rerank_query, text] for text in filtered_texts]
    
    if reranker_instance:
        try:
            scores = reranker_instance.predict(pairs, batch_size=1)
            if device == "cuda": torch.cuda.empty_cache()
        except Exception:
            scores = [0.0] * len(pairs)
    else:
        scores = [0.0] * len(pairs)
    
    scored_docs = sorted(list(zip(filtered_results, scores)), key=lambda x: x[1], reverse=True)
    final_data = []
    
    for doc, score in scored_docs:
        if len(final_data) >= top_k: break
        if score < -4.0: continue
        text = doc.page_content
        if text not in global_used_texts:
            final_data.append({
                "text": text, 
                "date": doc.metadata.get('date', 'Unknown'), 
                "relevance_score": float(score)
            })
            global_used_texts.add(text)
    
    del pairs, scores, scored_docs, results, filtered_results
    gc.collect()
    return final_data

def generate_ai_report(app_name, version, json_data, total_reviews, avg_rating):
    context_str = json.dumps(json_data, ensure_ascii=False, indent=2)
    prompt = f"""
당신은 모바일 앱 QA 수석 컨설턴트입니다. 아래 데이터를 분석하여 보고서를 작성하십시오.

🛑 **작성 원칙**
1. **Format 준수**: 아래 Markdown 형식을 그대로 사용.
2. **창작 금지**: '유저의 목소리'는 JSON의 text, date를 그대로 인용.
3. **데이터 매핑**: negative_reviews는 섹션 2, positive_reviews는 섹션 3에 사용.

---
[CONTEXT JSON]
{context_str}

---
[OUTPUT FORMAT]
# 📱 [{app_name}] v{version} 심층 분석 보고서

## 📑 보고서 개요
| 항목 | 내용 |
| :--- | :--- |
| **분석 대상 버전** | {version} |
| **사용자 평점** | {avg_rating:.2f} |
| **분석 표본 수** | {total_reviews} 개의 유효 리뷰 |

---

## 1. 📊 Executive Summary
- **종합 점수**: (평점 {avg_rating:.2f} 기준 판정)
- **핵심 요약**: (3문장 내외 요약)
- **긴급 대응 과제**: (1줄 요약)

## 2. 🚨 이슈 심층 분석 (Deep Dive)
### 2.1 [Aspect 이름] (부정 [neg_ratio])
**💬 유저의 목소리 (Evidence)**
> "[text]" ([date])
**🕵️ 원인 추정**
- (추론)
**💡 개선 솔루션**
- **🔧 Tech**: (제안)
- **🎨 UX/UI**: (제안)

## 3. 🏆 긍정 요소 및 강화 전략
### 3.1 [Aspect 이름] (긍정 요소)
**💬 유저의 목소리**
> "[text]" ([date])
**🚀 강화 및 마케팅 전략**
- (제안)

## 4. 📝 총평 및 다음 버전 제안
- (방향성 제시)
"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt, 
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        return response.text
    except Exception as e:
        print(f"❌ AI 생성 실패: {e}")
        return None

def process_single_version(v_idx, app_name, version, genre, 
                          vector_store_instance, reranker_instance):
    """단일 버전 처리 (최적화: 중복 체크 제거됨)"""
    try:
        # DB에서 이미 필터링했으므로 즉시 통계 조회
        base_stats, aspect_stats = get_version_statistics(app_name, version)
        
        # 안전장치: 혹시라도 통계가 비어있으면 스킵
        if not base_stats or not base_stats['total']:
            print(f"  ⚠️ [{app_name} v{version}] 통계 데이터 없음 (Skipping)")
            return False
        
        print(f"  🔄 [{app_name} v{version}] 처리 시작 (리뷰: {base_stats['total']}개, 평점: {base_stats['avg_rating']:.2f})")

        total_reviews = base_stats['total']
        avg_rating = base_stats['avg_rating'] or 0.0
        global_used_texts = set()
        rag_data = {}
        
        sorted_aspects = sorted(
            aspect_stats, 
            key=lambda x: (x['neg_count'], x['total_count']), 
            reverse=True
        )[:6]

        # RAG 데이터 수집
        for idx, stat in enumerate(sorted_aspects, 1):
            aspect = stat['aspect']
            total_cnt = stat['total_count']
            neg_count = stat['neg_count']
            neg_ratio = round((neg_count / total_cnt) * 100, 1) if total_cnt > 0 else 0.0
            
            print(f"    └ Aspect {idx}/6: {aspect} (부정 {neg_ratio}%)", end=" ")
            
            aspect_data = {
                "stats": {"total": total_cnt, "neg_ratio": f"{neg_ratio}%"},
                "negative_reviews": [],
                "positive_reviews": []
            }
            
            if neg_count > 0:
                neg_vocs = get_rag_contexts(
                    app_name, version, aspect, "부정", global_used_texts,
                    vector_store_instance, reranker_instance, top_k=3
                )
                aspect_data["negative_reviews"] = [{"text": v['text'], "date": v['date']} for v in neg_vocs]
                print(f"부정:{len(neg_vocs)}개", end=" ")
                
            if (total_cnt - neg_count) > 0:
                pos_vocs = get_rag_contexts(
                    app_name, version, aspect, "긍정", global_used_texts,
                    vector_store_instance, reranker_instance, top_k=3
                )
                aspect_data["positive_reviews"] = [{"text": v['text'], "date": v['date']} for v in pos_vocs]
                print(f"긍정:{len(pos_vocs)}개")
                
            if aspect_data["negative_reviews"] or aspect_data["positive_reviews"]:
                rag_data[aspect] = aspect_data
            else:
                print("데이터 없음")

        if not rag_data:
            print(f"  ❌ [{app_name} v{version}] RAG 데이터 부족으로 스킵")
            return False

        print(f"  🤖 AI 보고서 생성 중...", end=" ", flush=True)
        report_md = generate_ai_report(app_name, version, rag_data, total_reviews, avg_rating)
        
        if not report_md:
            print("실패!")
            return False
        
        print(f"성공! ({len(report_md)}자)")

        print(f"  💾 DB 저장 중...", end=" ", flush=True)
        success = save_report_to_db(v_idx, report_md)
        
        if success:
            print("성공! ✅ 완료")
        else:
            print("실패! ❌ 저장 에러")
        
        del rag_data
        gc.collect()
        return success
        
    except Exception as e:
        print(f"\n  ❌ [{app_name} v{version}] 처리 중 예외: {e}")
        gc.collect()
        return False

def aggressive_gc():
    gc.collect()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(1)

def main():
    global reranker, vector_store, embeddings
    
    aggressive_gc()
    if not initialize_runtime():
        return
    
    print(f"🔄 분석 시작 | 장르: {TARGET_GENRE} | 최소 리뷰: {MIN_REVIEW_LINES}줄 이상")
    print(f"⚙️ CPU 스레드: {torch.get_num_threads()}")
    
    # 🚀 DB에서 필터링된 목록 가져오기
    targets = get_target_versions(TARGET_GENRE, MIN_REVIEW_LINES)
    print(f"📚 실제 분석 대상: {len(targets)}개 (이미 완료된 건 및 데이터 부족 제외됨)")
    
    if not targets:
        print("✅ 처리할 대상이 없습니다. 종료합니다.")
        return

    # 모델 로드
    try:
        logger.info(f"🔄 Embeddings 로드...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
        )
        aggressive_gc()

        print(f"🔄 Vector Store 로드...", flush=True)
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name="review_sentences"
        )
        aggressive_gc()
        
        print(f"🔄 Reranker 로드...", flush=True)
        try:
            reranker = CrossEncoder(
                RERANKER_MODEL_NAME, 
                device=device,
                max_length=256,
                num_labels=1,
                automodel_args={'trust_remote_code': True}
            )
        except TypeError:
            reranker = CrossEncoder(
                RERANKER_MODEL_NAME, 
                device=device,
                max_length=256
            )
        aggressive_gc()

    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    success_count = 0
    skip_count = 0
    error_count = 0
    batch_counter = 0
    
    pbar = tqdm(targets, desc="진행")
    
    for target in pbar:
        v_idx = target['v_idx']
        app_name = target['a_name']
        version = target['v_version']
        
        try:
            # 중복 체크 로직이 제거되어 바로 처리
            is_success = process_single_version(
                v_idx, app_name, version, target['ag_name'],
                vector_store, reranker
            )
            
            if is_success:
                success_count += 1
                pbar.set_postfix({"✅": success_count, "❌": error_count})
            else:
                skip_count += 1
            
            batch_counter += 1
            if batch_counter >= MAX_BATCH_SIZE:
                aggressive_gc()
                time.sleep(SLEEP_BETWEEN_BATCH)
                batch_counter = 0
                
        except KeyboardInterrupt:
            print("\n🛑 중단됨")
            break
        except Exception as e:
            error_count += 1
            print(f"\n❌ {app_name} v{version}: {e}")
            aggressive_gc()
            time.sleep(SLEEP_AFTER_ERROR)
            continue

    print("\n🧹 정리 중...")
    try:
        del reranker, vector_store, embeddings
    except: pass
    
    aggressive_gc()
    logger.info(f"✅ 최종 완료: {success_count}개 | 실패/스킵: {skip_count+error_count}개")

if __name__ == "__main__":
    main()