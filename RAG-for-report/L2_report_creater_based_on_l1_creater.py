import os
import sys
import gc
import json
import time
import ctypes
from tqdm import tqdm
import mysql.connector
import torch
import google.generativeai as genai
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# 환경 변수 로드
load_dotenv()

# ==============================================================================
# ⚙️ 설정값
# ==============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PERSIST_DIRECTORY = "./chromadb_report"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "dragonkue/bge-reranker-v2-m3-ko"
TARGET_GENRE = "엔터테인먼트"
MIN_CHUNKS_FOR_QUARTER = 3

DB_CONFIG = {
    'host': os.getenv('host'),
    'user': os.getenv('user'),
    'password': os.getenv('passwd'),
    'database': os.getenv('dbname'),
    'autocommit': False
}

# 분석할 기간 설정
START_YEAR = 2023
END_YEAR = 2026
END_QUARTER = 1

# 전역 변수
device = "cuda" if torch.cuda.is_available() else "cpu"
vector_store = None
reranker = None

# ==============================================================================
# 🛠️ DB 유틸리티 함수
# ==============================================================================
def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        return None

def close_db_safely(conn, cursor=None):
    try:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()
    except: pass

def get_or_create_quarter_version(conn, app_id, quarter_id):
    """
    [수정됨] version 테이블 스키마에 맞춰 created_at 제거
    (v_idx, v_version, a_idx) 만 존재함
    """
    cursor = conn.cursor(dictionary=True)
    try:
        # 1. 해당 분기 버전이 이미 있는지 확인
        check_sql = "SELECT v_idx FROM version WHERE a_idx = %s AND v_version = %s"
        cursor.execute(check_sql, (app_id, quarter_id))
        row = cursor.fetchone()
        
        if row:
            return row['v_idx']
        
        # 2. 없다면 생성 (created_at 제외)
        insert_sql = "INSERT INTO version (a_idx, v_version) VALUES (%s, %s)"
        cursor.execute(insert_sql, (app_id, quarter_id))
        conn.commit()
        
        return cursor.lastrowid
        
    except mysql.connector.Error as e:
        # 동시성 문제로 이미 존재할 경우 재조회
        if e.errno == 1062: # Duplicate entry
            cursor.execute(check_sql, (app_id, quarter_id))
            row = cursor.fetchone()
            if row: return row['v_idx']
            
        print(f"  ❌ 버전 생성 실패: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()

def save_l2_report(app_id, quarter_id, report_text):
    """
    analytics 테이블에 L2 리포트 저장
    """
    conn = get_db_connection()
    if not conn: return False
    
    try:
        # 1. 분기용 가상 버전 ID 가져오기
        v_idx = get_or_create_quarter_version(conn, app_id, quarter_id)
        if not v_idx:
            print(f"    ❌ 가상 버전 생성 실패: {quarter_id}")
            return False

        cursor = conn.cursor()
        
        # 2. INSERT ... ON DUPLICATE KEY UPDATE
        # v_idx + an_level 조합이 Unique Key (uk_version_level)
        query = """
            INSERT INTO analytics (v_idx, an_text, an_level, an_vectorized_at) 
            VALUES (%s, %s, 'L2', NULL)
            ON DUPLICATE KEY UPDATE 
                an_text = VALUES(an_text),
                an_vectorized_at = NULL
        """
        cursor.execute(query, (v_idx, report_text))
        conn.commit()
        return True
        
    except Exception as e:
        print(f"  ❌ DB 저장 실패: {e}")
        conn.rollback()
        return False
    finally:
        close_db_safely(conn)

def get_target_apps(genre):
    conn = get_db_connection()
    if not conn: return []
    try:
        cursor = conn.cursor(dictionary=True)
        sql = """
            SELECT a.a_idx, a.a_name 
            FROM app a
            JOIN app_genre ag ON a.ag_idx = ag.ag_idx
            WHERE ag.ag_name = %s
        """
        cursor.execute(sql, (genre,))
        return cursor.fetchall()
    finally:
        close_db_safely(conn)

# ==============================================================================
# 🔍 RAG 및 LLM 로직
# ==============================================================================
def get_target_quarters():
    quarters = []
    for year in range(START_YEAR, END_YEAR + 1):
        for q in range(1, 5):
            if year == END_YEAR and q > END_QUARTER:
                break
            quarters.append(f"{year}-Q{q}")
    return quarters

def aggressive_gc():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass

def get_quarter_context(app_name, quarter_id, vector_store, reranker, top_k=20):
    search_queries = [
        f"{app_name} {quarter_id} 주요 이슈 및 문제점",
        f"{app_name} {quarter_id} 사용자 긍정 반응",
        f"{app_name} {quarter_id} 업데이트 반응 기능"
    ]
    
    search_filter = {
        "$and": [
            {"app_name": app_name},
            {"quarter_id": quarter_id}
        ]
    }
    
    unique_contents = set()
    all_docs = []
    
    for query in search_queries:
        try:
            results = vector_store.similarity_search(
                query=query,
                k=top_k,
                filter=search_filter
            )
            for doc in results:
                if doc.page_content not in unique_contents:
                    unique_contents.add(doc.page_content)
                    all_docs.append(doc)
        except Exception:
            continue

    if not all_docs:
        return [], set()

    doc_texts = [d.page_content for d in all_docs]
    rerank_query = f"{app_name} {quarter_id} 종합 분석"
    pairs = [[rerank_query, text] for text in doc_texts]
    
    try:
        scores = reranker.predict(pairs, batch_size=4, show_progress_bar=False) if reranker else [0.0] * len(pairs)
    except:
        scores = [0.0] * len(pairs)

    scored_docs = sorted(list(zip(all_docs, scores)), key=lambda x: x[1], reverse=True)
    
    final_context = []
    detected_versions = set()
    
    for doc, score in scored_docs[:25]:
        ver = doc.metadata.get('version', 'Unknown')
        if ver != 'Unknown':
            detected_versions.add(ver)
            
        final_context.append({
            "text": doc.page_content,
            "version": ver,
            "keywords": doc.metadata.get('keywords', ''),
            "sentiment": doc.metadata.get('sentiment', '')
        })
        
    return final_context, detected_versions

def generate_l2_report(app_name, quarter_id, context_data, detected_versions):
    if not context_data:
        return None

    versions_str = ", ".join(sorted(list(detected_versions)))
    context_json = json.dumps(context_data, ensure_ascii=False, indent=2)
    year, q = quarter_id.split('-')
    quarter_title = f"{year}년 {q[1]}분기"

    prompt = f"""
당신은 모바일 앱 서비스 총괄 분석가입니다.
아래 제공된 데이터는 **[{app_name}]**의 **[{quarter_title}]** 기간 동안 발행된 상세 리포트(L1)들의 핵심 내용입니다.
이를 바탕으로 해당 분기의 성과를 결산하는 보고서(L2)를 작성하십시오.

🛑 [필수 제약 사항]
1. 보고서 제목은 **"{app_name} {quarter_title} 결산 보고서"**여야 합니다.
2. **참조된 버전({versions_str})**을 개요에 명시하십시오.
3. 없는 사실을 지어내지 말고, 제공된 Context 데이터에 기반하여 작성하십시오.
4. 마크다운 형식을 사용하십시오.

[Context Data]
{context_json}

[보고서 양식]
# {app_name} {quarter_title} 결산 보고서

## 1. 📑 개요
*   **분석 기간**: {quarter_title}
*   **참조 버전**: {versions_str}
*   **분기 요약**: (전체적인 분위기와 핵심 이슈 1줄 요약)

## 2. 📊 분기 핵심 이슈 (Key Issues)
(가장 빈번하거나 심각했던 문제점 위주)
*   **이슈 1**: ...
*   **이슈 2**: ...

## 3. 🏆 주요 긍정 반응 (Highlights)
(사용자 호평 요소)
*   ...

## 4. 🚀 차기 분기 제언
(L1 리포트들을 종합했을 때 필요한 개선 방향)
*   ...
"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        return response.text
    except Exception as e:
        print(f"❌ LLM 생성 에러: {e}")
        return None

# ==============================================================================
# 🚀 메인 실행
# ==============================================================================
def main():
    global vector_store, reranker
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    print("🔄 모델 로드 중...", end=" ")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    
    try:
        reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device, max_length=512)
    except:
        reranker = None
    print("완료")

    target_apps = get_target_apps(TARGET_GENRE)
    target_quarters = get_target_quarters()
    
    print(f"📊 대상 앱: {len(target_apps)}개")
    print(f"📅 대상 기간: {target_quarters[0]} ~ {target_quarters[-1]}")

    for app in tqdm(target_apps, desc="App Loop"):
        app_name = app['a_name']
        app_id = app['a_idx']
        
        print(f"\n📱 [{app_name}] 분석 시작")
        
        for quarter_id in target_quarters:
            aggressive_gc()
            
            # 1. RAG 검색
            context, versions = get_quarter_context(
                app_name, quarter_id, vector_store, reranker
            )
            
            if len(context) < MIN_CHUNKS_FOR_QUARTER:
                continue
                
            print(f"  🔍 {quarter_id}: {len(context)}개 청크 확보 (v{len(versions)}개 버전 참조)")
            
            # 2. LLM 리포트 생성
            report_text = generate_l2_report(app_name, quarter_id, context, versions)
            
            if report_text:
                # 3. DB 저장 (L2, 가상 버전 생성 포함)
                if save_l2_report(app_id, quarter_id, report_text):
                    print(f"    ✅ 저장 완료")
                else:
                    print(f"    ❌ 저장 실패")
            else:
                print(f"    ⚠️ 리포트 생성 실패")
                
            time.sleep(1)

    print("\n🎉 모든 작업 완료!")

if __name__ == "__main__":
    main()