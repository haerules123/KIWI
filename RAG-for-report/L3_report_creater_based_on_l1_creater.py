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
MIN_CHUNKS_FOR_YEAR = 5  # 연간 리포트는 데이터가 더 많이 필요함

DB_CONFIG = {
    'host': os.getenv('host'),
    'user': os.getenv('user'),
    'password': os.getenv('passwd'),
    'database': os.getenv('dbname'),
    'autocommit': False
}

# 분석할 연도 설정
START_YEAR = 2023
END_YEAR = 2025  # 2026년은 아직 끝나지 않았으므로 제외 권장

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

def get_or_create_yearly_version(conn, app_id, year_str):
    """
    [L3 전용] 연도(예: '2024')를 나타내는 가상 버전을 생성/조회
    """
    cursor = conn.cursor(dictionary=True)
    try:
        # 1. 조회
        check_sql = "SELECT v_idx FROM version WHERE a_idx = %s AND v_version = %s"
        cursor.execute(check_sql, (app_id, year_str))
        row = cursor.fetchone()
        
        if row:
            return row['v_idx']
        
        # 2. 생성 (created_at 없이)
        insert_sql = "INSERT INTO version (a_idx, v_version) VALUES (%s, %s)"
        cursor.execute(insert_sql, (app_id, year_str))
        conn.commit()
        
        return cursor.lastrowid
        
    except mysql.connector.Error as e:
        if e.errno == 1062: # Duplicate entry
            cursor.execute(check_sql, (app_id, year_str))
            row = cursor.fetchone()
            if row: return row['v_idx']
        print(f"  ❌ 연간 버전 생성 실패: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()

def save_l3_report(app_id, year_str, report_text):
    """
    analytics 테이블에 L3 리포트 저장
    """
    conn = get_db_connection()
    if not conn: return False
    
    try:
        # 1. 연도용 가상 버전 ID (예: '2024')
        v_idx = get_or_create_yearly_version(conn, app_id, year_str)
        if not v_idx:
            return False

        cursor = conn.cursor()
        
        # 2. 저장 (an_level='L3')
        query = """
            INSERT INTO analytics (v_idx, an_text, an_level, an_vectorized_at) 
            VALUES (%s, %s, 'L3', NULL)
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
# 🔍 RAG 및 LLM 로직 (Yearly)
# ==============================================================================
def aggressive_gc():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass

def get_yearly_context(app_name, year_int, vector_store, reranker, top_k=30):
    """
    해당 연도(year_int)의 모든 L1 데이터를 검색
    """
    # 1. 검색어: 연간 흐름 파악을 위한 광범위한 쿼리
    search_queries = [
        f"{app_name} {year_int}년 주요 업데이트 및 이슈",
        f"{app_name} {year_int}년 사용자 피드백 종합",
        f"{app_name} {year_int}년 긍정 부정 평가"
    ]
    
    # 2. 메타데이터 필터 (Ingest 시 year를 int로 넣었음)
    search_filter = {
        "$and": [
            {"app_name": app_name},
            {"year": year_int}
        ]
    }
    
    unique_contents = set()
    all_docs = []
    
    # 3. 검색
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

    # 4. Reranking
    doc_texts = [d.page_content for d in all_docs]
    rerank_query = f"{app_name} {year_int}년 성과 분석"
    pairs = [[rerank_query, text] for text in doc_texts]
    
    try:
        scores = reranker.predict(pairs, batch_size=4, show_progress_bar=False) if reranker else [0.0] * len(pairs)
    except:
        scores = [0.0] * len(pairs)

    scored_docs = sorted(list(zip(all_docs, scores)), key=lambda x: x[1], reverse=True)
    
    final_context = []
    detected_quarters = set()
    
    # 상위 30개 사용
    for doc, score in scored_docs[:30]:
        q_id = doc.metadata.get('quarter_id', 'Unknown') # 분기 정보도 같이 수집
        if q_id != 'Unknown':
            detected_quarters.add(q_id)
            
        final_context.append({
            "text": doc.page_content,
            "quarter": q_id,
            "version": doc.metadata.get('version', 'Unknown'),
            "keywords": doc.metadata.get('keywords', ''),
            "sentiment": doc.metadata.get('sentiment', '')
        })
        
    return final_context, detected_quarters

def generate_l3_report(app_name, year_str, context_data, detected_quarters):
    if not context_data:
        return None

    quarters_str = ", ".join(sorted(list(detected_quarters)))
    context_json = json.dumps(context_data, ensure_ascii=False, indent=2)

    prompt = f"""
당신은 모바일 앱 비즈니스 전략가입니다.
아래 데이터는 **[{app_name}]**의 **[{year_str}년]** 전체 기간 동안 수집된 L1(버전별) 리포트들의 핵심 내용입니다.
이를 바탕으로 **연간 결산(L3) 보고서**를 작성하십시오.

🛑 [필수 제약 사항]
1. 보고서 제목: **"{app_name} {year_str}년 연간 종합 보고서"**
2. **분석 대상 분기({quarters_str})**를 개요에 명시하십시오.
3. 단순 나열이 아닌, 1년 동안의 **흐름(Trend)과 변화**에 집중하십시오.
4. 마크다운 형식을 사용하십시오.

[Context Data]
{context_json}

[보고서 양식]
# {app_name} {year_str}년 연간 종합 보고서

## 1. 📑 연간 개요 (Executive Summary)
*   **분석 연도**: {year_str}년
*   **포함된 분기**: {quarters_str}
*   **종합 평가**: (1년 간의 성장을 평가하는 3~4문장 요약)

## 2. 📈 연간 주요 변화 흐름 (Yearly Trend)
(시간 흐름에 따른 긍/부정 이슈의 변화 양상 서술)
*   **상반기**: ...
*   **하반기**: ...

## 3. 🚨 핵심 이슈 회고 (Critical Issues)
(한 해 동안 가장 치명적이었거나 반복된 문제점)
*   **Top 1**: ...
*   **Top 2**: ...

## 4. 🏆 올해의 성과 (Achievements)
(사용자들에게 가장 사랑받은 기능이나 성공적인 업데이트)
*   ...

## 5. 🔭 내년도 전략 제언 (Next Year Strategy)
(올해 데이터를 기반으로 내년에 집중해야 할 핵심 분야 제안)
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
        reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device, max_length=1024)
    except:
        reranker = None
    print("완료")

    target_apps = get_target_apps(TARGET_GENRE)
    target_years = [str(y) for y in range(START_YEAR, END_YEAR + 1)]
    
    print(f"📊 대상 앱: {len(target_apps)}개")
    print(f"📅 대상 연도: {target_years}")

    for app in tqdm(target_apps, desc="App Loop"):
        app_name = app['a_name']
        app_id = app['a_idx']
        
        print(f"\n📱 [{app_name}] L3 분석 시작")
        
        for year_str in target_years:
            aggressive_gc()
            year_int = int(year_str)
            
            # 1. RAG 검색 (Year Filter)
            context, quarters = get_yearly_context(
                app_name, year_int, vector_store, reranker
            )
            
            if len(context) < MIN_CHUNKS_FOR_YEAR:
                # print(f"  ⏭️ {year_str}: 데이터 부족 ({len(context)} chunks)")
                continue
                
            print(f"  🔍 {year_str}: {len(context)}개 청크 확보 ({len(quarters)}개 분기 데이터)")
            
            # 2. LLM 생성
            report_text = generate_l3_report(app_name, year_str, context, quarters)
            
            if report_text:
                # 3. DB 저장 (L3)
                if save_l3_report(app_id, year_str, report_text):
                    print(f"    ✅ L3 리포트 저장 완료")
                else:
                    print(f"    ❌ 저장 실패")
            else:
                print(f"    ⚠️ 생성 실패")
                
            time.sleep(1)

    print("\n🎉 L3 연간 분석 완료!")

if __name__ == "__main__":
    main()