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
MIN_CHUNKS_FOR_TOTAL = 10  # 전체 분석이므로 데이터가 더 많이 필요함

DB_CONFIG = {
    'host': os.getenv('host'),
    'user': os.getenv('user'),
    'password': os.getenv('passwd'),
    'database': os.getenv('dbname'),
    'autocommit': False
}

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

def get_or_create_total_version(conn, app_id):
    """
    [L4 전용] 서비스 전체를 의미하는 'TOTAL' 가상 버전 생성/조회
    (created_at 컬럼 사용 안 함)
    """
    version_name = "TOTAL"
    cursor = conn.cursor(dictionary=True)
    try:
        # 1. 조회
        check_sql = "SELECT v_idx FROM version WHERE a_idx = %s AND v_version = %s"
        cursor.execute(check_sql, (app_id, version_name))
        row = cursor.fetchone()
        
        if row:
            return row['v_idx']
        
        # 2. 생성
        insert_sql = "INSERT INTO version (a_idx, v_version) VALUES (%s, %s)"
        cursor.execute(insert_sql, (app_id, version_name))
        conn.commit()
        
        return cursor.lastrowid
        
    except mysql.connector.Error as e:
        if e.errno == 1062: # Duplicate entry
            cursor.execute(check_sql, (app_id, version_name))
            row = cursor.fetchone()
            if row: return row['v_idx']
        print(f"  ❌ Total 버전 생성 실패: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()

def save_l4_report(app_id, report_text):
    """
    analytics 테이블에 L4 리포트 저장
    """
    conn = get_db_connection()
    if not conn: return False
    
    try:
        # 1. TOTAL 가상 버전 ID 조회/생성
        v_idx = get_or_create_total_version(conn, app_id)
        if not v_idx:
            return False

        cursor = conn.cursor()
        
        # 2. 저장 (an_level='L4')
        query = """
            INSERT INTO analytics (v_idx, an_text, an_level, an_vectorized_at) 
            VALUES (%s, %s, 'L4', NULL)
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
# 🔍 RAG 및 LLM 로직 (All-Time / L4)
# ==============================================================================
def aggressive_gc():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass

def get_total_context(app_name, vector_store, reranker, top_k=40):
    """
    [L4] 시간 필터 없이 전체 데이터를 검색하여 앱의 '역사'와 '정체성'을 파악
    """
    # 1. 검색어: 과거부터 현재까지를 아우르는 포괄적 쿼리
    search_queries = [
        f"{app_name} 서비스 역사 및 주요 변화",
        f"{app_name} 고질적인 문제점과 해결 과정",
        f"{app_name} 사용자가 꼽는 최고의 기능 장점",
        f"{app_name} 업데이트 연혁 및 평가"
    ]
    
    # 2. 메타데이터 필터: 오직 앱 이름만! (날짜 제한 없음)
    search_filter = {"app_name": app_name}
    
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
        return [], set(), set()

    # 4. Reranking
    doc_texts = [d.page_content for d in all_docs]
    rerank_query = f"{app_name} 서비스 종합 평가 및 역사"
    pairs = [[rerank_query, text] for text in doc_texts]
    
    try:
        # L4는 데이터가 많을 수 있으니 배치 사이즈나 max_length 조절 주의
        scores = reranker.predict(pairs, batch_size=4, show_progress_bar=False) if reranker else [0.0] * len(pairs)
    except:
        scores = [0.0] * len(pairs)

    scored_docs = sorted(list(zip(all_docs, scores)), key=lambda x: x[1], reverse=True)
    
    final_context = []
    detected_years = set()
    detected_versions = set()
    
    # 상위 40개 사용 (방대한 양을 압축하기 위해 top_k를 늘림)
    for doc, score in scored_docs[:40]:
        # 연도 수집
        y = doc.metadata.get('year', 0)
        if y > 2000: detected_years.add(y)
        
        # 버전 수집
        v = doc.metadata.get('version', 'Unknown')
        if v != 'Unknown': detected_versions.add(v)
            
        final_context.append({
            "text": doc.page_content,
            "date": doc.metadata.get('date', 'Unknown'),
            "version": v,
            "year": y
        })
        
    return final_context, detected_years, detected_versions

def generate_l4_report(app_name, context_data, detected_years):
    if not context_data:
        return None

    # 분석 기간 파악 (예: 2023 ~ 2025)
    if detected_years:
        min_year = min(detected_years)
        max_year = max(detected_years)
        period_str = f"{min_year}년 ~ {max_year}년"
    else:
        period_str = "전체 기간"

    context_json = json.dumps(context_data, ensure_ascii=False, indent=2)

    prompt = f"""
당신은 IT 서비스 전문 컨설턴트이자 CIO(Chief Information Officer)입니다.
아래 데이터는 모바일 앱 **[{app_name}]**의 **서비스 런칭 이후 축적된 전체 히스토리 데이터({period_str})**입니다.
이를 바탕으로 **서비스 종합 진단 보고서(L4)**를 작성하십시오.

🛑 [작성 원칙]
1. 단순한 버그 나열이 아닌, **서비스의 '정체성(Identity)', '성장 과정', '시장 내 위상'**을 논하십시오.
2. 분석 기간: **{period_str}**
3. 마크다운 형식을 사용하여 전문적으로 작성하십시오.

[Context Data]
{context_json}

[보고서 양식]
# {app_name} 서비스 종합 진단 보고서

## 1. 🏛️ 서비스 오버뷰 (Executive Summary)
*   **분석 범위**: {period_str}
*   **서비스 정체성**: (이 앱은 사용자들에게 어떤 가치를 제공하는가?)
*   **종합 평점**: (리뷰 분위기를 기반으로 한 정성적 평가, 예: 최상/우수/보통/미흡)

## 2. 📜 서비스 진화의 역사 (History & Evolution)
(시간 흐름에 따라 서비스가 어떻게 변화하고 발전해왔는지 서술)
*   **초기/과거**: ...
*   **최근 동향**: ...

## 3. 💎 핵심 가치 및 강점 (Core Competencies)
(오랜 기간 변치 않고 사랑받은 이 앱만의 강력한 무기)
*   **Strength 1**: ...
*   **Strength 2**: ...

## 4. ⚠️ 고질적 리스크 및 과제 (Chronic Issues)
(단발성 버그가 아닌, 서비스 전체 기간 동안 지속적으로 제기된 근본적 문제)
*   **Risk 1**: ...
*   **Risk 2**: ...

## 5. 🔭 미래 로드맵 및 전략 제언 (Strategic Roadmap)
(축적된 데이터를 기반으로, 향후 3년 이상을 바라보는 장기적 전략)
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
    print(f"📊 대상 앱: {len(target_apps)}개")

    for app in tqdm(target_apps, desc="App Loop"):
        app_name = app['a_name']
        app_id = app['a_idx']
        
        print(f"\n📱 [{app_name}] L4 종합 분석 시작")
        aggressive_gc()
        
        # 1. RAG 검색 (Total)
        context, years, _ = get_total_context(
            app_name, vector_store, reranker
        )
        
        if len(context) < MIN_CHUNKS_FOR_TOTAL:
            # print(f"  ⏭️ 데이터 부족 ({len(context)} chunks)")
            continue
            
        print(f"  🔍 {len(context)}개 청크 확보 (분석 기간: {min(years) if years else '?'} ~ {max(years) if years else '?'})")
        
        # 2. LLM 생성
        report_text = generate_l4_report(app_name, context, years)
        
        if report_text:
            # 3. DB 저장 (L4)
            if save_l4_report(app_id, report_text):
                print(f"    ✅ L4 종합 리포트 저장 완료")
            else:
                print(f"    ❌ 저장 실패")
        else:
            print(f"    ⚠️ 생성 실패")
            
        time.sleep(1)

    print("\n🎉 L4 종합 분석 완료!")

if __name__ == "__main__":
    main()