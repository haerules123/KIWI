import mysql.connector
import asyncio
import os
import torch
import re
import gc
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings  # Ollama 대신 사용
from langchain_chroma import Chroma
from konlpy.tag import Okt
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
import sys
import traceback
from langchain_core.documents import Document

load_dotenv()

# [설정]
PERSIST_DIRECTORY = "./chromadb_review_line"
EMBEDDING_MODEL = "BAAI/bge-m3"  # HuggingFace 모델 (안정성)
BATCH_SIZE = 500  # 안전성 우선

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ 하드웨어 가속: {device.upper()}")

DB_CONFIG = {
    'host': os.getenv('host'),
    'user': os.getenv('user'),
    'password': os.getenv('passwd'),
    'database': os.getenv('dbname')
}

class KeywordExtractor:
    def __init__(self):
        self.okt = Okt()
        print("  └ KeyBERT 모델 로드 중...")
        # bge-m3와 동일 모델 사용 (일관성)
        sentence_model = SentenceTransformer('BAAI/bge-m3', device=device)
        self.kw_model = KeyBERT(model=sentence_model)

    def extract(self, text):
        if len(text) < 20:
            nouns = self.okt.nouns(text)
            return list(set(nouns))[:5]
        try:
            keywords = self.kw_model.extract_keywords(
                text, keyphrase_ngram_range=(1, 2), 
                stop_words=None, top_n=5
            )
            return [k[0] for k in keywords]
        except:
            return self.okt.nouns(text)[:5]

def clean_text(text):
    """텍스트 정제 + 검증"""
    if not text or not isinstance(text, str):
        return None
    
    # Null Byte 제거
    text = text.replace('\x00', '')
    
    # 제어 문자 제거
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    # 공백 정규화
    text = ' '.join(text.split())
    
    # 최소 길이 체크
    if len(text.strip()) < 5:
        return None
    
    return text.strip()

def validate_embedding_input(text):
    """임베딩 입력 유효성 검사"""
    if not text:
        return False
    
    # 너무 긴 텍스트 (모델 한계 초과)
    if len(text) > 8000:
        return False
    
    # 특수문자만 있는지 체크
    if not re.search(r'[가-힣a-zA-Z0-9]', text):
        return False
    
    return True

def fetch_unprocessed_batch(batch_size=500):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    
    query = """
    SELECT 
        rl.rl_idx,
        rl.rl_line as text,
        an.a_idx,
        at.at_type as aspect,
        et.et_type as sentiment,
        v.v_version as version,
        a.a_name as app_name,
        ag.ag_name as app_genre,
        r.r_date as date
    FROM review_line rl
    JOIN analysis an ON rl.rl_idx = an.rl_idx
    JOIN aspect_type at ON an.at_idx = at.at_idx
    JOIN emotion_type et ON an.et_idx = et.et_idx
    JOIN review r ON rl.r_idx = r.r_idx
    JOIN version v ON r.v_idx = v.v_idx
    JOIN app a ON v.a_idx = a.a_idx
    JOIN app_genre ag ON a.ag_idx = ag.ag_idx
    WHERE rl.rl_vectorized_at IS NULL
    LIMIT %s;
    """
    
    cursor.execute(query, (batch_size,))
    rows = cursor.fetchall()
    
    cursor.close()
    conn.close()
    return rows

def mark_as_vectorized(rl_ids):
    """성공한 ID만 마킹"""
    if not rl_ids:
        return

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    unique_ids = list(set(rl_ids))
    format_strings = ','.join(['%s'] * len(unique_ids))
    
    query = f"UPDATE review_line SET rl_vectorized_at = %s WHERE rl_idx IN ({format_strings})"
    
    try:
        params = [now] + unique_ids
        cursor.execute(query, params)
        conn.commit()
        print(f"    ✓ DB 업데이트 완료: {len(unique_ids)}개")
    except Exception as e:
        print(f"    ❌ DB 업데이트 실패: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def mark_as_failed(rl_ids):
    """실패한 ID는 특수 값으로 마킹 (재처리 방지)"""
    if not rl_ids:
        return

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    failed_timestamp = '1970-01-01 00:00:00'  # 실패 표시
    unique_ids = list(set(rl_ids))
    format_strings = ','.join(['%s'] * len(unique_ids))
    
    query = f"UPDATE review_line SET rl_vectorized_at = %s WHERE rl_idx IN ({format_strings})"
    
    try:
        params = [failed_timestamp] + unique_ids
        cursor.execute(query, params)
        conn.commit()
        print(f"    ⚠️ 실패 데이터 마킹: {len(unique_ids)}개")
    except Exception as e:
        print(f"    ❌ 실패 마킹 중 오류: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

async def ingest_reviews():
    print("🚀 리뷰 데이터 벡터화 파이프라인 시작...")
    print(f"⚙️  배치 사이즈: {BATCH_SIZE}개")
    
    # HuggingFace Embeddings (Ollama보다 안정적)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 32
        }
    )
    
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name="review_sentences"
    )
    
    extractor = KeywordExtractor()
    total_processed = 0
    total_failed = 0

    while True:
        reviews = fetch_unprocessed_batch(BATCH_SIZE)
        
        if not reviews:
            print(f"\n🎉 완료! 총 처리: {total_processed}개 | 실패: {total_failed}개")
            break
            
        print(f"\n📦 배치 처리: {len(reviews)}개")
        
        documents = []
        successful_ids = []
        failed_ids = []

        for row in tqdm(reviews, desc="  메타데이터 추출"):
            raw_text = row['text']
            rl_idx = row['rl_idx']
            
            # 1. 텍스트 정제
            text = clean_text(raw_text)
            
            if not text or not validate_embedding_input(text):
                failed_ids.append(rl_idx)
                continue

            # 2. 키워드 추출 (실패해도 계속 진행)
            try:
                keywords = extractor.extract(text)
            except:
                keywords = []
            
            # 3. 메타데이터 구성
            metadata = {
                "rl_idx": rl_idx,
                "a_idx": row['a_idx'],
                "app_name": row['app_name'],
                "app_genre": row['app_genre'],
                "version": row['version'],
                "aspect": row['aspect'],
                "sentiment": row['sentiment'],
                "date": str(row['date']),
                "keywords": ", ".join(keywords) if keywords else ""
            }
            
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
            successful_ids.append(rl_idx)

        # 4. 벡터 저장 (배치 → 개별 fallback)
        if documents:
            print(f"  💾 벡터화 중: {len(documents)}개 문서...")
            
            try:
                # 배치 저장 시도
                vector_store.add_documents(documents)
                mark_as_vectorized(successful_ids)
                total_processed += len(documents)
                
            except Exception as e:
                print(f"  ⚠️ 배치 저장 실패: {e}")
                print(f"  🔄 개별 저장 모드로 전환...")
                
                # 개별 저장
                individual_success = []
                individual_fail = []
                
                for idx, doc in enumerate(tqdm(documents, desc="    개별 저장")):
                    try:
                        vector_store.add_documents([doc])
                        individual_success.append(successful_ids[idx])
                    except Exception as inner_e:
                        print(f"      ❌ 실패 (ID: {successful_ids[idx]}): {str(inner_e)[:50]}")
                        individual_fail.append(successful_ids[idx])
                
                # 성공/실패 분리 마킹
                if individual_success:
                    mark_as_vectorized(individual_success)
                    total_processed += len(individual_success)
                
                if individual_fail:
                    mark_as_failed(individual_fail)
                    total_failed += len(individual_fail)
        
        # 5. 불량 데이터 마킹
        if failed_ids:
            mark_as_failed(failed_ids)
            total_failed += len(failed_ids)
        
        # 6. GPU 메모리 정리
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"  📊 누적 | 성공: {total_processed} | 실패: {total_failed}")

# main 함수를 try-except로 감싸기
if __name__ == "__main__":
    try:
        asyncio.run(ingest_reviews())
    except Exception as e:
        print(f"\n❌ 치명적 오류 발생:")
        print(f"   에러 타입: {type(e).__name__}")
        print(f"   메시지: {str(e)}")
        print(f"\n📍 상세 스택 트레이스:")
        traceback.print_exc()
        sys.exit(1)