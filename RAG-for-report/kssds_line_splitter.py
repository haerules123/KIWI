# 필요한 라이브러리 설치 목록
# pip install python=3.10
# pip install python-dotenv
# pip install KSSDS
# pip uninstall transformers -y
# pip install transformers==4.30.2
# pip install pymysql
# pip install soynlp
# pip install cryptography

from KSSDS import KSSDS
import os
import pymysql
from dotenv import load_dotenv
from soynlp.normalizer import repeat_normalize

# KSSDS 문장 분리기 초기화
kssds = KSSDS()
# .env 파일에서 환경 변수 로드
load_dotenv()


def get_db_connection():
    """
    환경 변수를 사용하여 MySQL DB에 연결하고 autocommit을 활성화합니다.
    """
    return pymysql.connect(
        host=os.getenv('host'),  # DB 호스트 주소
        port=int(os.getenv('port')),  # DB 포트 번호
        user=os.getenv('user'),  # DB 사용자명
        password=os.getenv('passwd'),  # DB 비밀번호
        database=os.getenv('dbname'),  # 사용할 데이터베이스명
        charset='utf8mb4',  # 한글 및 이모지 지원을 위한 문자셋
        autocommit=True  # 자동 커밋 활성화
    )

def process_reviews_and_save_lines_to_db():
    """
    DB에서 아직 review_line에 없는 리뷰를 가져와 repeat_normalize로 정규화 후 
    KSSDS로 문장 분리하여 저장합니다.
    개별 리뷰 처리 중 발생하는 오류는 건너뛰고 다음 리뷰를 처리하며, 실패한 r_idx를 기록합니다.
    """
    # DB 연결 생성
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 실패한 리뷰의 r_idx를 기록할 리스트
    log_failed_r_idx = []

    try:
        # (A) 리뷰 문장이 아직 저장되지 않은 리뷰만 가져오기
        # review 테이블에는 있지만 review_line 테이블에는 없는 리뷰 조회
        cursor.execute("""
            SELECT
                r.r_idx, r.r_content
            FROM
                review r
            LEFT JOIN
                review_line rl ON r.r_idx = rl.r_idx
            WHERE
                rl.r_idx IS NULL
        """)
        # LIMIT 1000 -> 한 번에 처리할 리뷰 개수 제한 (메모리 및 처리 시간 고려)
        reviews_to_process = cursor.fetchall()

        # 처리할 리뷰가 없는 경우 종료
        if not reviews_to_process:
            print("처리할 새로운 리뷰가 없습니다.")
            return

        print(f"총 {len(reviews_to_process)}개의 리뷰를 처리합니다.\n")

        # 개별 리뷰를 순회하며 처리
        for r_idx, r_content in reviews_to_process:
            try: 
                # 1. 반복 문자 정규화 (예: "ㅋㅋㅋㅋㅋ" -> "ㅋㅋ", "!!!!!" -> "!!")
                # num_repeats=2 는 반복 문자를 최대 2개까지만 남김
                normalized_content = repeat_normalize(r_content, num_repeats=2)
                
                # 2. 문장 분리 (KSSDS 사용)
                # 정규화된 텍스트를 문장 단위로 분리
                segments = kssds.split_sentences(normalized_content)

                # 문장 분리 결과가 비어있는 경우 건너뛰기
                if not segments:
                    print(f"[WARNING] Review {r_idx}의 문장 분리 결과가 없습니다. 건너뛰습니다.")
                    log_failed_r_idx.append(r_idx)
                    continue

                print(f"--- Review r_idx: {r_idx} ({len(segments)} segments) 처리 시작 ---")

                # 3. DB 삽입 데이터 준비
                # (r_idx, 문장) 형태의 튜플 리스트 생성
                review_line_data = [(r_idx, seg) for seg in segments]

                # 4. review_line 테이블에 문장 배치 삽입
                insert_rl_sql = "INSERT INTO review_line (r_idx, rl_line) VALUES (%s, %s)"
                
                # executemany로 한 번에 여러 문장 삽입 (성능 최적화)
                cursor.executemany(insert_rl_sql, review_line_data)
                print(f"  [SUCCESS] Review {r_idx}의 {len(review_line_data)}개 문장이 review_line에 저장되었습니다.\n")

            except Exception as e:
                # 문장 정규화 오류, 분리 오류 (KSSDS), DB 삽입 오류 (pymysql) 등 
                # 개별 리뷰 처리 중 발생하는 모든 오류 처리
                print(f"[PROCESS ERROR] Review {r_idx} 처리 중 오류 발생: {e}")
                # 실패한 r_idx 기록
                log_failed_r_idx.append(r_idx)
                # 오류가 발생한 리뷰는 건너뛰고 다음 리뷰로 진행
                
    except Exception as e:
        # DB 연결 실패, 전체 리뷰 조회 쿼리 오류 등 전체 프로세스에 치명적인 오류만 여기서 처리
        print(f"[CRITICAL ERROR] DB 연결 또는 전체 리뷰 조회 중 치명적인 오류 발생: {e}")
        
    finally:
        # 연결 종료 및 실패 목록 출력
        print("\n" + "="*70)
        print("처리 완료 요약")
        print("="*70)
        
        # 성공적으로 처리된 리뷰 개수 계산
        total_processed = len(reviews_to_process) if reviews_to_process else 0
        total_failed = len(log_failed_r_idx)
        total_success = total_processed - total_failed
        
        print(f"\n총 처리 시도: {total_processed}개")
        print(f"  ✓ 성공적으로 저장: {total_success}개")
        print(f"  ✗ 처리 실패: {total_failed}개")
        
        # 처리 실패 목록 출력
        if log_failed_r_idx:
            print("\n--- 처리 실패 목록 ---")
            print(f"r_idx 목록: {log_failed_r_idx}")
            print("이 리뷰들은 다음 실행 시 재시도되거나, r_idx를 이용해 수동으로 검토해야 합니다.")
        
        if not log_failed_r_idx:
            print("\n✓ 모든 리뷰가 성공적으로 처리되었습니다!")
        
        print("="*70 + "\n")
        
        # 커서가 존재하면 닫기
        if 'cursor' in locals() and cursor:
            cursor.close()
        # DB 연결이 존재하면 닫기
        if 'conn' in locals() and conn:
            conn.close()

# 스크립트가 직접 실행될 때만 함수 호출
if __name__ == "__main__":
    process_reviews_and_save_lines_to_db()
