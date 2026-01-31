# pip install google-play-scraper pandas pymysql cryptography python-dotenv tqdm

import os
import pymysql
from dotenv import load_dotenv
from google_play_scraper import app, reviews_all
from google_play_scraper.exceptions import NotFoundError
from tqdm import tqdm

# 1. .env 파일에서 환경변수 로드
load_dotenv()

def get_db_connection():
    """DB 연결 객체를 생성하는 함수"""
    return pymysql.connect(
        host=os.getenv('host'),
        port=int(os.getenv('port')),
        user=os.getenv('user'),
        password=os.getenv('passwd'),
        database=os.getenv('dbname'),
        charset='utf8mb4',
        autocommit=True # 자동으로 commit 수행
    )

target_app_ids = [
    'com.apple.atve.androidtv.appletv',
    'com.amazon.avod.thirdpartyclient',
    'kr.co.captv.pooqV2',
    'net.cj.cjhv.gs.tving',
    'com.frograms.wplay',
    'com.coupang.mobile.play',
    'com.disney.disneyplus',
    'com.netflix.mediaclient'
]

def save_data_to_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("데이터 수집 및 DB 저장을 시작합니다...")

    for target in target_app_ids:
        try:
            # --- [단계 1] 앱 상세 정보 수집 ---
            app_detail = app(target, lang='ko', country='kr')
            
            # --- [단계 2] APP_GENRE 처리 (장르 저장 및 IDX 가져오기) ---
            genre_name = app_detail['genre']
            cursor.execute("SELECT ag_idx FROM app_genre WHERE ag_name = %s", (genre_name,))
            genre_row = cursor.fetchone()
            
            if genre_row:
                ag_idx = genre_row[0]
            else:
                cursor.execute("INSERT INTO app_genre (ag_name) VALUES (%s)", (genre_name,))
                ag_idx = cursor.lastrowid

            # --- [단계 3] APP 정보 저장 또는 업데이트 ---
            cursor.execute("SELECT a_idx FROM app WHERE a_code = %s", (target,))
            app_row = cursor.fetchone()
            
            if app_row:
                a_idx = app_row[0]
                # 기존 앱 정보 업데이트 (평점, 업데이트일 등 최신화)
                update_app_sql = """
                    UPDATE app SET a_name=%s, a_score=%s, a_rating=%s, a_download_count=%s, 
                    a_last_update=%s, a_developer=%s, a_developer_email=%s, a_developer_link=%s, a_icon=%s, ag_idx=%s
                    WHERE a_idx=%s
                """
                cursor.execute(update_app_sql, (
                    app_detail['title'], app_detail['score'], app_detail['ratings'],
                    app_detail['installs'], app_detail['lastUpdatedOn'], app_detail['developer'],app_detail['developerEmail'],
                    app_detail['developerWebsite'], app_detail['icon'], ag_idx, a_idx
                ))
            else:
                # 새 앱 정보 저장
                insert_app_sql = """
                    INSERT INTO app (a_name, a_code, a_score, a_rating, a_download_count, 
                                     a_last_update, a_developer, a_developer_email, a_developer_link, a_icon, ag_idx)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_app_sql, (
                    app_detail['title'], target, app_detail['score'], app_detail['ratings'],
                    app_detail['installs'], str(app_detail['lastUpdatedOn']), app_detail['developer'],app_detail['developerEmail'],
                    app_detail['developerWebsite'], app_detail['icon'], ag_idx
                ))
                a_idx = cursor.lastrowid

            # --- [단계 4] 리뷰 수집 ---
            reviews = reviews_all(target, sleep_milliseconds=100, lang='ko', country='kr')
            
            # 성능을 위해 해당 앱의 기존 버전들을 미리 가져와서 메모리에 저장 (캐싱)
            cursor.execute("SELECT v_version, v_idx FROM version WHERE a_idx = %s", (a_idx,))
            version_cache = {row[0]: row[1] for row in cursor.fetchall()}

            # --- [단계 5] 리뷰 루프 돌며 저장 ---
            for rev in tqdm(reviews):
                # 1. 버전 체크 및 저장
                raw_version = rev.get('reviewCreatedVersion') or 'unknown' # 버전이 없으면 unknown
                
                if raw_version not in version_cache:
                    # DB에 새 버전 저장
                    cursor.execute("INSERT INTO version (v_version, a_idx) VALUES (%s, %s)", (raw_version, a_idx))
                    v_idx = cursor.lastrowid
                    version_cache[raw_version] = v_idx # 캐시에 추가하여 다음 리뷰 때 재사용
                else:
                    v_idx = version_cache[raw_version] # 이미 있으면 캐시에서 IDX 가져옴

                # 2. 리뷰 데이터 저장 (INSERT IGNORE로 중복 uuid 방지)
                insert_review_sql = """
                    INSERT IGNORE INTO review 
                    (r_uuid, r_content, r_score, r_date, r_like, r_reply, r_reply_date, v_idx)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_review_sql, (
                    rev['reviewId'],
                    rev['content'],
                    rev['score'],
                    rev['at'],
                    rev['thumbsUpCount'],
                    rev['replyContent'],
                    rev['repliedAt'],
                    v_idx # [핵심] 찾거나 생성한 버전 IDX 사용
                ))

            print(f"[{app_detail['title']}] 수집 및 저장 완료 (리뷰 {len(reviews)}건)")

        except NotFoundError:
            print(f"에러: '{target}' 앱을 찾을 수 없습니다.")
        except Exception as e:
            print(f"에러 발생 ({target}): {e}")

    cursor.close()
    conn.close()
    print("모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    save_data_to_db()