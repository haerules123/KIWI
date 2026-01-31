import requests
import pandas as pd
import time
import random

def get_app_store_reviews(app_id, country='kr', page_limit=10):
    """
    애플 앱스토어 RSS 피드를 통해 리뷰를 수집합니다.
    :param app_id: 앱 ID (숫자)
    :param country: 국가 코드 (예: 'kr', 'us')
    :param page_limit: 가져올 페이지 수 (최대 10페이지, 페이지당 50개 = 최대 500개)
    :return: 리뷰 DataFrame
    """
    reviews_list = []
    
    # 애플 차단 방지를 위한 헤더 설정
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }

    print(f"[{app_id}] 리뷰 수집 시작...")

    for page in range(1, page_limit + 1):
        url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status() # 404 등 에러 체크
            
            data = response.json()
            feed = data.get('feed', {})
            entries = feed.get('entry', [])

            # 리뷰가 없거나 형식이 다를 경우 중단
            if not entries:
                break
                
            # entry가 리스트가 아닌 경우(리뷰가 1개일 때) 처리
            if isinstance(entries, dict):
                entries = [entries]

            for entry in entries:
                try:
                    review = {
                        'rating': int(entry.get('im:rating', {}).get('label')),
                        'content': entry.get('content', {}).get('label'),
                    }
                    reviews_list.append(review)
                except Exception as e:
                    continue
            
            print(f"페이지 {page} 수집 완료 (누적 {len(reviews_list)}개)")
            
            # 너무 빠른 요청 방지
            time.sleep(random.uniform(1, 2))
            
        except Exception as e:
            print(f"페이지 {page} 수집 중 에러 발생 (더 이상 리뷰가 없거나 차단됨)")
            break

    return pd.DataFrame(reviews_list)

# --- 실행 부분 ---

# 1. 카카오톡 예시 (ID: 362057947)
app_id = '447188370'
app_name = "snapchat"
df = get_app_store_reviews(app_id, country='kr')

# 2. 결과 확인
if not df.empty:
    print("\n--- 수집 결과 (상위 5개) ---")
    print(df.head())
    
    # 3. 파일 저장
    df.to_csv(f'{app_name}_ios_reviews.csv', index=False, encoding='utf-8-sig')
    print(f"\n저장 완료: {app_name}_ios_reviews.csv")
else:
    print("수집된 리뷰가 없습니다.")