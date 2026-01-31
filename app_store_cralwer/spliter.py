# 필요한 라이브러리 설치 목록
# pip install python=3.10
# pip install python-dotenv
# pip install KSSDS
# pip install pymysql
# pip install soynlp

from KSSDS import KSSDS
import os
from dotenv import load_dotenv
from soynlp.normalizer import repeat_normalize
import pandas as pd

# KSSDS 문장 분리기 초기화
kssds = KSSDS()
# .env 파일에서 환경 변수 로드
load_dotenv()


def process_reviews_and_save_lines_to_csv():
    try:
        File_Path = "discord_ios_reviews.csv"
        # 파일명에서 'netflix' 추출 (언더바 기준)
        File_name = File_Path.split("_")[0]
        
        # 데이터 읽기
        df_raw = pd.read_csv(File_Path)
        reviews_to_process = df_raw["content"].dropna() # 결측치 제거
        
        # 수정: 판다스 시리즈 비어있는지 확인
        if reviews_to_process.empty:
            print("처리할 새로운 리뷰가 없습니다.")
            return

        print(f"총 {len(reviews_to_process)}개의 리뷰를 처리합니다.\n")
        
        all_segments = [] # 모든 문장을 담을 리스트

        # 개별 리뷰를 순회하며 처리
        for content in reviews_to_process:
            try: 
                # 1. 반복 문자 정규화
                normalized_content = repeat_normalize(content, num_repeats=2)
                
                # 2. 문장 분리
                segments = kssds.split_sentences(normalized_content)

                if not segments:
                    continue

                # 문장들을 리스트에 추가
                all_segments.extend(segments)
                
            except Exception as e:
                print(f"리뷰 처리 중 오류 발생: {e}")
                continue

        # 3. 전체 데이터 한 번에 저장
        if all_segments:
            result_df = pd.DataFrame(all_segments, columns=['sentence'])
            save_path = f"{File_name}_split.csv"
            result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"--- 처리 완료! 저장 경로: {save_path} ({len(all_segments)}개 문장) ---")
        else:
            print("추출된 문장이 없습니다.")

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {File_Path}")
    except Exception as e:
        print(f"프로세스 중 치명적 오류: {e}")

# 스크립트가 직접 실행될 때만 함수 호출
if __name__ == "__main__":
    process_reviews_and_save_lines_to_csv()