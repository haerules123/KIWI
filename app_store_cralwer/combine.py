import json
import os
import glob
import pandas as pd

# ==========================================
# [설정] 파일 패턴 및 저장할 파일명
# ==========================================
INPUT_PATTERN = "*_absa_results.json"  # 이 패턴과 일치하는 모든 파일을 찾음
OUTPUT_JSON = "messenger_train_data.json" # 최종 병합된 JSON

def merge_json_files():
    # 1. 파일 리스트 찾기
    files = glob.glob(INPUT_PATTERN)
    
    # 결과 파일이 이미 폴더에 존재하면 리스트에서 제외 (중복 병합 방지)
    if OUTPUT_JSON in files:
        files.remove(OUTPUT_JSON)
        
    print(f"📂 발견된 파일: {len(files)}개")

    if not files:
        print("❌ 병합할 파일이 없습니다.")
        return

    merged_data = []

    # 2. 파일 순회 및 단순 병합
    print(f"\n🔄 병합 시작...")
    
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # 데이터가 리스트 형태인지 확인 후 병합
                if isinstance(data, list):
                    merged_data.extend(data) # 리스트 이어붙이기
                    print(f"   ✅ {file_path}: {len(data)}개 데이터 병합")
                else:
                    print(f"   ⚠️ {file_path}: 형식이 리스트가 아닙니다. 건너뜁니다.")
            
        except Exception as e:
            print(f"   ⚠️ {file_path} 처리 중 오류 발생: {e}")

    # 3. JSON으로 저장
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
    print(f"\n💾 JSON 저장 완료: {OUTPUT_JSON} (총 {len(merged_data)}건)")

if __name__ == "__main__":
    merge_json_files()