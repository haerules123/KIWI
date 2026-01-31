import pandas as pd
from dotenv import load_dotenv
import os
import time
import json
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
app_name = 'disney'
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class SegmentAnalysis(BaseModel):
    aspect: str = Field(description="속성 (재생 및 화질, 앱 안정성 및 설치, 콘텐츠 및 기능, 로그인 및 인증, 구독 및 결제, 서비스 및 UI, 의견없음)")
    sentiment: str = Field(description="감성 (긍정, 부정, 중립)")

def run_analysis_and_save():
    # 1. 모델 설정 - Gemini 사용
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        convert_system_message_to_human=True  # 시스템 메시지를 human 메시지로 변환
    )
    parser = JsonOutputParser(pydantic_object=SegmentAnalysis)

    # 2. 파일 로드
    try:
        df = pd.read_csv(f"{app_name}_split.csv")
        segments = df["sentence"].dropna().tolist()
        print(f"총 {len(segments)}개의 문장을 분석합니다.")
    except Exception as e:
        print(f"파일 로드 실패: {e}")
        return

    # 3. 각 문장 분석
    for seg in tqdm(segments, desc="분석 중"):
        clean_seg = str(seg).strip()
        if len(clean_seg) < 2: 
            continue

        try:
            format_instructions = parser.get_format_instructions()
            
            # 시스템 프롬프트를 더 명확하게 작성
            system_prompt = f"""너는 앱 리뷰 분석 전문가야. 
주어진 문장을 분석하여 다음 형식으로만 응답해:

속성 카테고리:
- 재생 및 화질: 영상 재생, 화질, 버퍼링, 싱크 관련
- 앱 안정성 및 설치: 앱 오류, 크래시, 설치/업데이트 문제
- 콘텐츠 및 기능: 콘텐츠 종류, 검색, 자막, 배속 등 기능
- 로그인 및 인증: 로그인, 인증, 계정 관련
- 구독 및 결제: 요금제, 결제, 환불 관련
- 서비스 및 UI: UI/UX, 고객센터, 일반적 서비스 품질
- 의견없음: 분류 불가능한 경우

감성: 긍정,부정,중립

{format_instructions}"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"다음 문장을 분석해줘: '{clean_seg}'")
            ]

            # 모델 호출
            response = llm.invoke(messages)
            
            # 파싱
            parsed_data = parser.parse(response.content)
            
            # 결과 저장
            result_entry = {
                "original_segment": clean_seg,
                "aspect": parsed_data.get("aspect", "의견없음"),
                "sentiment": parsed_data.get("sentiment", "중립")
            }
            save_intermediate_result(f"{app_name}_absa_results.json", result_entry)
            
            time.sleep(0.5)  # Gemini API 안정성 확보 (무료 tier는 RPM 제한 있음)

        except KeyboardInterrupt:
            print("\n[알림] 사용자에 의해 중단되었습니다. 현재까지의 결과는 저장되었습니다.")
            break
            
        except Exception as e:
            print(f"\n[오류 발생] 문장: {clean_seg[:30]}... | 에러: {e}")
            # 오류 발생해도 계속 진행
            continue

def save_intermediate_result(file_path, new_data):
    """분석 결과를 하나씩 파일에 이어붙여 저장하는 함수"""
    existing_data = []
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    
    existing_data.append(new_data)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    print(f"앱 이름: {app_name}")
    print(f"API 키 로드 여부: {'성공' if GOOGLE_API_KEY else '실패'}")
    
    run_analysis_and_save()
    
    print(f"\n✅ 분석 프로세스가 종료되었습니다.")
    print(f"결과 파일: {app_name}_absa_results.json")