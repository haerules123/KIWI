import pandas as pd
from dotenv import load_dotenv
import os
import time
import json
import re  # 정규식 모듈 추가
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
app_name = '라인' # 분석할 앱 이름에 맞게 변경하세요 (예: discord)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class SegmentAnalysis(BaseModel):
    aspect: str = Field(description="속성 (메시지 및 통화 품질, 앱 안정성 및 설치, 로그인 및 인증, 계정 및 보안, 기능 및 커뮤니티, 구독 및 비즈니스, UI/UX 및 고객지원, 의견없음)")
    sentiment: str = Field(description="감성 (긍정, 부정, 중립)")

def run_analysis_and_save():
    # 1. 모델 설정 - Gemini 사용
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        convert_system_message_to_human=True
    )
    parser = JsonOutputParser(pydantic_object=SegmentAnalysis)

    # 2. 파일 로드
    try:
        df = pd.read_csv(f"{app_name}_split.csv")
        segments = df["sentence"].dropna().tolist()
        print(f"총 {len(segments)}개의 문장을 불러왔습니다.")
    except Exception as e:
        print(f"파일 로드 실패: {e}")
        return

    # 3. 각 문장 분석
    analyzed_count = 0
    
    for seg in tqdm(segments, desc="분석 중"):
        clean_seg = str(seg).strip()
        
        # [수정됨] 기본 길이 체크
        if len(clean_seg) < 2: 
            continue

        # [수정됨] 한글 필터링 정규식 적용
        # ㄱ-ㅎ (자음), ㅏ-ㅣ (모음), 가-힣 (완성형 한글) 중 하나라도 없으면 건너뜀
        # 영어로만 된 리뷰("Good app")나 숫자/이모티콘만 있는 리뷰 제외
        if not re.search('[ㄱ-ㅎㅏ-ㅣ가-힣]', clean_seg):
            continue

        try:
            format_instructions = parser.get_format_instructions()
            
            # 메신저 앱용으로 수정한 시스템 프롬프트 적용
            system_prompt = f"""너는 메신저 앱 리뷰 분석 전문가야. 
주어진 문장을 분석하여 다음 형식으로만 응답해:

속성 카테고리:
- 메시지 및 통화 품질 : 메신저의 본질적인 기능인 텍스트 전송, 음성/영상 통화, 화면 공유 품질 관련
- 앱 안정성 및 설치 : 앱 오류, 튕김(크래시), 로딩 지연, 설치/업데이트 문제
- 로그인 및 인증: 로그인, 인증번호(SMS) 수신
- 계정 및 보안 : 해킹/도용, 계정 비활성화(밴), 회원탈퇴, 사생활 보호
- 기능 및 커뮤니티 : 친구 관리, 이모티콘, 검색, 백업 부가 기능
- 구독 및 비즈니스 : 유료 구독, 인앱 결제, 환불, 광고 관련
- UI/UX 및 고객지원 : 디자인, 버튼 위치, 다크모드, 고객센터 응대
- 의견없음: 분류가 불가능하거나 단순 비방/칭찬인 경우

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
            analyzed_count += 1
            
            time.sleep(0.5) 

        except KeyboardInterrupt:
            print("\n[알림] 사용자에 의해 중단되었습니다. 현재까지의 결과는 저장되었습니다.")
            break
            
        except Exception as e:
            # print(f"\n[오류 발생] 문장: {clean_seg[:30]}... | 에러: {e}")
            continue
    
    print(f"\n총 {analyzed_count}개의 한글 문장을 분석하고 저장했습니다.")

def save_intermediate_result(file_path, new_data):
    """분석 결과를 하나씩 파일에 이어붙여 저장하는 함수"""
    existing_data = []
    
    # 파일이 존재하고 내용이 있으면 로드
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