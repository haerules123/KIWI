# 📱 App Store Review ABSA Pipeline

iOS 앱스토어의 리뷰를 수집(Crawling)하고, 문장 단위로 분리(Splitting)한 뒤, Google Gemini API를 활용해 **속성 기반 감성 분석(ABSA)**을 수행하는 파이프라인입니다.

## 📋 프로세스 개요 (Workflow)

1.  **Crawler (`app_store_crawler.py`)**: 앱스토어에서 리뷰 데이터 수집
2.  **Splitter (`spliter.py`)**: 수집된 리뷰를 문장 단위로 전처리
3.  **Labeling (`labeling.py`)**: LLM을 이용한 속성(Aspect) 및 감성(Sentiment) 라벨링

---

## 🚀 사용 방법 (Usage)

### 1. 앱스토어 크롤러 실행 (`app_store_crawler.py`)
원하는 iOS 앱의 리뷰를 최신순으로 최대 500개까지 수집합니다.

1.  [Apple App Store](https://apps.apple.com/kr/iphone/today) 웹사이트에서 분석할 앱을 검색합니다.
2.  URL에서 `App ID`를 확인합니다.
    *   예시: `https://apps.apple.com/kr/app/카카오톡-kakaotalk/id362057947`
    *   **App ID**: `362057947`
3.  `app_store_crawler.py` 파일을 열어 설정을 수정합니다.
    *   **Line 66**: `id_number` 변수에 위에서 확인한 **App ID** 입력
    *   **Line 67**: `app_name` 변수에 저장할 **앱 이름** 입력 (영문 권장)
4.  파일을 실행합니다.
    ```bash
    python app_store_crawler.py
    ```
    *   **결과물**: `{앱이름}_ios_reviews.csv` 생성

<br>

### 2. 문장 분리기 실행 (`spliter.py`)
수집된 리뷰 데이터를 분석하기 좋게 문장 단위로 쪼개는 전처리 과정입니다.

1.  `spliter.py` 파일을 엽니다.
2.  **Line 22**: `file_path` 변수에 크롤러로 생성된 파일명 입력
    *   예: `file_path = 'kakao_ios_reviews.csv'`
3.  파일을 실행합니다.
    ```bash
    python spliter.py
    ```
    *   **결과물**: `{앱이름}_split.csv` 생성

<br>

### 3. ABSA 라벨링 실행 (`labeling.py`)
Gemini API를 사용하여 각 문장의 속성(Aspect)과 감성(Sentiment)을 분석합니다.

1.  **환경 변수 설정**: 프로젝트 루트에 `.env` 파일을 생성하고 API 키를 입력합니다.
    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```
2.  `labeling.py` 파일을 열어 분석 설정을 수정합니다.
    *   **Line 14**: `app_name` 변수에 분석할 앱 이름 입력 (결과 파일명에 사용됨)
    *   **Line 18**: `aspect` 리스트 정의 (분석할 속성 카테고리 설정)
    *   **Line 60**: `system_prompt` 수정 (분석 주제와 앱 특성에 맞는 프롬프트 작성)
3.  파일을 실행합니다.
    ```bash
    python labeling.py
    ```
    *   **결과물**: `{앱이름}_absa_results.json` 생성

---

## 📂 파일 구조 (File Structure)

```text
📦 Project Root
├── 📄 app_store_crawler.py  # 앱스토어 리뷰 크롤러
├── 📄 spliter.py            # 문장 분리 전처리
├── 📄 labeling.py           # Gemini API 연동 및 라벨링
├── 📄 .env                  # API Key 관리
├── 📄 requirements.txt      # (Optional) 의존성 패키지 목록
├── {app}_ios_reviews.csv     # 크롤러 실행 시 저장되는 파일
├── {app}_split.csv           # spliter 실행 시 저장되는 파일
└── {app}_absa_results.json   # labeling 실행 시 저장되는 파일
