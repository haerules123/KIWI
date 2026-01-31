# 🚀 Advanced App Review Analysis RAG System

본 프로젝트는 앱 리뷰 분석 보고서를 기반으로 사용자 질문에 대해 고도로 정제된 답변을 제공하는 **고급 RAG(Retrieval-Augmented Generation)** 시스템입니다. 단순 검색을 넘어 메타데이터 앙상블 추출과 MoA(Mixture of Agents) 구조를 도입하여 답변의 정확도와 전문성을 극대화했습니다.
본 프로젝트는 python 3.11 버전에서 최적화가 되어있습니다.

## ✨ Key Features

1.  **Metadata Ensemble Ingestion**:
    *   로컬 NLP(`KeyBERT`, `KoNLPy`)와 LLM(`Groq`)을 결합하여 키워드, 감성, 주요 기능을 자동으로 추출하고 벡터 DB에 태깅합니다.
2.  **Hybrid Search Pipeline**:
    *   **BM25**(키워드 기반)와 **Vector Search**(의미론적 기반)를 결합한 Ensemble Retriever를 사용합니다.
3.  **Cross-Encoder Reranking**:
    *   `bge-reranker-v2-m3-ko` 모델을 사용하여 검색된 문서의 우선순위를 재정렬함으로써 컨텍스트 품질을 높였습니다.
4.  **MoA (Mixture of Agents) Generation**:
    *   `Llama-3.1-8B`, `Qwen-2.5-32B` 등 여러 모델이 초안을 작성하고, `Llama-3.3-70B`가 최종 판결 및 종합을 수행하는 전문가 협업 구조를 가집니다.
5.  **Local & Cloud Hybrid**:
    *   임베딩은 로컬(`Ollama/bge-m3`)에서, 추론은 초고속 클라우드 API(`Groq`)에서 수행하여 효율성을 최적화했습니다.

## 🛠 Tech Stack

*   **Framework**: LangChain
*   **LLM API**: Groq (Llama-3.3-70B, Llama-3.1-8B, Qwen)
*   **Local Embedding**: Ollama (bge-m3)
*   **Vector DB**: ChromaDB
*   **NLP Tools**: KeyBERT, KoNLPy (Okt)
*   **Reranker**: HuggingFace Cross-Encoder (`dragonkue/bge-reranker-v2-m3-ko`)
*   **Database**: MySQL (MariaDB)

## 📂 Project Structure

*   `ingest_db.py`: MySQL에서 마크다운 보고서를 읽어와 메타데이터 추출 후 벡터 DB에 저장합니다.
*   `main.py`: 대화형 챗봇 인터페이스 및 검색/생성 파이프라인을 실행합니다.
*   `test.py`: 가상 데이터를 이용한 전체 프로세스(Ingestion to Generation) 통합 테스트 스크립트입니다.
*   `requirements.txt`: 프로젝트 실행을 위한 의존성 패키지 목록입니다.

## ⚙️ Setup & Installation

### 1. 환경 변수 설정
`.env` 파일을 생성하고 필요한 API 키와 DB 정보를 입력합니다.
```env
GROQ_API_KEY=your_groq_api_key
host=your_db_host
user=your_db_user
passwd=your_db_password
dbname=your_db_name
```

### 2. 로컬 모델 준비 (Ollama)
임베딩 모델을 미리 다운로드해야 합니다.
```bash
ollama pull bge-m3
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```
*참고: KoNLPy 사용을 위해 Java(JDK) 설치가 필요할 수 있습니다.*

## 🚀 Usage

### 1. 데이터 베이스 적재 (Data Ingestion)
MySQL에 저장된 보고서를 벡터 DB로 변환합니다.
```bash
python ingest_db.py
```

### 2. 챗봇 실행 (Inference)
질의응답 시스템을 실행합니다.
```bash
python main.py
```

## 🧠 Workflow Details

1.  **Markdown Splitting**: 마크다운 헤더(`#`, `##`, `###`)를 인식하여 문서의 구조적 계층을 유지하며 분할합니다.
2.  **Metadata Enrichment**: 
    *   KeyBERT로 핵심 단어를 뽑고, LLM으로 요약 및 감성 분석을 수행하여 검색 성능을 보조합니다.
3.  **Retriever Ensemble**: 
    *   `BM25Retriever` (40%) + `Chroma Vector Retriever` (60%) 가중치 조합.
4.  **Reranking**: 검색된 20개의 후보 중 가장 관련성이 높은 5개를 Cross-Encoder로 선별합니다.
5.  **MoA Synthesis**:
    *   **Drafters**: 여러 하위 모델이 독립적으로 답변 생성.
    *   **Judge**: 생성된 답변들의 모순을 제거하고 팩트 체크를 거쳐 최종 답변 산출.

---

### 💡 Note
이 시스템은 고성능 한국어 분석을 위해 `bge-m3` 시리즈 모델에 최적화되어 있습니다. Groq 모델 ID는 사용 가능한 최신 버전으로 업데이트하여 사용하세요.
