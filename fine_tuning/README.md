# 📱 앱 리뷰 감성 & 속성 분석 (App Review ABSA)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)
![Model](https://img.shields.io/badge/Model-KcELECTRA-green)

이 프로젝트는 **Google Play Store/App Store**의 앱 리뷰 데이터를 분석하여, 리뷰가 어떤 부분(**Aspect**)에 대한 이야기인지, 그리고 그 감정(**Sentiment**)이 무엇인지 동시에 파악하는 **ABSA(Aspect-Based Sentiment Analysis)** 모델입니다.

한국어 구어체와 댓글에 특화된 `beomi/KcELECTRA-base-v2022` 모델을 기반으로 하며, 데이터 불균형 문제를 해결하기 위해 **Weighted Loss** 방식을 적용했습니다.

---

## 🛠️ 설치 (Installation)

모델 학습 및 추론을 위해 필요한 라이브러리를 설치합니다.

```bash
# 기본 필수 라이브러리
pip install -U accelerate transformers scikit-learn

# Accelerate 버전 호환성
pip install 'accelerate>=0.26.0'

# PyTorch (CUDA 12.1 버전 기준)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 📂 데이터셋 (Dataset)

학습 데이터는 **애플 앱스토어 리뷰** 등을 수집하여 구성하였으며, `Gemini API`를 활용하여 1차 라벨링 후 검수 과정을 거쳤습니다.

### 📊 데이터 분포 (Total Statistics)
데이터의 불균형(Class Imbalance)이 존재하여, 학습 시 **Class Weighting** 기법을 적용했습니다.

#### 1. Aspect (속성) - 총 7개 클래스
| Aspect | 데이터 개수 (건) | 비고 |
|:---:|---:|:---|
| **콘텐츠 및 기능** | `1,498` | 🔥 가장 많은 비중 |
| **서비스 및 UI** | `1,134` | |
| **의견없음** | `774` | 단순 비방 등 |
| **구독 및 결제** | `753` | |
| **앱 안정성 및 설치** | `750` | 팅김, 발열 등 |
| **재생 및 화질** | `544` | 스트리밍 관련 |
| **로그인 및 인증** | `284` | |

#### 2. Sentiment (감정) - 총 3개 클래스
| Sentiment | 데이터 개수 (건) | 비율 |
|:---:|---:|:---|
| **부정 (Negative)** | `4,536` | 🔴 압도적 다수 |
| **긍정 (Positive)** | `628` | 🟢 |
| **중립 (Neutral)** | `575` | ⚪ |

---

## 🧠 모델 아키텍처 (Model Architecture)

### Base Model: `beomi/KcELECTRA-base-v2022`
*   **특징**: 한국어 뉴스 댓글 3.5억 건 이상을 학습한 모델.
*   **선정 이유**: 앱 리뷰 특유의 **구어체, 신조어, 오타, 이모티콘** 처리에 일반 BERT 모델보다 월등한 성능을 보임.

### Custom Head & Optimization
*   **Multi-Task Learning**: 하나의 모델이 `Aspect`와 `Sentiment`를 동시에 분류하도록 설계 (2개의 Output Head).
*   **Weighted Loss**: 부정 리뷰가 압도적으로 많은 데이터 불균형 문제를 해결하기 위해, 소수 클래스(긍정, 중립, 로그인 등)에 더 높은 가중치를 부여하여 학습.

---

## 📁 파일 구조 (File Structure)

| 파일명 | 설명 |
|:---|:---|
| `labeling.py` | **데이터 전처리 도구**<br>Google Gemini API를 활용하여 수집된 리뷰 데이터에 자동으로 라벨링(Aspect/Sentiment)을 수행합니다. |
| `train_data.json` | **학습 데이터셋**<br>전처리 및 라벨링이 완료된 JSON 포맷의 데이터 파일입니다. |
| `model2.py` | **모델 학습 (Training)**<br>`KcELECTRA` 모델을 로드하여 파인튜닝을 수행합니다.<br>학습이 완료되면 `./final2_absa_weighted` 폴더에 모델이 저장됩니다. |
| `predict2.py` | **추론 및 테스트 (Inference)**<br>학습된 모델을 불러와 실제 새로운 리뷰 데이터를 넣고 분석 결과를 출력합니다. |

---

## 🚀 사용 방법 (Usage)

### 1. 모델 학습
```bash
python model2.py
```
*   학습이 완료되면 `final2_absa_weighted/` 폴더가 생성됩니다.

### 2. 모델 추론 (테스트)
```bash
python predict2.py
```
*   DB 또는 테스트 문장을 입력하여 분석 결과를 확인합니다.

---