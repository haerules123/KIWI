import json
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer, 
    ElectraPreTrainedModel, 
    ElectraModel,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
MODEL_NAME = "beomi/KcELECTRA-base-v2022"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10              # 데이터가 적으므로 에폭을 넉넉히
LEARNING_RATE = 3e-5
DATA_FILE = "train_data.json"
OUTPUT_DIR = "./final2_absa_weighted"

# 라벨 정의
aspect_list = [
    "재생 및 화질", "앱 안정성 및 설치", "콘텐츠 및 기능", 
    "로그인 및 인증", "구독 및 결제", "서비스 및 UI", "의견없음"
]
sentiment_list = ["긍정", "부정", "중립"]

aspect2id = {label: i for i, label in enumerate(aspect_list)}
sentiment2id = {label: i for i, label in enumerate(sentiment_list)}

# ==========================================
# 2. 커스텀 모델 정의 (Multi-Head + Weighted Loss)
# ==========================================

# ElectraPreTrainedModel -> HuggingFace에서 만든 설계도
# 모델을 저장하는 법, 설정값을 불러오는 법 등 기초적인 기능이 들어가있음
class KcElectraForWeightedABSA(ElectraPreTrainedModel):
    def __init__(self, config, aspect_weights=None, sentiment_weights=None):
        super().__init__(config)
        # 뇌장착
        # ElectraModel은 한국어 문장을 읽고 임베딩 해주는 객체
        self.electra = ElectraModel(config)
        
        # Aspect 분류기 (7 classes)
        self.aspect_classifier = nn.Sequential(
            # 10% 확률로 일부 데이터를 무작위로 무시 : 정답을 외우는 걸 방지
            nn.Dropout(0.1),
            # 실제로 분류하는 장치
            nn.Linear(config.hidden_size, len(aspect_list))
        )
        
        # Sentiment 분류기 (3 classes)
        self.sentiment_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, len(sentiment_list))
        )
        
        # Class Weight 저장 (학습 시 Loss 계산에 사용)
        self.aspect_weights = aspect_weights
        self.sentiment_weights = sentiment_weights
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                aspect_labels=None, sentiment_labels=None):
        # 문장을 숫자들의 행렬로 변환
        outputs = self.electra(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        # [CLS] 토큰 벡터 추출
        # 문장 맨 앞에 있는 요약용 숫자 뭉치 하나를 골라냄(CLS)
        # 컴퓨터는 문장을 3차원의 덩어리로 기억
        # : -> 전체 문장을 다 가져와
        # 0 -> 각 문장에서 0 번째 단어만 가져와 (CLS)
        # : -> 그 단어가 가진 숫자 전부 다 가져와
        sequence_output = outputs.last_hidden_state[:, 0, :]

        # 요약용 숫자 뭉치를 aspect와 sentiment 분류기에 전달
        aspect_logits = self.aspect_classifier(sequence_output)
        sentiment_logits = self.sentiment_classifier(sequence_output)

        loss = None
        if aspect_labels is not None and sentiment_labels is not None:
            # 장치(Device) 동기화 및 가중치 적용
            device = input_ids.device
            
            # GPU에서 관리할 수 있도록 가중치가 CPU에 있는 걸 방지
            weight_a = self.aspect_weights.to(device) if self.aspect_weights is not None else None
            weight_s = self.sentiment_weights.to(device) if self.sentiment_weights is not None else None
            
            # Loss Function 정의 (Class Weight 적용)
            # 오답의 확률이 높을 경우 벌점을 주는 형식
            loss_fct_aspect = nn.CrossEntropyLoss(weight=weight_a)
            loss_fct_sentiment = nn.CrossEntropyLoss(weight=weight_s)
            # aspect / sentiment 중 하나가 틀렸을 경우, 그쪽을 좀 더 신경쓰기 위해 Loss 정의
            loss_aspect = loss_fct_aspect(aspect_logits, aspect_labels)
            loss_sentiment = loss_fct_sentiment(sentiment_logits, sentiment_labels)
            
            # 두 Loss의 합 (단순 합산)
            loss = loss_aspect + loss_sentiment

        return {
            'loss': loss,
            'logits': (aspect_logits, sentiment_logits)
        }

# ==========================================
# 3. 데이터셋 클래스
# ==========================================
class ABSADataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        # 전체 리뷰 데이터
        self.data = data
        # 글자를 숫자로 바꿀 번역기
        self.tokenizer = tokenizer
        # 최대 문장 길이
        self.max_len = max_len
    # 전체 양 파악
    def __len__(self):
        return len(self.data)
    # 데이터 꺼내는 함수
    def __getitem__(self, index):
        item = self.data[index]
        text = item['original_segment']
        
        # 텍스트에 없는 라벨이 들어올 경우 방어 코드 (기본값 설정)
        a_label = aspect2id.get(item['aspect'], aspect2id['의견없음'])
        s_label = sentiment2id.get(item['sentiment'], sentiment2id['중립'])
        # 데이터를 숫자로 변경
        encoding = self.tokenizer(
            text,
            # 문장 앞뒤에 CLS(요약), SEP(끝 알림) 같은 특수 기호 붙임
            add_special_tokens=True,
            max_length=self.max_len,
            # 문장이 너무 짧으면 빈칸을 0으로 채워 모든 데이터의 크기를 똑같이 변경
            padding='max_length',
            # 문장이 너무 길면 자름
            truncation=True,
            # 결과를 파이토치 전용 숫자 배열(Tensor)로 만듬
            return_tensors='pt'
        )

        return {
            # 숫자로 바뀐 문장
            'input_ids': encoding['input_ids'].flatten(),
            # 어디가 진짜 글자고, 어디가 0인지 알려주는 지도
            'attention_mask': encoding['attention_mask'].flatten(),
            # aspect 라벨 (숫자)
            'aspect_labels': torch.tensor(a_label, dtype=torch.long),
            # sentiment 라벨 (숫자)
            'sentiment_labels': torch.tensor(s_label, dtype=torch.long)
        }

# Data Collator (Trainer가 배치를 만들 때 라벨을 잘 묶어주도록 함)
# batch 사이즈에 맞게 데이터를 수집하여 전달
# padding의 기능도 있는데 앞에 max_len으로 고정해서 여기선 안함
class ABSADataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        return batch

# ==========================================
# 4. 유틸리티 함수 (가중치 계산, 평가)
# ==========================================
def get_class_weights(data_list):
    """데이터 리스트에서 라벨 빈도를 계산하여 Class Weight 생성"""
    aspects = [aspect2id.get(item['aspect'], aspect2id['의견없음']) for item in data_list]
    sentiments = [sentiment2id.get(item['sentiment'], sentiment2id['중립']) for item in data_list]

    # Aspect 가중치 (Balanced)
    unique_a = np.unique(aspects)
    # 데이터 개수를 세어서 자동으로 가중치를 계산해줌
    a_weights = compute_class_weight(class_weight="balanced", classes=unique_a, y=aspects)
    
    # Sentiment 가중치 (Balanced)
    unique_s = np.unique(sentiments)
    s_weights = compute_class_weight(class_weight="balanced", classes=unique_s, y=sentiments)

    # 전체 클래스 크기에 맞는 배열 생성 (없는 클래스는 가중치 1.0)
    # 데이터에 혹시 한 번도 등장하지 않은 주제가 있다면 기본값 1.0 유지
    final_a_weights = np.ones(len(aspect_list), dtype=np.float32)
    final_s_weights = np.ones(len(sentiment_list), dtype=np.float32)

    for cls, weight in zip(unique_a, a_weights):
        final_a_weights[cls] = weight
    for cls, weight in zip(unique_s, s_weights):
        final_s_weights[cls] = weight
    # 파이토치용으로 변환
    return torch.tensor(final_a_weights), torch.tensor(final_s_weights)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits: (aspect_logits, sentiment_logits)
    # labels: (aspect_labels, sentiment_labels)

    # 모델이 낸 라벨 스코어중 제일 높은 라벨로 찍어주는 역할
    a_preds = np.argmax(logits[0], axis=1)
    s_preds = np.argmax(logits[1], axis=1)

    a_labels = labels[0]
    s_labels = labels[1]

    # 불균형 데이터이므로 F1 Score(Macro/Weighted)가 중요
    metrics = {
        'aspect_acc': accuracy_score(a_labels, a_preds),
        'aspect_f1': f1_score(a_labels, a_preds, average='weighted'),
        'sentiment_acc': accuracy_score(s_labels, s_preds),
        'sentiment_f1': f1_score(s_labels, s_preds, average='weighted'),
    }

    # 모델 선택 기준이 될 통합 점수
    metrics['combined_score'] = (metrics['aspect_f1'] + metrics['sentiment_f1']) / 2
    return metrics

# ==========================================
# 5. 메인 실행 함수
# ==========================================
def main():
    # 1. 데이터 로드
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} 파일이 없습니다.")
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"총 데이터 개수: {len(raw_data)}")

    # 2. 학습/검증 분리 (9:1)
    train_data, val_data = train_test_split(raw_data, test_size=0.1, random_state=42, shuffle=True)

    # 3. Class Weights 계산
    a_weights, s_weights = get_class_weights(train_data)
    print(f"Aspect Class Weights: {a_weights}")
    print(f"Sentiment Class Weights: {s_weights}")

    # 4. 토크나이저 및 데이터셋 준비
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = ABSADataset(train_data, tokenizer, MAX_LEN)
    val_dataset = ABSADataset(val_data, tokenizer, MAX_LEN)

    # 5. 모델 초기화 (가중치 주입)
    model = KcElectraForWeightedABSA.from_pretrained(
        MODEL_NAME,
        aspect_weights=a_weights,
        sentiment_weights=s_weights
    )
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # 6. 트레이너 설정
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        # 말그대로 웜업 초기 10% 동안은 좀 작은 보폭으로 조심스럽게 학습
        warmup_steps=int(len(train_dataset) / BATCH_SIZE * EPOCHS * 0.1), # 10% warmup
        # 암기 방지 장치
        weight_decay=0.01,
        logging_dir='./logs',
        # 중간 보고
        logging_steps=50,
        # epoch 마다 테스트
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # 가장 좋은 모델을 선택
        load_best_model_at_end=True,
        # 가장 좋은 모델을 뽑는 기준
        metric_for_best_model="combined_score", # F1 평균 점수가 높은 모델 저장
        learning_rate=LEARNING_RATE,
        # 저장된파일 중 최신 2개만 남기고 나머지는 지우라는 뜻
        save_total_limit=2,
        # 데이터를 버리지 말고 다 쓰라고 명령
        remove_unused_columns=False, # 커스텀 입력(labels) 유지를 위해 필수
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=ABSADataCollator(tokenizer)
    )

    # 7. 학습 시작
    print("학습을 시작합니다...")
    trainer.train()
    
    # 8. 최종 모델 저장
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 커스텀 모델이므로 state_dict 방식으로 저장 권장
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "pytorch_model.bin"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.config.save_pretrained(OUTPUT_DIR)
    
    print(f"학습 완료! 모델이 저장되었습니다: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()