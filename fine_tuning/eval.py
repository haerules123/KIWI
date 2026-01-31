import json
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
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
# 1. 설정 (경로 및 파라미터)
# ==========================================
# 저장된 모델이 있는 폴더 경로
MODEL_DIR = "./final2_absa_weighted"
# 원본 데이터 파일 (검증셋을 다시 나누기 위해 필요)
DATA_FILE = "train_data.json"
MAX_LEN = 128

# 라벨 정의 (학습 때와 순서가 정확히 일치해야 함)
aspect_list = [
    "재생 및 화질", "앱 안정성 및 설치", "콘텐츠 및 기능", 
    "로그인 및 인증", "구독 및 결제", "서비스 및 UI", "의견없음"
]
sentiment_list = ["긍정", "부정", "중립"]

aspect2id = {label: i for i, label in enumerate(aspect_list)}
sentiment2id = {label: i for i, label in enumerate(sentiment_list)}

# ==========================================
# 2. 모델 클래스 정의 
# (저장된 파일에는 가중치만 있고 설계도가 없으므로 다시 정의해야 함)
# ==========================================
class KcElectraForWeightedABSA(ElectraPreTrainedModel):
    def __init__(self, config, aspect_weights=None, sentiment_weights=None):
        super().__init__(config)
        self.electra = ElectraModel(config)
        
        self.aspect_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, len(aspect_list))
        )
        
        self.sentiment_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, len(sentiment_list))
        )
        
        # 평가 시에는 Loss 가중치가 필수는 아니므로 None 처리 가능하게 설정
        self.aspect_weights = aspect_weights
        self.sentiment_weights = sentiment_weights
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                aspect_labels=None, sentiment_labels=None):
        outputs = self.electra(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state[:, 0, :]

        aspect_logits = self.aspect_classifier(sequence_output)
        sentiment_logits = self.sentiment_classifier(sequence_output)

        loss = None
        if aspect_labels is not None and sentiment_labels is not None:
            device = input_ids.device
            
            # 가중치가 없으면 None (기본 Loss 사용)
            weight_a = self.aspect_weights.to(device) if self.aspect_weights is not None else None
            weight_s = self.sentiment_weights.to(device) if self.sentiment_weights is not None else None
            
            loss_fct_aspect = nn.CrossEntropyLoss(weight=weight_a)
            loss_fct_sentiment = nn.CrossEntropyLoss(weight=weight_s)
            
            loss_aspect = loss_fct_aspect(aspect_logits, aspect_labels)
            loss_sentiment = loss_fct_sentiment(sentiment_logits, sentiment_labels)
            
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
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item['original_segment']
        
        a_label = aspect2id.get(item['aspect'], aspect2id['의견없음'])
        s_label = sentiment2id.get(item['sentiment'], sentiment2id['중립'])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'aspect_labels': torch.tensor(a_label, dtype=torch.long),
            'sentiment_labels': torch.tensor(s_label, dtype=torch.long)
        }

class ABSADataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        return batch

# ==========================================
# 4. 평가 지표 계산 함수
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits: (aspect_logits, sentiment_logits)
    # labels: (aspect_labels, sentiment_labels)

    a_preds = np.argmax(logits[0], axis=1)
    s_preds = np.argmax(logits[1], axis=1)

    a_labels = labels[0]
    s_labels = labels[1]

    metrics = {
        'aspect_acc': accuracy_score(a_labels, a_preds),
        'aspect_f1': f1_score(a_labels, a_preds, average='weighted'),
        'sentiment_acc': accuracy_score(s_labels, s_preds),
        'sentiment_f1': f1_score(s_labels, s_preds, average='weighted'),
    }
    return metrics

# ==========================================
# 5. 메인 실행 (평가)
# ==========================================
def evaluate_saved_model():
    print(">>> 평가 준비 중...")

    # 1. 파일 존재 확인
    if not os.path.exists(MODEL_DIR):
        print(f"Error: {MODEL_DIR} 폴더를 찾을 수 없습니다.")
        return
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} 파일을 찾을 수 없습니다.")
        return

    # 2. 토크나이저 로드 (저장된 폴더에서)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        print("토크나이저 로드 완료.")
    except Exception as e:
        print(f"토크나이저 로드 실패: {e}")
        return

    # 3. 데이터 준비
    # 학습 때와 동일한 Validation Set을 만들기 위해 random_state=42 고정
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 9:1 분리 중 뒤쪽 10% (val_data) 사용
    _, val_data = train_test_split(raw_data, test_size=0.1, random_state=42, shuffle=True)
    print(f"검증 데이터 개수: {len(val_data)}")
    
    val_dataset = ABSADataset(val_data, tokenizer, MAX_LEN)

    # 4. 모델 로드 (저장된 가중치 불러오기)
    # 평가만 할 것이므로 class weights는 None으로 두어도 무방 (Loss값만 약간 다를 뿐 Acc/F1은 동일)
    model = KcElectraForWeightedABSA.from_pretrained(MODEL_DIR)
    
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print("모델 로드 완료.")

    # 5. Trainer 설정 (평가 모드)
    eval_args = TrainingArguments(
        output_dir="./temp_eval_output",
        per_device_eval_batch_size=32,
        do_train=False,
        do_eval=True,
        dataloader_num_workers=0,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=ABSADataCollator(tokenizer)
    )

    # 6. 평가 실행
    print(">>> 모델 평가 시작...")
    metrics = trainer.evaluate()

    # 7. 결과 출력
    print("\n" + "="*40)
    print(f"   [ 모델 평가 결과 (폴더: {MODEL_DIR}) ]")
    print("="*40)
    print(f"1. Aspect (주제) 분류")
    print(f"   - 정확도 (Accuracy): {metrics['eval_aspect_acc']:.4f}")
    print(f"   - F1 Score         : {metrics['eval_aspect_f1']:.4f}")
    print("-" * 40)
    print(f"2. Sentiment (감성) 분류")
    print(f"   - 정확도 (Accuracy): {metrics['eval_sentiment_acc']:.4f}")
    print(f"   - F1 Score         : {metrics['eval_sentiment_f1']:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_saved_model()