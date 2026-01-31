import pymysql
import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, ElectraPreTrainedModel, ElectraModel, AutoConfig
from tqdm import tqdm

# DB 로드
load_dotenv()

def get_db_connection():
    """DB 연결 객체를 생성하는 함수"""
    return pymysql.connect(
        host=os.getenv('host'),
        port=int(os.getenv('port')),
        user=os.getenv('user'),
        password=os.getenv('passwd'),
        database=os.getenv('dbname'),
        charset='utf8mb4',
        autocommit=True 
    )

# ==========================================
# 1. 모델 클래스 정의
# ==========================================
class KcElectraForWeightedABSA(ElectraPreTrainedModel):
    def __init__(self, config, aspect_weights=None, sentiment_weights=None):
        super().__init__(config)
        self.electra = ElectraModel(config)
        
        self.aspect_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 7) 
        )
        self.sentiment_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 3) 
        )
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state[:, 0, :]
        
        aspect_logits = self.aspect_classifier(sequence_output)
        sentiment_logits = self.sentiment_classifier(sequence_output)
        
        return aspect_logits, sentiment_logits

# ==========================================
# 2. 설정 및 로드
# ==========================================
MODEL_PATH = "./final2_absa_weighted" 

aspect_list = [
    "재생 및 화질", "앱 안정성 및 설치", "콘텐츠 및 기능", 
    "로그인 및 인증", "구독 및 결제", "서비스 및 UI", "의견없음"
]
sentiment_list = ["긍정", "부정", "중립"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 토크나이저 및 Config 로드
try:
    # 주의: config.json에 "model_type": "electra" 가 없으면 여기서 또 에러가 날 수 있습니다.
    # 에러가 나면 config.json 파일을 열어 "model_type": "electra"를 추가해주세요.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = AutoConfig.from_pretrained(MODEL_PATH)
except OSError:
    print(f"오류: {MODEL_PATH} 경로에 모델 파일이 없습니다. train.py를 먼저 실행하세요.")
    exit()

# 모델 로드
model = KcElectraForWeightedABSA(config)

# [수정된 부분] strict=False 추가
model.load_state_dict(torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location=device), strict=False)

model.to(device)
model.eval()

# ==========================================
# 3. 예측 함수
# ==========================================
def predict_review(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=128, 
        padding="max_length", 
        truncation=True
    )
    # CPU에 있는 input_ids, attention_mask를 GPU로 이동
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    # 딥러닝 특성상 계산할 때마다 오답 노트를 작성하려고 메모리 저장하는데, 라벨링하는 거니까 제외하고자 작성
    with torch.no_grad():
        a_logits, s_logits = model(input_ids, attention_mask=attention_mask)
    
    # 모델이 내놓은 답을 사람이 추측할 수 있는 확률로 계산
    # Aspect 예측
    a_probs = F.softmax(a_logits, dim=1)
    # 확률이 제일 높은 aspect 인덱스 추출
    a_idx = torch.argmax(a_probs, dim=1).item()
    # 몇 % 확신하는지 추출
    a_conf = a_probs[0][a_idx].item()
    
    # Sentiment 예측
    s_probs = F.softmax(s_logits, dim=1)
    s_idx = torch.argmax(s_probs, dim=1).item()
    s_conf = s_probs[0][s_idx].item()
    
    return {
        'a_idx' : a_idx + 1,
        's_idx' : s_idx + 1,
        'aspect': aspect_list[a_idx],
        'aspect_conf': a_conf,
        'sentiment': sentiment_list[s_idx],
        'sentiment_conf': s_conf
    }

# ==========================================
# 4. 테스트 실행
# ==========================================
if __name__ == "__main__":
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # [수정 1] rl_line뿐만 아니라 rl_idx(리뷰 번호)도 같이 가져옵니다.
    cursor.execute("SELECT rl_idx, rl_line FROM review_line")
    rows = cursor.fetchall() 
    
    for row in tqdm(rows):
        rl_idx = row[0]
        text = row[1]
        
        if not text:
            continue

        # [추가된 부분] 해당 rl_idx가 이미 analysis 테이블에 있는지 확인
        check_sql = "SELECT 1 FROM analysis WHERE rl_idx = %s"
        cursor.execute(check_sql, (rl_idx,))
        
        # 이미 데이터가 있으면 건너뜀
        if cursor.fetchone():
            print(f"스킵: {rl_idx}번 데이터는 이미 존재합니다.")
            continue 
            
        # ---------------------------------------------------------
        # 아래는 기존 예측 및 저장 로직 (데이터가 없을 때만 실행됨)
        # ---------------------------------------------------------
        res = predict_review(text)
        
        sql = """
            INSERT INTO analysis (rl_idx, at_idx, a_a_score, et_idx, a_e_score) 
            VALUES (%s, %s, %s, %s, %s) 
        """
        
        datas = (
            rl_idx, 
            res['a_idx'], 
            round(res['aspect_conf'], 2), 
            res['s_idx'], 
            round(res['sentiment_conf'], 2)
        )
        
        cursor.execute(sql, datas)

    
    print("-" * 100)
    
    cursor.close()
    conn.close()