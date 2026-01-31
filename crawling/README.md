# crawling

## 필요 라이브러리
  - pip install google-play-scraper
  - pip install pandas
  - pip install pymysql
  - pip install cryptography
  - pip install python-dotenv
  - pip install tqdm

## 계정 권한 설정
  - root 계정으로 접속 후, crawler 계정에 모든 권한 부여
    - GRANT ALL PRIVILEGES ON kiwi.* TO 'crawler'@'%';
    - FLUSH PRIVILEGES;

### 통합 SQL (테이블 생성 + 외래키 설정 포함)

MySQL 워크벤치나 터미널, DBeaver 같은 툴에서 아래 명령어를 실행하세요.
```sql
CREATE DATABASE IF NOT EXISTS kiwi
DEFAULT CHARACTER SET utf8mb4
COLLATE utf8mb4_general_ci;

USE kiwi;

-- =====================================================
-- 1️⃣ 완전 독립 테이블 (부모 종속 없음)
-- =====================================================

CREATE TABLE app_genre
(
  ag_idx  INT         NOT NULL AUTO_INCREMENT COMMENT '앱 장르 IDX',
  ag_name VARCHAR(50) NULL                    COMMENT '앱 장르 이름',
  PRIMARY KEY (ag_idx),
  UNIQUE KEY uk_ag_name (ag_name)
) COMMENT '앱 장르';

CREATE TABLE aspect_type
(
  at_idx  INT         NOT NULL AUTO_INCREMENT COMMENT '요소 타입 IDX',
  at_type VARCHAR(50) NULL                    COMMENT '요소 종류',
  PRIMARY KEY (at_idx),
  UNIQUE KEY uk_at_type (at_type)
) COMMENT '요소 타입';

CREATE TABLE emotion_type
(
  et_idx  INT         NOT NULL AUTO_INCREMENT COMMENT '감성 타입 IDX',
  et_type VARCHAR(50) NULL                    COMMENT '감성 종류',
  PRIMARY KEY (et_idx),
  UNIQUE KEY uk_et_type (et_type)
) COMMENT '감성 타입';

CREATE TABLE user
(
  u_idx   INT          NOT NULL AUTO_INCREMENT COMMENT '유저 IDX',
  u_name  VARCHAR(50)  NULL                    COMMENT '유저 이름',
  u_id    VARCHAR(50)  NULL                    COMMENT '유저 아이디',
  u_pw    VARCHAR(255) NULL                    COMMENT '유저 비밀번호',
  u_email VARCHAR(255) NULL                    COMMENT '유저 이메일',
  u_admin BOOLEAN NOT NULL DEFAULT 0           COMMENT '유저 레벨',
  PRIMARY KEY (u_idx),
  UNIQUE KEY uk_u_id (u_id)
) COMMENT '유저';

-- =====================================================
-- 2️⃣ 1차 종속 테이블
-- =====================================================

CREATE TABLE app
(
  a_idx             INT           NOT NULL AUTO_INCREMENT COMMENT '앱 IDX',
  a_name            VARCHAR(255)  NULL                    COMMENT '앱 이름',
  a_code            VARCHAR(255)  NULL                    COMMENT '앱 패키지명',
  a_score           FLOAT         NULL                    COMMENT '앱 총 평점',
  a_rating          INT           NULL                    COMMENT '앱 평점 참여 수',
  a_download_count  VARCHAR(50)   NULL                    COMMENT '앱 다운로드 수',
  a_last_update     VARCHAR(50)   NULL                    COMMENT '앱 최근 업데이트 날짜',
  a_developer       VARCHAR(100)  NULL                    COMMENT '앱 개발자',
  a_developer_email VARCHAR(255) NULL                    COMMENT '앱 개발자 이메일',
  a_developer_link  VARCHAR(500)  NULL                    COMMENT '앱 개발자 사이트 URL',
  a_icon            VARCHAR(500)  NULL                    COMMENT '앱 아이콘 이미지 URL',
  ag_idx            INT           NOT NULL                COMMENT '앱 장르 IDX',
  PRIMARY KEY (a_idx),
  FOREIGN KEY (ag_idx) REFERENCES app_genre (ag_idx),
  UNIQUE KEY uk_a_code (a_code)
) COMMENT '앱';

-- =====================================================
-- 3️⃣ 앱 하위 구조
-- =====================================================

CREATE TABLE user_app_list
(
  ual_idx INT NOT NULL AUTO_INCREMENT COMMENT '유저 앱 리스트 IDX',
  u_idx   INT NOT NULL                COMMENT '유저 IDX',
  a_idx   INT NOT NULL                COMMENT '앱 IDX',
  PRIMARY KEY (ual_idx),
  FOREIGN KEY (u_idx) REFERENCES user (u_idx),
  FOREIGN KEY (a_idx) REFERENCES app (a_idx),
  UNIQUE KEY uk_user_app (u_idx, a_idx)
) COMMENT '유저 앱 리스트';

CREATE TABLE version
(
  v_idx     INT          NOT NULL AUTO_INCREMENT COMMENT '버전 IDX',
  v_version VARCHAR(100) NOT NULL                COMMENT '버전 값',
  a_idx     INT          NOT NULL                COMMENT '앱 IDX',
  PRIMARY KEY (v_idx),
  FOREIGN KEY (a_idx) REFERENCES app (a_idx),
  UNIQUE KEY uk_app_version (a_idx, v_version)
) COMMENT '버전';

-- =====================================================
-- 4️⃣ 리뷰 구조
-- =====================================================

CREATE TABLE review
(
  r_idx        INT          NOT NULL AUTO_INCREMENT COMMENT '리뷰 IDX',
  r_uuid       VARCHAR(255) NULL                    COMMENT '리뷰 고유 ID',
  r_content    TEXT         NULL                    COMMENT '리뷰 원본 내용',
  r_score      FLOAT        NULL                    COMMENT '리뷰 평점',
  r_date       DATETIME     NULL                    COMMENT '리뷰 작성 날짜',
  r_like       INT          NULL                    COMMENT '리뷰 도움됨 수',
  r_reply      TEXT         NULL                    COMMENT '개발자 답글',
  r_reply_date DATETIME     NULL                    COMMENT '답글 작성 날짜',
  v_idx        INT          NOT NULL                COMMENT '버전 IDX',
  PRIMARY KEY (r_idx),
  FOREIGN KEY (v_idx) REFERENCES version (v_idx),
  UNIQUE KEY uk_r_uuid (r_uuid)
) COMMENT '리뷰';

CREATE TABLE review_line
(
  rl_idx  INT  NOT NULL AUTO_INCREMENT 
  COMMENT '리뷰 문장 IDX',
  rl_line TEXT NULL                    
  COMMENT '분리된 리뷰 문장',
  r_idx   INT  NOT NULL                
  COMMENT '리뷰 IDX',
  PRIMARY KEY (rl_idx),
  FOREIGN KEY (r_idx) REFERENCES review (r_idx)
) COMMENT '리뷰 문장';

-- =====================================================
-- 5️⃣ 분석 / 보고서
-- =====================================================

CREATE TABLE analysis
(
  a_idx     INT   NOT NULL AUTO_INCREMENT COMMENT '분석 IDX',
  a_a_score FLOAT NULL                    COMMENT '분석 요소 점수',
  a_e_score FLOAT NULL                    COMMENT '분석 감성 점수',
  rl_idx    INT   NOT NULL                COMMENT '리뷰 문장 IDX',
  at_idx    INT   NOT NULL                COMMENT '요소 타입 IDX',
  et_idx    INT   NOT NULL                COMMENT '감성 타입 IDX',
  PRIMARY KEY (a_idx),
  FOREIGN KEY (rl_idx) REFERENCES review_line (rl_idx),
  FOREIGN KEY (at_idx) REFERENCES aspect_type (at_idx),
  FOREIGN KEY (et_idx) REFERENCES emotion_type (et_idx)
) COMMENT '분석';

CREATE TABLE analytics
(
  an_idx           INT      NOT NULL AUTO_INCREMENT COMMENT '보고서 IDX',
  an_text          LONGTEXT NULL                    COMMENT '보고서 내용',
  an_level         ENUM('L1', 'L2', 'L3', 'L4') NOT NULL DEFAULT 'L1' COMMENT '보고서 레벨',
  an_vectorized_at DATETIME NULL                    COMMENT '보고서 벡터화 완료 시간',
  v_idx            INT      NOT NULL                COMMENT '버전 IDX',
  PRIMARY KEY (an_idx),
  FOREIGN KEY (v_idx) REFERENCES version (v_idx),
  UNIQUE KEY uk_version_level (v_idx, an_level),
  INDEX idx_an_level (an_level)
) COMMENT '보고서';

-- =====================================================
-- 6️⃣ 최종 관계 테이블
-- =====================================================

CREATE TABLE saved
(
  s_idx  INT NOT NULL AUTO_INCREMENT COMMENT '저장된 보고서 IDX',
  u_idx  INT NOT NULL                COMMENT '유저 IDX',
  an_idx INT NOT NULL                COMMENT '보고서 IDX',
  PRIMARY KEY (s_idx),
  UNIQUE KEY uk_user_analytics (u_idx, an_idx),
  FOREIGN KEY (u_idx)
    REFERENCES user (u_idx)
    ON DELETE CASCADE,
  FOREIGN KEY (an_idx)
    REFERENCES analytics (an_idx)
    ON DELETE CASCADE
) COMMENT '저장된 보고서';
```

## aspect_type, emotion_type에 필요한 데이터

```sql
-- 요소 타입 초기화
INSERT INTO aspect_type (at_type) VALUES 
    ('재생 및 화질'),
    ('앱 안정성 및 설치'),
    ('콘텐츠 및 기능'),
    ('로그인 및 인증'),
    ('구독 및 결제'),
    ('서비스 및 UI'),
    ('의견없음');


-- 감성 타입 초기화
  INSERT INTO emotion_type (et_type) VALUES ('긍정'), ('부정'), ('중립');

```

## aspect_type, emotion_type 데이터 삭제 (idx 초기화)
```sql
  -- 1. 외래키 제약 조건 체크를 잠시 끕니다.
SET FOREIGN_KEY_CHECKS = 0;

-- 2. 내용만 비우고 싶은 테이블들을 비웁니다.
TRUNCATE TABLE aspect_type;
TRUNCATE TABLE emotion_type;

-- 만약 분석 결과까지 싹 지우고 싶다면 아래도 같이 수행하세요.
-- TRUNCATE TABLE analysis;
-- TRUNCATE TABLE review_line;

-- 3. 외래키 체크를 다시 켭니다. (반드시 수행!)
SET FOREIGN_KEY_CHECKS = 1;
```

## INDEX 처리 > 속도 향상

```sql
-- analytics > version JOIN
CREATE INDEX idx_analytics_v_idx ON analytics (v_idx);

-- version > app JOIN
CREATE INDEX idx_version_a_idx ON version (a_idx);

-- review JOIN + MIN(r_date)
CREATE INDEX idx_review_v_idx_r_date ON review (v_idx, r_date);

-- 유저 기준 앱 필터링 (중요)
CREATE INDEX idx_ual_user_app ON user_app_list (u_idx, a_idx);
```
