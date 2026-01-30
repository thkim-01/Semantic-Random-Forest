# Semantic Forest

SMILES 기반 분자 구조를 **Drug Target Ontology (DTO)** 기반 온톨로지로 변환한 뒤,
여러 개의 설명가능한 결정트리를 **배깅(bootstrap aggregating)** 으로 학습해 분류 성능을 높이는 모델입니다.

단일 트리로 학습하는 버전은 별도 레포로 분리했습니다:

- https://github.com/thkim-01/Semantic-Decision-Tree

## 개발 환경

- OS: Windows (테스트 환경)
- Python: 3.9+
- 주요 의존성: `owlready2`, `rdkit`, `scikit-learn`, `pandas`, `numpy`

## 설치

```bash
pip install -r requirements.txt
```

## 실행 방법

### 1) BBBP 배깅 모델 실행 (여러 트리 학습)

```bash
python experiments/verify_semantic_forest.py
```

실행 결과는 콘솔 로그로 출력되며, 요약 파일이 `forest_results.txt`로 저장됩니다.

## 간단 설명

- 입력: `data/` 아래 CSV (예: `data/bbbp/BBBP.csv`)
- 처리: SMILES → 피처 추출(RDKit) → DTO 기반 온톨로지 인스턴스 생성 → 트리 다수 학습(배깅)
- 평가: AUC-ROC, Accuracy
