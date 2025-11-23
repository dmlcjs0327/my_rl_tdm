# Grid Search 가이드

이 가이드는 TDM 하이퍼파라미터 튜닝을 위한 Grid Search 사용법을 설명합니다.

## 개요

이 프로젝트는 다음 기능을 제공합니다:

1. **하이퍼파라미터 그리드 정의**: 논문 기반 일반적인 하이퍼파라미터 범위
2. **분산적 Grid Search**: 멀티프로세싱을 통한 병렬 실행
3. **Curriculum Learning**: 쉬운 목표부터 점진적으로 어려운 목표로 학습
4. **Warm-up Period**: 초기 학습 단계에서 탐험 강화
5. **Early Stopping & Checkpointing**: 최고 성능 모델 자동 저장

## 파일 구조

```
.
├── hyperparameter_grid.py      # 하이퍼파라미터 그리드 정의
├── curriculum_learning.py      # Curriculum Learning 구현
├── train_with_curriculum.py    # Curriculum Learning 포함 학습 스크립트
├── grid_search.py              # Grid Search 메인 스크립트
└── config.yaml                 # 기본 설정 파일
```

## 빠른 시작

### 1. 기본 Grid Search 실행

```bash
# Reacher-v5 환경에 대해 축소 그리드로 검색
python grid_search.py --env Reacher-v5 --grid-type reduced

# 전체 그리드로 검색 (시간이 오래 걸림)
python grid_search.py --env Reacher-v5 --grid-type full

# 최소 그리드로 빠른 테스트
python grid_search.py --env Reacher-v5 --grid-type minimal
```

### 2. 병렬 작업자 수 지정

```bash
# 4개의 병렬 작업자 사용
python grid_search.py --env Reacher-v5 --grid-type reduced --workers 4
```

### 3. 최대 실험 수 제한

```bash
# 처음 10개 조합만 테스트
python grid_search.py --env Reacher-v5 --grid-type reduced --max-experiments 10
```

## 그리드 타입

### minimal (최소)
- 빠른 테스트용
- 주요 하이퍼파라미터만 변동
- 약 2-4개 조합

### reduced (축소)
- 권장 설정
- 균형잡힌 탐색
- 약 50-200개 조합 (환경에 따라 다름)

### full (전체)
- 완전한 탐색
- 시간이 오래 걸림
- 수백~수천 개 조합

## 하이퍼파라미터 그리드

현재 탐색되는 하이퍼파라미터:

- `learning_rate_actor`: [0.00005, 0.0001, 0.0003]
- `learning_rate_critic`: [0.0005, 0.001, 0.003]
- `tau_max`: [15, 20, 25] (환경별로 다름)
- `batch_size`: [64, 128, 256]
- `updates_per_step`: [5, 10, 20]
- `polyak`: [0.995, 0.999]
- `noise_std`: [0.1, 0.2, 0.3]
- `grad_clip`: [0.5, 1.0, 2.0]
- `reward_scale`: [0.1, 1.0, 10.0]

## 결과 확인

Grid Search가 완료되면 다음 파일들이 생성됩니다:

```
logs/grid_search_Reacher-v5_20240101_120000/
├── best_hyperparameters.yaml      # 최고 성능 하이퍼파라미터 설정
├── grid_search_summary.json       # 전체 결과 요약
├── 0000/                          # 실험 0 디렉토리
│   ├── config.yaml
│   ├── model_best.pt
│   └── ...
├── 0001/                          # 실험 1 디렉토리
│   └── ...
└── ...
```

### 최고 하이퍼파라미터 사용

```bash
# 최고 하이퍼파라미터로 학습
python train_with_curriculum.py --config logs/grid_search_Reacher-v5_.../best_hyperparameters.yaml
```

## Curriculum Learning

`config.yaml`에서 Curriculum Learning 설정:

```yaml
training:
  use_curriculum: true
  curriculum:
    initial_difficulty: 0.1  # 초기 난이도 (0.0 ~ 1.0)
    final_difficulty: 1.0    # 최종 난이도
    type: "distance"         # "distance" or "complexity"
    schedule: "linear"       # "linear", "exponential", "step"
```

### Curriculum Learning 타입

- **distance**: 가까운 목표부터 시작하여 점진적으로 먼 목표로
- **complexity**: 단순한 목표부터 시작하여 점진적으로 복잡한 목표로

### 스케줄 타입

- **linear**: 선형적으로 난이도 증가
- **exponential**: 지수적으로 난이도 증가
- **step**: 단계적으로 난이도 증가

## Warm-up Period

`config.yaml`에서 Warm-up 설정:

```yaml
training:
  use_warmup: true
  warmup:
    steps: 10000              # Warm-up 기간
    initial_noise_std: 0.5    # 초기 노이즈
    final_noise_std: 0.2      # 최종 노이즈
    initial_lr_multiplier: 0.1  # 초기 학습률 배수
    final_lr_multiplier: 1.0    # 최종 학습률 배수
```

Warm-up 기간 동안:
- 탐험 노이즈가 높게 유지됨
- 학습률이 점진적으로 증가함

## Early Stopping

`config.yaml`에서 Early Stopping 설정:

```yaml
training:
  patience: 10  # 10번 평가 동안 개선 없으면 중단
```

Early Stopping이 활성화되면:
- 최고 성능 모델이 자동으로 저장됨
- 성능이 개선되지 않으면 학습이 조기 종료됨
- 최고 모델이 최종 모델로 복사됨

## 환경별 사용 예시

### Reacher-v5

```bash
python grid_search.py --env Reacher-v5 --grid-type reduced --workers 4
```

### Pusher-v5

```bash
python grid_search.py --env Pusher-v5 --grid-type reduced --workers 4
```

### HalfCheetah-v5

```bash
python grid_search.py --env HalfCheetah-v5 --grid-type reduced --workers 4
```

### Ant-v5

```bash
python grid_search.py --env Ant-v5 --grid-type reduced --workers 4
```

## 성능 모니터링

각 실험의 TensorBoard 로그를 확인:

```bash
# 특정 실험의 로그 확인
tensorboard --logdir logs/grid_search_Reacher-v5_.../0000

# 모든 실험 비교
tensorboard --logdir logs/grid_search_Reacher-v5_...
```

## 주의사항

1. **메모리 사용량**: 병렬 실행 시 메모리 사용량이 증가합니다. 작업자 수를 조정하세요.

2. **실행 시간**: 전체 그리드는 수일이 걸릴 수 있습니다. 축소 그리드부터 시작하세요.

3. **디스크 공간**: 각 실험마다 모델과 로그가 저장되므로 충분한 디스크 공간이 필요합니다.

4. **재현성**: 각 실험은 고정된 시드로 실행되지만, 멀티프로세싱 환경에서는 실행 순서가 달라질 수 있습니다.

## 문제 해결

### 메모리 부족

```bash
# 작업자 수 줄이기
python grid_search.py --env Reacher-v5 --grid-type reduced --workers 2
```

### 특정 실험만 재실행

`grid_search.py`를 수정하여 특정 실험 ID 범위만 실행하도록 설정할 수 있습니다.

### 결과 분석

`grid_search_summary.json` 파일을 Python으로 로드하여 분석:

```python
import json

with open('logs/grid_search_.../grid_search_summary.json', 'r') as f:
    summary = json.load(f)

# 최고 실험 확인
print(summary['best_experiment'])

# 모든 결과 분석
for result in summary['all_results']:
    if result['success']:
        print(f"{result['experiment_id']}: {result['best_distance']:.4f}")
```

## 참고

- 논문: "Temporal Difference Models: Model-Free Deep RL for Model-Based Control" (ICLR 2018)
- 기본 하이퍼파라미터는 논문 및 일반적인 강화학습 실험에서 사용되는 값들을 기반으로 설정되었습니다.

