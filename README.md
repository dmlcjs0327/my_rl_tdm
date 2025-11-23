# Temporal Difference Models (TDM)

이 프로젝트는 ICLR 2018 논문 "Temporal Difference Models: Model-Free Deep RL for Model-Based Control"의 재현 구현입니다.

## 논문 정보

**제목**: Temporal Difference Models: Model-Free Deep RL for Model-Based Control  
**저자**: Vitchyr Pong, Shixiang Gu, Murtaza Dalal, Sergey Levine  
**학회**: ICLR 2018  
**arXiv**: [1802.09081](https://arxiv.org/abs/1802.09081)

## 개요

TDM(Temporal Difference Models)은 model-free와 model-based 강화학습의 장점을 결합한 알고리즘입니다:

- **Model-free RL의 장점**: 높은 점근적 성능, model bias 없음
- **Model-based RL의 장점**: 높은 샘플 효율성
- **TDM의 혁신**: Goal-conditioned value function을 통해 두 접근법을 연결

### 핵심 아이디어

TDM은 다음과 같은 goal-conditioned value function을 학습합니다:

```
Q(s, a, sg, τ) = -||f(s, a, sg, τ) - sg||
```

여기서:
- `s`: 현재 상태
- `a`: 행동
- `sg`: 목표 상태
- `τ`: 계획 지평선(planning horizon)
- `f`: 학습된 모델 (상태 예측)

이를 통해:
- τ=0: 1-step 모델 (model-based)
- τ>0: multi-step 예측 (model-free)

## 프로젝트 구조

```
.
├── config.yaml                    # 설정 파일
├── networks.py                    # 신경망 구조 (Actor, Critic)
├── replay_buffer.py               # Replay Buffer 및 Goal Relabeling
├── tdm.py                         # TDM 알고리즘 구현
├── env_wrapper.py                 # 환경 래퍼 (goal 추출 등)
├── mpc_planner.py                 # MPC 기반 정책 추출
├── train.py                       # 기본 훈련 스크립트
├── train_with_curriculum.py       # Curriculum Learning 포함 훈련 스크립트 (권장)
├── evaluate.py                    # 평가/시연 스크립트
├── hyperparameter_grid.py         # 하이퍼파라미터 그리드 정의
├── curriculum_learning.py         # Curriculum Learning 구현
├── grid_search.py                 # 분산적 Grid Search 스크립트
├── test_implementation.py         # 구현 검증 테스트
├── requirements.txt               # 의존성 패키지
├── README.md                      # 이 파일
├── GRID_SEARCH_GUIDE.md          # Grid Search 사용 가이드
├── PAPER_REPRODUCIBILITY_CHECK.md # 논문 재현성 검증 보고서
└── REPRODUCIBILITY_FINAL_REVIEW.md # 재현성 최종 검토
```

## 설치

### 방법 1: 아나콘다 사용 (권장)

```bash
# 1. 아나콘다 환경 생성
conda env create -f environment.yml

# 2. 환경 활성화
conda activate tdm

# 3. 환경 확인
python --version  # Python 3.9 이상 확인
```

### 방법 2: pip 사용

```bash
# Python 3.8 이상 권장
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### MuJoCo 환경 (필수)

로봇 환경(Reacher, Pusher, Ant 등)을 사용하려면 MuJoCo를 설치해야 합니다:

```bash
# conda 환경에서
conda install -c conda-forge mujoco

# 또는 pip로
pip install mujoco
```

## 빠른 시작

### 1. 기본 훈련

```bash
# 설정 파일 수정 (config.yaml)
# 환경 선택: Reacher-v5, Pusher-v5, HalfCheetah-v5, Ant-v5

# Curriculum Learning 포함 훈련 (권장)
python train_with_curriculum.py

# 또는 기본 훈련
python train.py
```

### 2. 하이퍼파라미터 튜닝 (Grid Search)

```bash
# 특정 환경에 대해 최적 하이퍼파라미터 자동 탐색
python grid_search.py --env Reacher-v5 --grid-type reduced

# 최고 성능 하이퍼파라미터로 학습
python train_with_curriculum.py --config logs/grid_search_.../best_hyperparameters.yaml
```

### 3. 모델 평가

```bash
# 학습된 모델 평가
python evaluate.py --model ./logs/Reacher-v5_.../model_final.pt --episodes 50
```

## 주요 기능

### 1. 논문 재현성 ✅
- **TDM 알고리즘**: Goal-conditioned Q-function, Bellman equation
- **Goal Relabeling**: Future state sampling, Horizon relabeling
- **MPC Planner**: Direct 및 Optimization 기반 정책 추출
- **실험 환경**: Reacher, Pusher, HalfCheetah, Ant 모두 지원

### 2. 하이퍼파라미터 튜닝 🔍
- **분산적 Grid Search**: 멀티프로세싱을 통한 병렬 탐색
- **자동 최적화**: 최고 성능 하이퍼파라미터 자동 저장
- **환경별 그리드**: 각 환경에 특화된 하이퍼파라미터 범위

### 3. 학습 안정화 🛡️
- **Curriculum Learning**: 쉬운 목표부터 점진적 학습
- **Warm-up Period**: 초기 탐험 강화
- **Early Stopping**: 성능 개선 없을 시 조기 종료
- **Checkpointing**: 최고 성능 모델 자동 저장
- **Gradient Clipping**: 그래디언트 폭주 방지
- **Learning Rate Decay**: 수렴 후 학습률 감소

### 4. 모니터링 및 분석 📊
- **TensorBoard**: 실시간 학습 모니터링
- **자동 평가**: 주기적 성능 평가
- **결과 저장**: 모든 실험 결과 자동 저장

## 지원하는 환경

| 환경 | 작업 | Goal Space | 특징 |
|------|------|------------|------|
| **Reacher-v5** | 7-DoF 로봇 팔로 목표 위치 도달 | End-effector 위치 (2D) | 직접적인 도달 작업 |
| **Pusher-v5** | 퍽을 목표 위치로 밀기 | Hand + Puck XY | 2단계 작업 |
| **HalfCheetah-v5** | 목표 속도로 달리기 | 속도 | 연속 제어 |
| **Ant-v5** | 목표 위치로 이동 | 위치 또는 위치+속도 | 복잡한 dynamics |

## 사용 방법

### 설정 파일 (config.yaml)

```yaml
# 환경 선택
env:
  name: "Reacher-v5"  # Reacher-v5, Pusher-v5, HalfCheetah-v5, Ant-v5
  max_episode_steps: 50

# TDM 하이퍼파라미터
tdm:
  tau_max: 25  # 최대 계획 지평선
  vectorized_supervision: true  # 벡터화된 supervision 사용
  distance_metric: "L1"  # L1 or L2

# 훈련 설정
training:
  total_timesteps: 1000000
  learning_rate_actor: 0.0001
  learning_rate_critic: 0.001
  batch_size: 128
  updates_per_step: 10
  polyak: 0.999  # Soft target update
  
  # Curriculum Learning
  use_curriculum: true
  curriculum:
    initial_difficulty: 0.1
    final_difficulty: 1.0
    type: "distance"  # distance or complexity
    schedule: "linear"  # linear, exponential, step
  
  # Warm-up Period
  use_warmup: true
  warmup:
    steps: 10000
    initial_noise_std: 0.5
    final_noise_std: 0.2
  
  # Early Stopping
  patience: 10  # null to disable
  
  # Gradient Clipping
  grad_clip: 1.0  # null to disable
```

### Grid Search 사용법

```bash
# 축소 그리드로 빠른 탐색 (권장)
python grid_search.py --env Reacher-v5 --grid-type reduced --workers 4

# 전체 그리드로 완전한 탐색
python grid_search.py --env Reacher-v5 --grid-type full

# 최대 실험 수 제한
python grid_search.py --env Reacher-v5 --grid-type reduced --max-experiments 10
```

자세한 사용법은 [GRID_SEARCH_GUIDE.md](GRID_SEARCH_GUIDE.md)를 참조하세요.

## 하이퍼파라미터 가이드

### 중요한 하이퍼파라미터

1. **tau_max** (계획 지평선)
   - 작을수록: model-based에 가까움, 빠른 학습
   - 클수록: model-free에 가까움, 더 긴 계획
   - 권장: 15-25

2. **updates_per_step** (업데이트 빈도)
   - Goal relabeling 덕분에 높은 값 가능
   - 권장: 5-10

3. **vectorized_supervision**
   - True로 설정하면 성능 향상
   - 권장: True

4. **reward_scale**
   - 환경에 따라 조정 필요
   - 권장: Grid Search로 탐색

### Grid Search를 통한 최적화

논문에 명시된 정확한 하이퍼파라미터 값이 없으므로, Grid Search를 통해 최적값을 탐색할 수 있습니다:

```bash
python grid_search.py --env Reacher-v5 --grid-type reduced
```

## 논문 재현성

이 프로젝트는 논문의 실험 환경과 로직을 모두 사용할 수 있도록 구현되었습니다:

- ✅ **핵심 알고리즘**: TDM Q-function, Goal-conditioned policy, Bellman equation
- ✅ **Goal Relabeling**: Future state sampling, Horizon relabeling
- ✅ **MPC Planner**: Direct 및 Optimization 기반 정책 추출
- ✅ **실험 환경**: 4개 환경 모두 지원
- ✅ **학습 파이프라인**: Replay buffer, Target network, Exploration

자세한 검증 내용은 [PAPER_REPRODUCIBILITY_CHECK.md](PAPER_REPRODUCIBILITY_CHECK.md)를 참조하세요.

## 실험 결과

논문에서 보고된 결과와 유사한 성능을 기대할 수 있습니다:

- **샘플 효율성**: Model-based RL 수준
- **최종 성능**: Model-free RL 수준
- **HER 대비**: 더 빠른 수렴

## 문제 해결

### 1. GPU 메모리 부족
- `batch_size` 줄이기 (128 → 64)
- `updates_per_step` 줄이기 (10 → 5)

### 2. 학습이 느림
- `updates_per_step` 늘리기
- `vectorized_supervision` 활성화
- Grid Search의 `--workers` 수 조정

### 3. 성능이 낮음
- Grid Search로 하이퍼파라미터 탐색
- `tau_max` 조정 (15-25)
- `reward_scale` 조정
- 더 오래 훈련 (`total_timesteps` 증가)

### 4. 환경을 찾을 수 없음
```bash
# MuJoCo 설치 확인
conda install -c conda-forge mujoco
# 또는
pip install mujoco
```

## 참고 문서

- [GRID_SEARCH_GUIDE.md](GRID_SEARCH_GUIDE.md): Grid Search 상세 가이드
- [PAPER_REPRODUCIBILITY_CHECK.md](PAPER_REPRODUCIBILITY_CHECK.md): 논문 재현성 검증
- [REPRODUCIBILITY_FINAL_REVIEW.md](REPRODUCIBILITY_FINAL_REVIEW.md): 재현성 최종 검토

## 참고 문헌

```bibtex
@inproceedings{pong2018temporal,
  title={Temporal Difference Models: Model-Free Deep RL for Model-Based Control},
  author={Pong, Vitchyr and Gu, Shixiang and Dalal, Murtaza and Levine, Sergey},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}
```

## 라이선스

이 구현은 교육 및 연구 목적으로 제공됩니다.

## 기여

버그 리포트나 개선 제안은 이슈로 등록해 주세요.

## 감사의 말

원 논문의 저자들에게 감사드립니다.
