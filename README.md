# Temporal Difference Models (TDM)

ICLR 2018 논문 "Temporal Difference Models: Model-Free Deep RL for Model-Based Control"의 재현 구현입니다.

**논문**: [1802.09081](https://arxiv.org/abs/1802.09081)

## 설치

### Conda 환경 설정 (권장)

```bash
# 1. Conda 환경 생성 및 활성화
conda env create -f environment.yml
conda activate tdm

# 2. PyTorch GPU 버전 설치 (GPU 사용 시)
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 또는 CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 또는 pip 사용 (conda 설치 실패 시)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. MuJoCo는 이미 environment.yml에 포함되어 있습니다

# 4. 환경 확인
python --version  # Python 3.9 이상 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**참고**: Conda 설치 중 다운로드 오류가 발생하면:
```bash
# 캐시 정리 후 재시도
conda clean --all -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 또는 pip 사용 (더 안정적일 수 있음)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 사용법

### 1. 학습 (PBT - 기본 방법)

Population-based Training (PBT)로 학습합니다.

**실시간 모니터링:**
- 학습 도중 각 개체의 진행 상황이 자동으로 출력됩니다 (기본 5초마다)
- TensorBoard로도 실시간 모니터링 가능:
  ```bash
  tensorboard --logdir ./logs/pbt_Reacher-v5_...
  ``` 

**PBT 동작 방식:**
- 초기: 여러 개의 서로 다른 랜덤 하이퍼파라미터 조합으로 시작
- 병렬 학습: 모든 개체를 동시에 병렬로 학습 (빠른 결과 확인)
- 학습 중: 주기적으로 성능이 좋은 개체의 하이퍼파라미터를 나쁜 개체에 복사 (exploit)
- 탐색: 복사된 하이퍼파라미터에 약간의 변형을 가함 (explore)
- 결과: 학습 곡선을 보고 자동으로 최적 하이퍼파라미터를 찾음

**자동 Population Size 결정:**
- `population_size: auto`로 설정하면 시스템 자원(CPU, GPU, 메모리)을 자동으로 확인하여 최적의 개체 수를 결정합니다
- GPU가 있으면: GPU 개수 기반
- GPU가 없으면: CPU 코어 수 기반 (75% 사용, 최소 1개는 남김)
- 메모리도 고려하여 안전하게 설정됩니다

```bash
# Reacher-v5
python train_pbt.py --config config_reacher.yaml

# Pusher-v5
python train_pbt.py --config config_pusher.yaml

# HalfCheetah-v5
python train_pbt.py --config config_halfcheetah.yaml

# Ant-v5
python train_pbt.py --config config_ant.yaml

# 기본 설정 사용 (Reacher-v5)
python train_pbt.py
```

### 2. 결과 시각화

```bash
# 학습 결과 디렉토리에서 시각화
python visualize_pbt_results.py --log-dir ./logs/pbt_Reacher-v5_20240101_120000
```

### 3. 모델 평가

```bash
python evaluate.py --model ./logs/pbt_Reacher-v5_.../model_final.pt --episodes 50
```


## 프로젝트 구조

```
.
├── config.yaml                    # 기본 설정
├── config_*.yaml                  # 환경별 설정 파일
├── train_pbt.py                   # PBT 학습 스크립트 (기본)
├── train_with_curriculum.py       # PBT에서 사용하는 유틸리티 함수들
├── evaluate.py                    # 모델 평가
├── visualize_pbt_results.py       # 결과 시각화
├── tdm.py                         # TDM 알고리즘
├── networks.py                    # 신경망 구조
├── replay_buffer.py               # Replay Buffer
├── env_wrapper.py                 # 환경 래퍼
├── mpc_planner.py                 # MPC Planner
├── pbt.py                         # PBT 구현
├── curriculum_learning.py         # Curriculum Learning
├── policy_collapse_detector.py    # 정책 붕괴 감지
└── hyperparameter_grid.py         # 하이퍼파라미터 설정 유틸리티
```

## 주요 기능

- **PBT (Population-based Training)**: 학습 곡선 기반 자동 하이퍼파라미터 조정
- **Curriculum Learning**: 쉬운 목표부터 점진적 학습
- **정책 붕괴 감지**: 학습 중 성능 급락 자동 감지 및 복구
- **TensorBoard**: 실시간 학습 모니터링

## 지원 환경

| 환경 | 설명 |
|------|------|
| Reacher-v5 | 7-DoF 로봇 팔로 목표 위치 도달 |
| Pusher-v5 | 퍽을 목표 위치로 밀기 |
| HalfCheetah-v5 | 목표 속도로 달리기 |
| Ant-v5 | 목표 위치로 이동 |

## 설정 파일

각 환경별로 최적화된 설정 파일이 제공됩니다:
- `config_reacher.yaml`
- `config_pusher.yaml`
- `config_halfcheetah.yaml`
- `config_ant.yaml`

자세한 내용은 [CONFIG_GUIDE.md](CONFIG_GUIDE.md)를 참조하세요.

## 문제 해결

### MuJoCo 설치 오류
```bash
conda install -c conda-forge mujoco
```

### GPU 메모리 부족
`config.yaml`에서 `batch_size`와 `updates_per_step` 값을 줄이세요.

## 참고 문서

- [CONFIG_GUIDE.md](CONFIG_GUIDE.md): 설정 파일 가이드

## 라이선스

교육 및 연구 목적으로 제공됩니다.
