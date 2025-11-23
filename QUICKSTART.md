# TDM 빠른 시작 가이드

이 가이드는 TDM을 빠르게 시작하는 방법을 설명합니다.

## 1. 설치 (5분)

### 필수 요구사항
- Anaconda 또는 Miniconda
- Python 3.9 이상
- PyTorch 2.0 이상
- Gymnasium

### 설치 명령어 (아나콘다 - 권장)

```bash
# 1. 아나콘다 환경 생성
conda env create -f environment.yml

# 2. 환경 활성화
conda activate tdm

# 3. 환경 확인
python --version  # Python 3.9 이상 확인
```

### 설치 명령어 (pip 사용)

```bash
# 1. 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. MuJoCo 환경 설치 (로봇 환경 사용 시)
pip install gymnasium[mujoco]
```

## 2. 빠른 테스트 (2분)

구현이 제대로 작동하는지 확인합니다:

```bash
python test_tdm.py
```

모든 테스트가 통과하면 ✓ 표시가 나타납니다.

## 3. 간단한 훈련 (10분)

가장 간단한 예제로 TDM을 훈련합니다:

```bash
# Reacher 환경에서 훈련 (간단한 작업)
# config.yaml에서 name을 "Reacher-v5"로 설정
python train.py
```

훈련이 시작되면:
- 콘솔에 진행 상황이 출력됩니다
- `./logs/` 폴더에 로그가 저장됩니다
- TensorBoard로 시각화 가능합니다

### TensorBoard로 모니터링

새 터미널에서:

```bash
tensorboard --logdir=./logs
```

브라우저에서 `http://localhost:6006` 접속

## 4. 훈련된 모델 평가 (3분)

훈련이 완료되면 모델을 평가합니다:

```bash
# 기본 평가
python evaluate.py --model ./logs/Reacher-v5_*/model_final.pt

# 시각화와 함께 평가
python evaluate.py --model ./logs/Reacher-v5_*/model_final.pt --render

# 궤적 시각화
python evaluate.py --model ./logs/Reacher-v5_*/model_final.pt --visualize
```

## 5. 다른 환경으로 변경

### Pusher 환경

`config.yaml` 파일을 열고:

```yaml
env:
  name: "Pusher-v5"
  max_episode_steps: 50
```

그 다음 훈련:

```bash
python train.py
```

### HalfCheetah 환경

```yaml
env:
  name: "HalfCheetah-v5"
  max_episode_steps: 99
```

### Ant 환경

```yaml
env:
  name: "Ant-v5"
  max_episode_steps: 50

task:
  locomotion_task_type: "position"  # 또는 "position_velocity"
```

## 6. 하이퍼파라미터 튜닝

성능을 향상시키려면 `config.yaml`에서 조정:

### 빠른 학습 (낮은 성능)

```yaml
tdm:
  tau_max: 10  # 작은 값

training:
  updates_per_step: 5  # 적은 업데이트
  total_timesteps: 100000  # 짧은 훈련
```

### 높은 성능 (느린 학습)

```yaml
tdm:
  tau_max: 25  # 큰 값

training:
  updates_per_step: 20  # 많은 업데이트
  total_timesteps: 5000000  # 긴 훈련
```

## 7. 일반적인 문제 해결

### 문제: GPU 메모리 부족

**해결책**: `config.yaml`에서 배치 크기 줄이기

```yaml
training:
  batch_size: 64  # 128에서 64로
```

### 문제: 학습이 너무 느림

**해결책**: 업데이트 빈도 줄이기

```yaml
training:
  updates_per_step: 1  # 10에서 1로
```

### 문제: 성능이 낮음

**해결책**:
1. 더 오래 훈련 (`total_timesteps` 증가)
2. `tau_max` 조정 (15-25 시도)
3. `reward_scale` 조정 (0.1-10 시도)

### 문제: 환경을 찾을 수 없음

**해결책**: Gymnasium 환경 설치

```bash
# 아나콘다 환경에서
conda install -c conda-forge mujoco

# 또는 pip로
pip install gymnasium[mujoco]
```

### 문제: 아나콘다 환경이 활성화되지 않음

**해결책**: 환경 활성화 확인

```bash
# 환경 활성화
conda activate tdm

# 환경 목록 확인
conda env list

# 환경 삭제 후 재생성 (필요시)
conda env remove -n tdm
conda env create -f environment.yml
```

## 8. 예제 코드

### Python 스크립트에서 사용

```python
import yaml
import torch
from tdm import TDM
from env_wrapper import TDMEnvWrapper
from mpc_planner import TaskSpecificPlanner

# 설정 로드
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 환경 생성
import gymnasium as gym
env = gym.make('Reacher-v5')
env = TDMEnvWrapper(env, 'end_effector', config['task'])

# TDM 생성
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
goal_dim = env.goal_dim
action_range = (env.action_space.low, env.action_space.high)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tdm = TDM(state_dim, action_dim, goal_dim, action_range, config, device)

# 훈련된 모델 로드
tdm.load('./logs/Reacher-v5_*/model_final.pt')

# Planner 생성
planner = TaskSpecificPlanner(tdm, config, 'Reacher-v5', 'end_effector')

# 실행
obs, info = env.reset()
goal = env.get_goal()

for step in range(100):
    action = planner.select_action(obs, goal, tau=25)
    obs, reward, done, info = env.step(action)
    
    if done:
        break

env.close()
```

## 9. 다음 단계

### 더 배우기

1. **README.md** - 전체 문서
2. **example_usage.py** - 다양한 예제
3. **utils.py** - 분석 도구

### 고급 기능

1. **Goal Relabeling 분석**: `utils.py`의 `analyze_goal_relabeling_impact()`
2. **Horizon 비교**: `utils.py`의 `compare_horizons()`
3. **훈련 곡선 시각화**: `utils.py`의 `plot_training_curves()`

### 논문 재현

논문의 실험을 재현하려면:

1. 각 환경에 대해 별도로 훈련
2. 여러 random seed로 실행 (seed 변경)
3. TensorBoard로 결과 비교
4. `utils.py`의 도구로 분석

## 10. 빠른 참조

### 주요 명령어

```bash
# 훈련
python train.py

# 평가
python evaluate.py --model <model_path>

# 테스트
python test_tdm.py

# 예제 실행
python example_usage.py
```

### 주요 파일

- `config.yaml` - 설정
- `train.py` - 훈련 스크립트
- `evaluate.py` - 평가 스크립트
- `tdm.py` - TDM 알고리즘
- `networks.py` - 신경망 구조
- `mpc_planner.py` - MPC 계획

### 주요 디렉토리

- `./logs/` - 훈련 로그 및 모델
- `./logs/*/` - 환경별 로그

## 도움말

문제가 있으면:
1. `test_tdm.py` 실행하여 문제 확인
2. TensorBoard 로그 확인
3. `config.yaml` 설정 검토
4. GitHub Issues에 문제 보고

행운을 빕니다! 🚀

