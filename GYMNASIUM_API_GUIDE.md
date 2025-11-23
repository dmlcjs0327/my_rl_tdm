# Gymnasium API 가이드

이 문서는 TDM 프로젝트에서 사용하는 Gymnasium API에 대한 가이드입니다.

## 목차

1. [Gymnasium 개요](#gymnasium-개요)
2. [환경 API](#환경-api)
3. [지원 환경](#지원-환경)
4. [API 변경사항](#api-변경사항)
5. [문제 해결](#문제-해결)

## Gymnasium 개요

Gymnasium은 OpenAI Gym의 포크로, 강화학습 환경을 제공하는 표준 라이브러리입니다.

### 주요 특징

- **표준화된 인터페이스**: 모든 환경이 동일한 API 사용
- **확장성**: 다양한 환경 지원 (MuJoCo, Atari, Classic Control 등)
- **버전 관리**: 환경 버전 관리 (v1, v2, v3, v4 등)

### 설치

```bash
# 기본 설치
pip install gymnasium

# MuJoCo 환경 포함
pip install gymnasium[mujoco]

# 또는
conda install -c conda-forge gymnasium
```

## 환경 API

### 기본 사용법

```python
import gymnasium as gym

# 환경 생성
env = gym.make('Reacher-v5')

# 환경 초기화
obs, info = env.reset()

# 스텝 실행
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# 환경 종료
env.close()
```

### 핵심 메서드

#### 1. `reset(seed=None, options=None)`

환경을 초기 상태로 재설정합니다.

```python
obs, info = env.reset()
```

**반환값**:
- `obs`: 초기 관찰값 (numpy array)
- `info`: 추가 정보 (dict)

**파라미터**:
- `seed`: 재현성을 위한 시드 (선택사항)
- `options`: 환경별 옵션 (선택사항)

#### 2. `step(action)`

주어진 행동을 환경에 적용합니다.

```python
obs, reward, terminated, truncated, info = env.step(action)
```

**반환값**:
- `obs`: 다음 관찰값
- `reward`: 보상 (float)
- `terminated`: 에피소드가 자연스럽게 종료되었는지 (bool)
- `truncated`: 에피소드가 시간 제한으로 잘렸는지 (bool)
- `info`: 추가 정보 (dict)

**중요**: Gymnasium v0.26+에서는 `done` 대신 `terminated`와 `truncated`를 분리합니다.

#### 3. `render()`

환경을 시각화합니다.

```python
env.render()
```

#### 4. `close()`

환경을 종료하고 리소스를 해제합니다.

```python
env.close()
```

### 환경 속성

```python
# 관찰 공간
print(env.observation_space)
# Box(-inf, inf, (11,), float32)

# 행동 공간
print(env.action_space)
# Box(-1.0, 1.0, (2,), float32)

# 최대 에피소드 길이
print(env.spec.max_episode_steps)
# 50

# 환경 ID
print(env.spec.id)
# 'Reacher-v5'
```

## 지원 환경

### 1. Reacher-v5

**설명**: 2D 로봇 팔이 목표 지점에 도달하는 작업

**관찰 공간**: 
- Shape: (11,)
- 구성: 6개 관절 각도/속도 + 3개 end-effector 위치 + 2개 목표 위치

**행동 공간**:
- Shape: (2,)
- 범위: [-1, 1] (토크)

**Goal 추출**:
```python
# End-effector 위치 (마지막 3개 차원)
end_effector = obs[-3:]
```

### 2. Pusher-v5

**설명**: 로봇 팔이 퍽을 테이블 위에서 목표 위치로 밀기

**관찰 공간**:
- Shape: (23,)
- 구성: 7개 관절 각도/속도 + hand XY + puck XY + 목표 XY

**행동 공간**:
- Shape: (7,)
- 범위: [-2, 2] (토크)

**Goal 추출**:
```python
# Hand와 puck의 XY 위치
hand_puck = obs[8:12]  # [hand_x, hand_y, puck_x, puck_y]
```

### 3. HalfCheetah-v5

**설명**: 2D 케타가 목표 속도로 달리기

**관찰 공간**:
- Shape: (18,)
- 구성: 위치, 속도, 관절 각도/속도

**행동 공간**:
- Shape: (6,)
- 범위: [-1, 1] (토크)

**Goal 추출**:
```python
# 속도 (index 9)
velocity = obs[9]
```

### 4. Ant-v5

**설명**: 4족 보행 로봇이 목표 위치로 이동

**관찰 공간**:
- Shape: (27,)
- 구성: 위치, 속도, 관절 각도/속도

**행동 공간**:
- Shape: (8,)
- 범위: [-1, 1] (토크)

**Goal 추출**:
```python
# 위치 (XY)
position = obs[2:4]

# 속도 (VX, VY)
velocity = obs[8:10]

# 위치 + 속도
goal = np.concatenate([position, velocity])
```

## API 변경사항

### Gymnasium v0.26+ 주요 변경

#### 1. `done` → `terminated`, `truncated`

**이전 (Gym v0.21)**:
```python
obs, reward, done, info = env.step(action)
```

**현재 (Gymnasium v0.26+)**:
```python
obs, reward, terminated, truncated, info = env.step(action)
```

**차이점**:
- `terminated`: 에피소드가 자연스럽게 종료됨 (예: 목표 도달)
- `truncated`: 에피소드가 시간 제한으로 잘림
- `done = terminated or truncated`

#### 2. `reset()` 반환값

**이전**:
```python
obs = env.reset()
```

**현재**:
```python
obs, info = env.reset()
```

### TDM 프로젝트에서의 처리

```python
# env_wrapper.py
def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    
    # Goal 도달 시 terminated 업데이트
    terminated = terminated or goal_reached
    
    return obs, tdm_reward, terminated, truncated, info

# train.py
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

## 문제 해결

### 문제 1: "TypeError: too many values to unpack"

**원인**: `step()` 메서드가 5개 값을 반환하는데 4개만 받으려고 함

**해결**:
```python
# 잘못된 코드
obs, reward, done, info = env.step(action)

# 올바른 코드
obs, reward, terminated, truncated, info = env.step(action)
```

### 문제 2: 환경을 찾을 수 없음

**원인**: MuJoCo 환경이 설치되지 않음

**해결**:
```bash
# MuJoCo 설치
pip install gymnasium[mujoco]

# 또는
conda install -c conda-forge mujoco
```

### 문제 3: 환경 버전 불일치

**원인**: 환경 버전이 존재하지 않음

**해결**:
```bash
# 사용 가능한 환경 확인
python -c "import gymnasium as gym; print([env.id for env in gym.envs.registry.all() if 'Reacher' in env.id])"

# 출력: ['Reacher-v5', 'Reacher-v5', ...]
```

### 문제 4: Observation shape 불일치

**원인**: 환경 버전에 따라 observation shape이 다를 수 있음

**해결**:
```python
# 환경 생성 후 확인
env = gym.make('Reacher-v5')
print(env.observation_space.shape)
print(env.action_space.shape)
```

## API 검증 스크립트

프로젝트에 포함된 `gymnasium_api_check.py`를 실행하여 환경을 검증할 수 있습니다:

```bash
python gymnasium_api_check.py
```

이 스크립트는:
- Gymnasium 버전 확인
- 각 환경의 API 테스트
- Observation shape 확인
- Step/Reset 동작 검증

## 추가 리소스

- [Gymnasium 공식 문서](https://gymnasium.farama.org/)
- [Gymnasium GitHub](https://github.com/Farama-Foundation/Gymnasium)
- [MuJoCo 환경 문서](https://gymnasium.farama.org/environments/mujoco/)
- [API 변경사항](https://gymnasium.farama.org/content/migration-guide/)

## 요약

### 기본 사용 패턴

```python
import gymnasium as gym

# 환경 생성
env = gym.make('Reacher-v5')

# 에피소드 실행
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

# 환경 종료
env.close()
```

### TDM 프로젝트에서의 사용

```python
# 환경 생성 및 래핑
env = gym.make('Reacher-v5')
env = TDMEnvWrapper(env, task_type='end_effector', config=config)

# 훈련 루프
obs, info = env.reset()
goal = env.get_goal()

for step in range(max_steps):
    action = planner.select_action(obs, goal, tau)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
        goal = env.get_goal()
```

이 가이드가 도움이 되기를 바랍니다! 🚀


