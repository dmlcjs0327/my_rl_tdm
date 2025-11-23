"""
Gymnasium API 검증 스크립트
각 환경의 API를 확인하고 올바르게 작동하는지 테스트합니다.
"""
import gymnasium as gym
import numpy as np


def check_environment(env_name):
    """환경의 API를 확인하고 테스트합니다"""
    print(f"\n{'='*60}")
    print(f"환경 검증: {env_name}")
    print(f"{'='*60}")
    
    try:
        # 환경 생성
        env = gym.make(env_name)
        print(f"✅ 환경 생성 성공")
        
        # 환경 정보 출력
        print(f"\n환경 정보:")
        print(f"  Observation Space: {env.observation_space}")
        print(f"  Action Space: {env.action_space}")
        print(f"  Max Episode Steps: {env.spec.max_episode_steps if env.spec else 'N/A'}")
        
        # Reset 테스트
        obs, info = env.reset()
        print(f"\n✅ Reset 성공")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation dtype: {obs.dtype}")
        print(f"  Info keys: {info.keys()}")
        
        # Step 테스트
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n✅ Step 성공")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Info keys: {info.keys()}")
        
        # 여러 스텝 테스트
        print(f"\n여러 스텝 테스트...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                print(f"  Episode ended at step {i+1}, reset successful")
        
        print(f"✅ 여러 스텝 테스트 성공")
        
        # Render 테스트 (선택사항)
        try:
            env.render()
            print(f"✅ Render 성공")
        except Exception as e:
            print(f"⚠️  Render 실패 (선택사항): {e}")
        
        # Close
        env.close()
        print(f"✅ Close 성공")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_all_environments():
    """모든 환경을 확인합니다"""
    environments = [
        'Reacher-v5',
        'Pusher-v5',
        'HalfCheetah-v5',
        'Ant-v5',
        'Pendulum-v1',  # 추가 테스트
        'CartPole-v1',  # 추가 테스트
    ]
    
    results = {}
    
    for env_name in environments:
        results[env_name] = check_environment(env_name)
    
    # 결과 요약
    print(f"\n{'='*60}")
    print(f"검증 결과 요약")
    print(f"{'='*60}")
    
    for env_name, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{env_name}: {status}")
    
    print(f"\n성공: {sum(results.values())}/{len(results)}")
    
    return results


def check_gymnasium_version():
    """Gymnasium 버전 확인"""
    print(f"\n{'='*60}")
    print(f"Gymnasium 버전 정보")
    print(f"{'='*60}")
    
    try:
        import gymnasium as gym
        print(f"Gymnasium 버전: {gym.__version__}")
        
        # 환경 목록 확인
        print(f"\n사용 가능한 환경 목록:")
        envs = gym.envs.registry.all()
        
        mujoco_envs = [env.id for env in envs if 'MuJoCo' in env.entry_point or 'mujoco' in env.id.lower()]
        print(f"\nMuJoCo 환경 ({len(mujoco_envs)}개):")
        for env_id in sorted(mujoco_envs):
            print(f"  - {env_id}")
        
        return True
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False


def test_observation_extraction():
    """관찰값 추출 테스트"""
    print(f"\n{'='*60}")
    print(f"관찰값 추출 테스트")
    print(f"{'='*60}")
    
    test_cases = {
        'Reacher-v4': {
            'end_effector_indices': 'last 3 dims',
            'expected_shape': (3,)
        },
        'Pusher-v4': {
            'end_effector_indices': '8:12 (hand and puck XY)',
            'expected_shape': (4,)
        },
        'HalfCheetah-v4': {
            'velocity_index': 9,
            'expected_shape': (1,)
        },
        'Ant-v4': {
            'position_indices': '2:4 (XY)',
            'velocity_indices': '8:10 (VX, VY)',
            'expected_shape': (2,)
        }
    }
    
    for env_name, info in test_cases.items():
        print(f"\n{env_name}:")
        try:
            env = gym.make(env_name)
            obs, _ = env.reset()
            
            print(f"  Observation shape: {obs.shape}")
            print(f"  Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
            
            # Reacher: end effector (last 3 dims)
            if 'Reacher' in env_name:
                end_effector = obs[-3:]
                print(f"  End effector (last 3): {end_effector}")
            
            # Pusher: hand and puck XY
            elif 'Pusher' in env_name:
                hand_puck = obs[8:12]
                print(f"  Hand and puck XY: {hand_puck}")
            
            # HalfCheetah: velocity
            elif 'Cheetah' in env_name:
                velocity = obs[9]
                print(f"  Velocity (index 9): {velocity:.2f}")
            
            # Ant: position and velocity
            elif 'Ant' in env_name:
                position = obs[2:4]
                velocity = obs[8:10]
                print(f"  Position (2:4): {position}")
                print(f"  Velocity (8:10): {velocity}")
            
            env.close()
            print(f"  ✅ 성공")
            
        except Exception as e:
            print(f"  ❌ 실패: {e}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Gymnasium API 검증")
    print("="*60)
    
    # 버전 확인
    check_gymnasium_version()
    
    # 모든 환경 확인
    check_all_environments()
    
    # 관찰값 추출 테스트
    test_observation_extraction()
    
    print("\n" + "="*60)
    print("검증 완료!")
    print("="*60 + "\n")


