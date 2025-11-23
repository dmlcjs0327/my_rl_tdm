# 아나콘다 환경 설정 가이드

TDM 프로젝트를 아나콘다 환경에서 실행하기 위한 상세 가이드입니다.

## 📋 목차

1. [아나콘다 설치](#아나콘다-설치)
2. [환경 생성](#환경-생성)
3. [환경 활성화/비활성화](#환경-활성화비활성화)
4. [패키지 관리](#패키지-관리)
5. [문제 해결](#문제-해결)

## 아나콘다 설치

### Windows

1. [Anaconda 공식 웹사이트](https://www.anaconda.com/products/distribution)에서 다운로드
2. 설치 프로그램 실행
3. "Add Anaconda to PATH" 옵션 선택 (권장)

### Linux/Mac

```bash
# 다운로드
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

# 설치
bash Anaconda3-2023.09-0-Linux-x86_64.sh

# 재시작 후 확인
conda --version
```

### Miniconda (경량 버전)

더 가벼운 버전을 원한다면 Miniconda를 사용할 수 있습니다:

```bash
# 다운로드
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 설치
bash Miniconda3-latest-Linux-x86_64.sh
```

## 환경 생성

### 방법 1: 자동 스크립트 사용 (권장)

#### Windows

```cmd
setup_conda.bat
```

#### Linux/Mac

```bash
chmod +x setup_conda.sh
./setup_conda.sh
```

### 방법 2: 수동 생성

```bash
# 1. 환경 생성
conda env create -f environment.yml

# 2. 생성 확인
conda env list
```

### 방법 3: 단계별 생성

```bash
# 1. Python 3.9 환경 생성
conda create -n tdm python=3.9 -y

# 2. 환경 활성화
conda activate tdm

# 3. PyTorch 설치 (CUDA 버전 선택)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. 기타 패키지 설치
conda install numpy matplotlib pyyaml -c conda-forge

# 5. pip 패키지 설치
pip install gymnasium tensorboard tqdm mujoco
```

## 환경 활성화/비활성화

### 환경 활성화

```bash
conda activate tdm
```

활성화되면 프롬프트에 `(tdm)`이 표시됩니다.

### 환경 비활성화

```bash
conda deactivate
```

### 환경 확인

```bash
# 현재 활성화된 환경 확인
conda info --envs

# Python 버전 확인
python --version

# 설치된 패키지 확인
conda list
```

## 패키지 관리

### 패키지 설치

```bash
# conda로 설치 (권장)
conda install package_name -c conda-forge

# pip로 설치
pip install package_name
```

### 패키지 업데이트

```bash
# 특정 패키지 업데이트
conda update package_name

# 모든 패키지 업데이트
conda update --all

# pip 패키지 업데이트
pip install --upgrade package_name
```

### 패키지 제거

```bash
# conda로 제거
conda remove package_name

# pip로 제거
pip uninstall package_name
```

### 환경 내보내기

```bash
# 현재 환경을 YAML 파일로 내보내기
conda env export > environment.yml

# pip 패키지만 내보내기
pip freeze > requirements.txt
```

## 문제 해결

### 문제 1: conda 명령어를 찾을 수 없음

**증상**: `conda: command not found`

**해결책**:

#### Windows
1. Anaconda Prompt 사용
2. 또는 시스템 환경 변수 PATH에 Anaconda 추가

#### Linux/Mac
```bash
# .bashrc 또는 .zshrc에 추가
export PATH="$HOME/anaconda3/bin:$PATH"

# 적용
source ~/.bashrc  # 또는 source ~/.zshrc
```

### 문제 2: 환경 생성 실패

**증상**: `ResolvePackageNotFound` 오류

**해결책**:
```bash
# 1. conda 업데이트
conda update conda

# 2. 채널 추가
conda config --add channels conda-forge

# 3. 다시 시도
conda env create -f environment.yml
```

### 문제 3: PyTorch CUDA 버전 문제

**증상**: CUDA 관련 오류

**해결책**:
```bash
# CUDA 버전 확인
nvidia-smi

# 해당 CUDA 버전에 맞는 PyTorch 설치
# CUDA 11.8 예시
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CPU 버전만 사용
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 문제 4: 환경 충돌

**증상**: 패키지 버전 충돌

**해결책**:
```bash
# 1. 환경 삭제
conda env remove -n tdm

# 2. 캐시 정리
conda clean --all

# 3. 환경 재생성
conda env create -f environment.yml
```

### 문제 5: MuJoCo 설치 오류

**증상**: MuJoCo 관련 오류

**해결책**:
```bash
# conda로 설치 시도
conda install -c conda-forge mujoco

# 또는 pip로
pip install mujoco

# 버전 확인
python -c "import mujoco; print(mujoco.__version__)"
```

## 유용한 명령어

### 환경 관리

```bash
# 모든 환경 목록
conda env list

# 환경 복사
conda create --name tdm_backup --clone tdm

# 환경 삭제
conda env remove -n tdm

# 환경 이름 변경
conda create --name new_name --clone tdm
conda env remove -n tdm
```

### 패키지 검색

```bash
# 패키지 검색
conda search package_name

# 설치된 패키지 검색
conda list | grep package_name
```

### 캐시 관리

```bash
# 캐시 확인
conda clean --dry-run --all

# 캐시 정리
conda clean --all
```

## Jupyter Notebook 사용

아나콘다 환경에서 Jupyter Notebook을 사용하려면:

```bash
# Jupyter 설치
conda install jupyter ipykernel -c conda-forge

# 커널 등록
python -m ipykernel install --user --name tdm --display-name "Python (TDM)"

# Jupyter 실행
jupyter notebook
```

## VS Code 연동

VS Code에서 아나콘다 환경을 사용하려면:

1. Python 확장 설치
2. `Ctrl+Shift+P` → "Python: Select Interpreter"
3. `tdm` 환경 선택
4. 또는 `.vscode/settings.json`에 추가:

```json
{
    "python.defaultInterpreterPath": "C:\\Users\\YourName\\anaconda3\\envs\\tdm\\python.exe"
}
```

## 성능 최적화

### MKL 사용

```bash
# Intel MKL 설치 (선택사항)
conda install mkl mkl-service
```

### GPU 사용

```bash
# CUDA 버전 확인
nvidia-smi

# 해당 버전 PyTorch 설치
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# GPU 사용 확인
python -c "import torch; print(torch.cuda.is_available())"
```

## 추가 리소스

- [Anaconda 공식 문서](https://docs.anaconda.com/)
- [Conda 사용 가이드](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)

## 요약

```bash
# 환경 생성
conda env create -f environment.yml

# 환경 활성화
conda activate tdm

# 코드 실행
python train.py

# 환경 비활성화
conda deactivate
```

더 많은 정보는 `README.md`와 `QUICKSTART.md`를 참조하세요.









