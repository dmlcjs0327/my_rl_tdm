@echo off
REM 아나콘다 환경 설정 스크립트 (Windows)

echo ==========================================
echo TDM 아나콘다 환경 설정
echo ==========================================
echo.

REM 아나콘다가 설치되어 있는지 확인
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [오류] 아나콘다가 설치되어 있지 않습니다.
    echo 다음 링크에서 아나콘다를 설치하세요:
    echo https://www.anaconda.com/products/distribution
    pause
    exit /b 1
)

echo [확인] 아나콘다가 설치되어 있습니다.
echo.

REM 기존 환경 확인
conda env list | findstr "^tdm " >nul
if %errorlevel% equ 0 (
    echo [경고] 기존 'tdm' 환경이 있습니다.
    set /p response="삭제하고 다시 생성하시겠습니까? (y/n): "
    if /i "%response%"=="y" (
        echo 기존 환경 삭제 중...
        conda env remove -n tdm -y
        echo [완료] 기존 환경이 삭제되었습니다.
    ) else (
        echo 설정을 취소했습니다.
        pause
        exit /b 0
    )
)

REM 환경 생성
echo.
echo 아나콘다 환경 생성 중...
conda env create -f environment.yml

if %errorlevel% equ 0 (
    echo.
    echo ==========================================
    echo [완료] 환경 설정 완료!
    echo ==========================================
    echo.
    echo 다음 명령어로 환경을 활성화하세요:
    echo   conda activate tdm
    echo.
    echo 환경 비활성화:
    echo   conda deactivate
    echo.
    echo 환경 삭제:
    echo   conda env remove -n tdm
    echo.
    pause
) else (
    echo.
    echo [오류] 환경 생성 실패
    pause
    exit /b 1
)









