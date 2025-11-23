#!/bin/bash
# 아나콘다 환경 설정 스크립트

echo "=========================================="
echo "TDM 아나콘다 환경 설정"
echo "=========================================="
echo ""

# 아나콘다가 설치되어 있는지 확인
if ! command -v conda &> /dev/null
then
    echo "❌ 오류: 아나콘다가 설치되어 있지 않습니다."
    echo "다음 링크에서 아나콘다를 설치하세요:"
    echo "https://www.anaconda.com/products/distribution"
    exit 1
fi

echo "✅ 아나콘다가 설치되어 있습니다."
echo ""

# 기존 환경이 있으면 삭제
if conda env list | grep -q "^tdm "; then
    echo "⚠️  기존 'tdm' 환경이 있습니다."
    read -p "삭제하고 다시 생성하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "기존 환경 삭제 중..."
        conda env remove -n tdm -y
        echo "✅ 기존 환경이 삭제되었습니다."
    else
        echo "설정을 취소했습니다."
        exit 0
    fi
fi

# 환경 생성
echo ""
echo "아나콘다 환경 생성 중..."
conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 환경 설정 완료!"
    echo "=========================================="
    echo ""
    echo "다음 명령어로 환경을 활성화하세요:"
    echo "  conda activate tdm"
    echo ""
    echo "환경 비활성화:"
    echo "  conda deactivate"
    echo ""
    echo "환경 삭제:"
    echo "  conda env remove -n tdm"
    echo ""
else
    echo ""
    echo "❌ 환경 생성 실패"
    exit 1
fi




