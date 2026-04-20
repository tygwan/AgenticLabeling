#!/bin/bash

# Support Set 관리 도구 실행 스크립트

echo "Support Set 관리 도구를 시작합니다..."

# 스크립트 실행 디렉토리로 변경
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 필요한 환경변수 설정
export PYTHONPATH="$PWD:$PYTHONPATH"

# Python 가상 환경 활성화 (필요한 경우)
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
fi

# 인수 확인
if [ -z "$1" ]; then
    echo "사용법: $0 <카테고리> [options]"
    echo "옵션:"
    echo "  --create       구조화된 support set 생성"
    echo "  --validate     Support set 유효성 검증"
    echo "  --report       Support set 상태 보고서 생성"
    echo "  --force        기존 구조화된 support set 덮어쓰기"
    echo ""
    echo "예시:"
    echo "  $0 test_category --create --force   # test_category의 구조화된 support set 강제 생성"
    echo "  $0 test_category --report           # test_category의 support set 상태 보고서 생성"
    exit 1
fi

CATEGORY="$1"
shift

# 옵션 변수 초기화
CREATE=false
VALIDATE=false
REPORT=false
FORCE=false

# 옵션 파싱
while [ "$#" -gt 0 ]; do
    case "$1" in
        --create)
            CREATE=true
            ;;
        --validate)
            VALIDATE=true
            ;;
        --report)
            REPORT=true
            ;;
        --force)
            FORCE=true
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            exit 1
            ;;
    esac
    shift
done

# 명령줄 구성
CMD="python scripts/support_set_manager.py --category $CATEGORY"

if $VALIDATE; then
    CMD="$CMD --validate"
fi

if $REPORT; then
    CMD="$CMD --report"
fi

if $CREATE; then
    CMD="$CMD --create"
fi

if $FORCE; then
    CMD="$CMD --force"
fi

# 명령 실행
echo "실행 명령: $CMD"
eval $CMD

# 종료 코드 확인
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Support Set 관리 도구가 성공적으로 완료되었습니다."
else
    echo "Support Set 관리 도구가 오류와 함께 종료되었습니다 (코드: $EXIT_CODE)."
fi

exit $EXIT_CODE 