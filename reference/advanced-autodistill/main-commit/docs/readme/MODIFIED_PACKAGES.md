# 수정된 패키지 모듈 관리

이 문서는 Python 패키지 모듈을 수정했을 때 Git으로 관리하는 방법을 설명합니다.

## 문제 상황

일반적으로 Python 가상 환경의 `lib/`, `lib64/` 디렉토리는 `.gitignore`에 의해 Git 추적에서 제외됩니다. 하지만 패키지 모듈을 수정했다면, 이 변경사항도 Git으로 추적해야 합니다.

## 해결 방법

### 1. 수정된 파일 추적하기

수정한 패키지 파일을 Git에 추가하려면 `-f` (force) 옵션을 사용해야 합니다:

```bash
git add -f lib/python3.10/site-packages/패키지명/수정한파일.py
```

예:
```bash
git add -f lib/python3.10/site-packages/autodistill/detection.py
```

### 2. 스크립트 사용하기

편의를 위해 `track_modified_packages.sh` 스크립트를 제공합니다:

```bash
./track_modified_packages.sh lib/python3.10/site-packages/패키지명/수정한파일.py
```

### 3. 수정된 파일 확인하기

현재 Git이 추적 중인 lib 디렉토리의 파일을 확인하려면:

```bash
git ls-files lib/
```

## 주의사항

1. **선택적 추적**: 수정한 파일만 Git에 추가하고, 나머지 가상 환경 파일은 여전히 `.gitignore`에 의해 무시됩니다.

2. **명시적 문서화**: 어떤 패키지를 수정했는지, 왜 수정했는지 문서화하는 것이 좋습니다.

3. **배포 고려사항**: 수정된 패키지가 있는 프로젝트를 배포할 때, 이 수정사항을 어떻게 처리할지 계획이 필요합니다.

## 패키지 수정 대신 고려할 수 있는 대안

1. **패치 적용**: 원본 코드를 수정하는 대신 monkey patching을 사용할 수 있습니다.

2. **포크 및 설치**: 패키지를 포크하여 수정 후, 이를 pip로 설치할 수 있습니다.

3. **래퍼 클래스**: 기존 클래스를 상속하여 필요한 기능만 오버라이드할 수 있습니다.

---

## 현재 수정된 패키지 목록

이 프로젝트에서 수정된 패키지 모듈들을 아래에 기록하세요:

1. `lib/python3.10/site-packages/autodistill/detection.py` - 로컬 모델 경로 지원을 위해 수정
2. (다른 수정된 패키지 모듈들을 여기에 추가...) 