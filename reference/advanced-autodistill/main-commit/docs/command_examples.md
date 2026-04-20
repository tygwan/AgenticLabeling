# 카테고리 관리 및 데이터 처리 명령어 예시

이 문서는 프로젝트의 데이터 처리 및 카테고리 관리를 위한 명령어 예시를 제공합니다.

## 카테고리 관리 (manage_categories.py)

### 새 카테고리 생성

```bash
# 기본 데이터 디렉토리에 새 카테고리 생성
python scripts/manage_categories.py create my_category

# 사용자 지정 데이터 디렉토리에 카테고리 생성
python scripts/manage_categories.py create my_category --base-dir /path/to/data
```

### 카테고리 목록 표시

```bash
# 모든 유효한 카테고리 나열
python scripts/manage_categories.py list

# 사용자 지정 디렉토리의 카테고리 나열
python scripts/manage_categories.py list --base-dir /path/to/data
```

### YOLO 데이터셋 구성 파일 생성

```bash
# YOLO 훈련용 data.yaml 파일 생성
python scripts/manage_categories.py create-yaml my_category --classes person car bicycle dog
```

### 카테고리 경로 가져오기

```bash
# 카테고리 루트 경로 표시
python scripts/manage_categories.py get-path my_category

# 이미지 경로 표시
python scripts/manage_categories.py get-path my_category --type images

# 데이터셋 경로 표시
python scripts/manage_categories.py get-path my_category --type dataset
```

## Material 폴더 관리 (material_utils.py)

### Material 카테고리 목록

```bash
# Material 폴더 내 모든 카테고리 나열
python scripts/material_utils.py list
```

### Material 폴더에서 데이터 폴더로 이미지 복사

```bash
# Material 폴더에서 데이터 폴더로 이미지 복사
python scripts/material_utils.py copy my_category
```

### Autodistill 어노테이션을 YOLO 형식으로 변환

```bash
# 클래스 매핑 파일을 사용하여 어노테이션 변환
python scripts/material_utils.py convert my_category annotations.json --class-mapping class_mapping.json

# 직접 이미지 크기 지정
python scripts/material_utils.py convert my_category annotations.json --img-width 1280 --img-height 720
```

## MCP 설정 관리 (update_mcp.py)

### MCP 구성 보기

```bash
# 현재 MCP 설정 표시
python scripts/update_mcp.py
```

### MCP 구성 업데이트

```bash
# 웹훅 URL 업데이트
python scripts/update_mcp.py --webhook-url https://example-tunnel.trycloudflare.com --update-config

# 포트 업데이트
python scripts/update_mcp.py --port 8080 --update-config
```

### 명령 표시

```bash
# Cloudflare 터널 명령 표시
python scripts/update_mcp.py --show-cloudflare-command

# Docker 실행 명령 표시
python scripts/update_mcp.py --show-docker-command
``` 