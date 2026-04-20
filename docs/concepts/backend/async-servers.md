---
type: engineering
primary_evidence: mvp_app/main.py, Dockerfile.mvp, requirements.txt
---

# Async Servers — ASGI, uvicorn, FastAPI sync/async

## Concept framework

### WSGI vs ASGI

**WSGI (Web Server Gateway Interface)** — Python 웹 서버의 전통적 계약. 요청 하나당 스레드·프로세스 하나를 할당해 handler가 완료될 때까지 대기. 예: Flask + gunicorn. 단순한 모델이지만 **I/O 대기 중에도 워커가 묶여 있어** 동시성이 낮고 자원 효율이 떨어진다.

**ASGI (Asynchronous Server Gateway Interface)** — 비동기 버전. handler가 `await`로 I/O 대기에 들어가면 이벤트 루프가 **그 대기를 중단시키지 않고 다른 요청 처리**. 단일 프로세스에서 수천 connection 유지 가능. WebSocket·Server-Sent Events 같은 long-lived 연결도 자연스럽게 지원.

### uvicorn 내부

uvicorn은 파이썬의 대표적 ASGI 서버. 구성:

- **`asyncio` 이벤트 루프** 위에서 동작 (Python 표준)
- **`uvloop` (Linux/macOS)** — libuv 기반 C 구현으로 asyncio 기본 대비 2~4× 빠름. Windows 미지원.
- **`httptools`** — HTTP 파싱을 C로 구현. 파이썬 parser 대비 대폭 속도 개선.
- **프로세스 모델** — 기본 단일 프로세스 (`--workers 1`). 다중 워커는 `--workers N` 또는 gunicorn 워커 클래스.

대안 ASGI 서버: `hypercorn` (HTTP/2·HTTP/3 지원), `daphne` (Django Channels 기원). uvicorn이 범용·빠름·안정으로 de facto 표준.

### FastAPI의 sync/async 공존 모델

FastAPI는 `async def` 와 `def` handler 둘 다 받지만 처리 경로가 다르다:

| 선언 | 처리 | 블로킹 특성 |
|---|---|---|
| `async def` | 이벤트 루프에서 직접 실행 | `await` 없는 동기 코드가 길면 **이벤트 루프 차단**. 다른 요청 못 받음. |
| `def` | **starlette 스레드풀**(기본 40 threads)에 dispatch | 이벤트 루프 차단 안 됨. 해당 요청은 스레드풀 slot 하나 소비. |

**함정**: `async def` 안에 **동기 DB 호출·파일 I/O·PIL 작업**을 직접 쓰면 async의 의미가 없다. 차라리 `def`로 선언하거나 `asyncio.to_thread(...)` / `ProcessPoolExecutor`로 밀어내야 한다.

**또한**: 스레드풀에 N초 걸리는 handler N+1 개가 동시 도착하면 스레드 고갈로 나머지는 대기. 40 threads는 MVP 초기엔 충분하지만 대규모 concurrent load에서는 한계.

### Compute-bound 작업에서 async가 도움이 안 되는 이유

Async의 이득은 **I/O 대기 중 양보**에 있다. GPU matmul, CPU-heavy 이미지 처리, 긴 암호화 연산 같은 **compute-bound** 작업은 대기가 아니라 **실제 계산을 하고 있으므로** 양보할 지점이 없다. `async def` 안에 put해도 여전히 이벤트 루프를 점유한다 (GIL도 관련).

해법: compute-bound handler를 `def`로 선언해 스레드풀로 보내거나, `asyncio.to_thread()` 로 명시적으로 밀어내 이벤트 루프를 해방시킨다. 진짜 scale out이 필요하면 **작업 큐**(Celery, RQ, Dramatiq)로 HTTP 응답과 실제 계산을 분리한다.

### Production 배포 패턴

- **gunicorn + uvicorn worker class**: `gunicorn -k uvicorn.workers.UvicornWorker -w N` — gunicorn이 프로세스 매니저, uvicorn이 각 워커 내부 ASGI 서버. graceful shutdown·hot reload·프로세스 모니터링에 유리.
- **단일 uvicorn**: `uvicorn app:app --workers N` — 더 단순. 프로세스 관리가 덜 정교.
- **리버스 프록시**: 대부분 nginx·caddy 뒤에 두고 uvicorn 자신은 HTTP/1.1만 담당. TLS·HTTP/2·static caching은 프록시가.
- **`--reload`는 dev only**. production에선 끄고 SIGHUP·SIGTERM 기반 graceful restart 전략.

## Evidence in this project

### 1. uvicorn이 default runtime — Dockerfile과 run script

- `Dockerfile.mvp:31` — `CMD ["uvicorn", "mvp_app.main:app", "--host", "0.0.0.0", "--port", "8090"]`
- `scripts/run_mvp.sh` — 로컬 개발용 uvicorn 기동
- 단일 프로세스 1 worker 기본. 이유는 GPU 모델 싱글턴과의 관계(아래 참조).

### 2. FastAPI handler의 sync/async 혼용

- `async def run_auto_label(...)` [`mvp_app/main.py:161`] — `await image.read()` 에서 async 이득 실현. 업로드 바디 수신 중 다른 요청 서빙 가능.
- `def api_workspace(...)` [`mvp_app/main.py:343`], `def api_runs(...)`, `def api_validate_object(...)` — 내부에 동기 sqlite3 + PIL 처리만 있어 `async def`로 감쌀 실익 없음. 스레드풀 dispatch로 병렬성 확보.

### 3. GPU 싱글턴과 다중 워커의 충돌

- `mvp_app/detector.py` `DetectionService` / `mvp_app/segmenter.py` `SegmentationService` 둘 다 **프로세스별 싱글턴**. Florence-2 (~0.5-1.5 GB) + SAM3 가 프로세스마다 로드된다.
- `uvicorn --workers 4` 를 돌리면 RTX 4090 24 GB 에서도 **2-3 워커만 돼도 VRAM 포화**. MVP가 1 worker + thread pool 을 택하는 본질적 이유.
- 진짜 수평 scale out을 원하면 inference를 **별도 서비스**로 분리하고 게이트웨이만 다중 워커로 가는 구조(원래 microservices 계획 `docker-compose.legacy.yml` 에서 이 설계 확인 가능).

### 4. Compute-bound inference는 암묵적 스레드풀 사용

- `run_auto_label` 의 `async def` 내에서 `detector.detect(...)` 호출은 동기 GPU 호출. 이 handler가 `async def`라 **이벤트 루프 위에서 GPU 작업이 블록**되는 상황. 현재 MVP에서 이 경로가 유일한 이벤트 루프 consumer라 문제 안 됨.
- 동시에 리뷰·export API를 호출하면 이들은 `def` → 스레드풀로 감 → 이벤트 루프 자유 → OK.
- 향후 개선: `detector.detect` 를 `await asyncio.to_thread(detector.detect, ...)`로 래핑하면 이벤트 루프를 inference 중에도 해방.

### 5. dev vs prod uvloop 격차

- dev 환경은 Windows — asyncio 기본 루프 사용. uvloop 설치 불가.
- prod 컨테이너 (`Dockerfile.mvp`)는 Linux — uvloop 사용 시 2-4× 이벤트 루프 처리량.
- 이 격차가 관측되면 `requirements.txt`에 `uvloop; sys_platform != 'win32'` 같은 조건부 설치로 명시해 환경간 동작 차이를 문서화한다.

## Related rules / decisions

- [LEARNING: TestClient는 lifespan을 위해 with 컨텍스트로 쓴다](../../progress/LEARNINGS.md) — FastAPI lifespan은 이 async 모델의 일부
- [WORKLOG 2026-04-20 UI 전체 구현](../../progress/WORKLOG.md) — `/api/review/workspace`, `/api/runs` 등 본 concept의 핵심 evidence 엔드포인트 신설
- [WORKLOG 2026-04-21 Wave B.1](../../progress/WORKLOG.md) — `nested try` 구조가 async handler 안에서 failure를 깔끔히 surfacing

## Open follow-ups

- `asyncio.to_thread(detector.detect, ...)` 로 inference 스레드풀 명시화 — 동시 요청시 이벤트 루프 해방 효과 측정
- `gunicorn -k uvicorn.workers.UvicornWorker` production 스크립트 작성 (Phase 4 Deployment Ops)
- inference를 별도 서비스로 분리하고 게이트웨이만 다중 워커로 scale — 이미 `services/gateway/` 레거시에 유사 구조 있음, MVP 확장 시 참고
