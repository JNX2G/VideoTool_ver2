# VideoTool Repository 관리 가이드
---

## 목적
작업 충돌을 최소화하고, 변경 이력을 명확히 남기며, 최종 산출물의 품질을 안정적으로 유지한다.

## 기본 규칙
- `main` 브랜치에는 최종 확정본만 반영한다.
- 개인 작업은 각자 브랜치에서만 진행한다.
- 회의 이후 확정된 수정사항은 커밋으로 남긴다.
- `main` 반영은 `Pull Request`(또는 동등한 리뷰 절차)로만 진행한다.

## GitHub Desktop 표준 작업 흐름
- 작업 시작 전
    - **GitHub Desktop**에서 `Fetch origin` 클릭
    - 현재 브랜치가 본인 브랜치인지 확인
    - `main` 최신 반영
- 기능 및 수정 작업 진행
    - 변경 단위를 가능한 작게 유지
    - 작업 단위가 끝나면 커밋
- 커밋 방법
    - `Changes` 탭에서 파일 선택(필요한 것만 체크)
    - `Summary`(제목), `Description`(내용) 작성
    - 확정 커밋이면 `Push origin`까지 수행

## 커밋 메시지 규칙
- 형식
    - `Summary`: `type: 요약`
    - `Description`에 무엇을/왜/영향 범위 기록
- `type` 종류
    - `feat`: 기능 추가
    - `fix`: 버그 수정
    - `refactor`: 리팩토링(동작 변경 최소)
    - `chore`: 설정/잡무(의존성, 빌드 설정)
    - `docs`: 문서
    - `test`: 테스트
- 작성 예시
    - `feat: 영상 업로드 전처리 추가 (videos)`
    - `fix: FBV->CBV 변환 시 프레임 누락 수정 (analysis)`
    - `refactor: 노드 실행 파이프라인 구조 정리 (core)`

## 충돌 처리 규칙
- 충돌이 나면 우선 `main`을 기준으로 해결 방향을 잡는다.
- 충돌 해결 후
    - 로컬 실행/테스트(최소 실행 확인)
    - 커밋 메시지 예: `chore: resolve merge conflict from main`

## README.md 작성 이력
- 2025/12/17: 최초 작성 (작성자: 문승환)
