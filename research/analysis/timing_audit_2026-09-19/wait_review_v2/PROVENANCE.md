# 기존 v2 측정의 대기 분해

사용자 제공 timing_audit_20260919_175307_zz6mlt 원자료를 timing_audit_v4/analyze.py로 재분석했다. 새 GPU 실행 결과가 아니다. v2의 프로파일러 시작 경계 및 수집 경고가 그대로 적용된다. 모든 내부 간격을 CUDA API 시점에 따라 분해하며 원인은 별도로 해석한다. 원본 trace는 read-only·immutable로 읽고 hash를 확인한다.
