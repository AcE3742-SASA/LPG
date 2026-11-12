# core_dev 반환 자료 검산

사용자 반환 ZIP `core_dev_20261028_143530_VEnIY5_17236749.zip`은 새 Linear/FFN 개발 수집 960측정이다. 추가 GEMM 10,000형상 수집본과 다르다. 원본 SHA-256은 `5e32c060c2cd477c5675e1048b065b373df878bf9cbb20099c137cac86a3ec0f`다. 원본 ZIP과 해제 자료는 그대로 보존했다.

`archive_verification.json`은 CRC·안전한 경로·결과 파일 해시 2,255개·알려진 release 135개를 대조한 기록이다. `verification_standard.json`은 네 학습기 protocol을 동결한 뒤 표준 480측정의 시간·정확성·입력과 전체 센서를 검산한 기록이다. 이 단계는 explicit 시간값을 파싱하지 않았다.

`verification_all.json`은 RQ3의 `selection_frozen.json` 및 선택표 해시를 확인한 뒤 표준/분해 960측정과 160비교를 검산한 기록이다. 선택 동결 시각보다 늦은 별도 `explicit_diagnostics_access.json`을 저장하고 explicit 진단을 읽었다. 별도 `recomputed/`에서 원래 public summary와 explicit JSON·CSV를 정확히 재현했다. 모든 비교와 원측정은 PASS이며 원자료 해시는 전후 일치했다. 기존 GEMM 및 새 core_locked의 시간값은 열지 않았다.

검산 코드는 `verify_result.py`다. 기본 실행은 표준 자료만 허용한다. `--all-development`는 고정된 RQ3 선택을 요구하고 `recomputed/`가 이미 있으면 덮어쓰지 않는다. 새 GPU 실행과 반환 자료 안의 코드를 실행하는 방식 대신 보존된 로컬 release의 타이머·품질·센서 계산을 재사용한다.

학습 결과와 비용, 형상별 오차, 그림은 [레이어 개발 결과](../layer_transfer_fit_2026-10-28/레이어_개발실험_결과_2026-10-28.md)에 정리했다. `implementation_group_summary.csv`와 `implementation_ratios.png/.pdf`는 선택 동결 이후의 RQ2 개발 진단이다.
