# check 결과

수집 60/60, 판정 REVIEW

```json
{
  "gate": "REVIEW",
  "size": "check",
  "expected": 60,
  "recorded": 60,
  "missing": 0,
  "attempts": 60,
  "counts": {
    "PASS": 60
  },
  "timed_blocks": 420,
  "triplets_expected": 20,
  "triplets_complete": 20,
  "timing_and_sampled_clocks_pass": true,
  "max_relative_iqr": 0.0007367669107816498,
  "max_abs_drift": 0.0007166272956692349,
  "max_between_round_range": 0.006896831054969464,
  "between_round_flags": 0,
  "clock_flags": [],
  "counter_anomaly_blocks": 33,
  "known_counter_issue": "UNRESOLVED; retained separately from timing and sampled clock quality",
  "all_restorations_accepted": false,
  "results_digest": "7df10ff41b32312aaca648e86fe2f5ecb1dbf612b2289787a123c3dcc3c0b862",
  "task_wall_seconds_median": null,
  "reference_change_abs_max_pct": 2.1642654501723424,
  "full_sweep": "HOLD_PENDING_CHECK_REVIEW",
  "sampling_note": "Host-envelope samples are not continuous/effective kernel clocks."
}
```

allocate/prealloc/Graph는 triplets.csv의 별도 시간이다. 같은 조건의 round는 독립 학습 표본이 아니다.
check의 두 round는 별도 worker이며 동일 머신·인접 시간의 실행이다. full은 재시작 전까지 같은 worker다.
fresh_holdout 시간은 LOCKED CSV와 원자료에 보존한다. 모델 선택 전 성능값을 열어보지 않는다.
카운터 문제는 미해결이며 제한 시간 비율로 환산하지 않는다. 본수집 자동 전환은 없다.
