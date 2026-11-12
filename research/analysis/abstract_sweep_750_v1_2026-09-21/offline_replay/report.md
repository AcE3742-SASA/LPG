# full 결과

수집 36/3600, 판정 INCOMPLETE

```json
{
  "gate": "INCOMPLETE",
  "size": "full",
  "expected": 3600,
  "recorded": 36,
  "missing": 3564,
  "attempts": 36,
  "counts": {
    "PASS": 36
  },
  "timed_blocks": 252,
  "triplets_expected": 1200,
  "triplets_complete": 12,
  "timing_and_sampled_clocks_pass": false,
  "max_relative_iqr": 0.00037353771873995683,
  "max_abs_drift": 0.000460886608767596,
  "max_between_round_range": 0.007143648816737423,
  "between_round_flags": 0,
  "round_group_sizes": {
    "2": 18
  },
  "round_groups_complete": 0,
  "round_groups_expected": 1200,
  "clock_flags": [],
  "counter_anomaly_blocks": 13,
  "known_counter_issue": "UNRESOLVED; retained separately from timing and sampled clock quality",
  "all_restorations_accepted": false,
  "results_digest": "21f4aa6fe3cc95bf4b25df31c106e164ee3eee06ff4f83ac88f6fdae64226c26",
  "task_wall_seconds_median": null,
  "reference_change_abs_max_pct": null,
  "full_sweep": "COLLECTION_ONLY",
  "sampling_note": "Host-envelope samples are not continuous/effective kernel clocks."
}
```

allocate/prealloc/Graph는 triplets.csv의 별도 시간이다. 같은 조건의 round는 독립 학습 표본이 아니다.
check의 두 round는 별도 worker이며 동일 머신·인접 시간의 실행이다. full은 재시작 전까지 같은 worker다.
fresh_holdout 시간은 LOCKED CSV와 원자료에 보존한다. 모델 선택 전 성능값을 열어보지 않는다.
카운터 문제는 미해결이며 제한 시간 비율로 환산하지 않는다. 본수집 자동 전환은 없다.
