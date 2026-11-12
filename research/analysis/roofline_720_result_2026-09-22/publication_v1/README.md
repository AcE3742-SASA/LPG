# 720 MHz Roofline 논문 삽입용 그림

검산한 720 MHz 결과의 수치는 유지하고, 그림의 최종 크기와 글자·선·캡션 배치를 수정했다. 기존 설명용 그림과 원자료는 상위 폴더에 보존했다.

## 파일 선택

| 용도 | 단폭 | 전체폭 |
|---|---|---|
| 그림 1: Roofline | figure1_roofline_89mm.pdf, 89 × 80 mm | figure1_roofline_182mm.pdf, 182 × 92 mm |
| 그림 2: 크기별 결과 | figure2_scaling_89mm.pdf, 89 × 180 mm, 세로 배치 | figure2_scaling_182mm.pdf, 182 × 80 mm, 가로 배치 |

각 그림의 SVG와 600 dpi PNG도 포함했다. 기본 글자는 최종 크기에서 10 pt이며 로그 눈금의 위첨자만 7 pt다. 그림 1은 단폭 PDF, 그림 2는 전체폭 PDF 사용을 권장한다. SVG는 편집용이며 다른 프로그램에서 열 때 글꼴이 바뀔 수 있으므로 삽입에는 글꼴이 포함된 PDF를 우선 사용한다. PDF를 지정 폭 그대로 배치하고 과도하게 축소하지 않는다.

큰 제목·장비 설명·하단 주석은 외부 캡션으로 옮겼다. `captions_ko.txt`와 `captions_en.txt`는 붙여넣을 국문·영문 원문이고, `captions.md`는 두 언어를 함께 확인하는 파일이다. 그림 번호는 원고 순서에 맞춰 바꾼다. 측정 정의와 범위는 캡션에 유지한다.

## 측정값과 그림의 의미

| 항목 | 값 |
|---|---:|
| FP32 SIMT | 6.14 TFLOP/s |
| FP16 입출력, FP32 누산 Tensor Core | 35.02 TFLOP/s |
| BabelStream Dot 논리적 대역폭 | 262.3 GB/s |
| FP32 교차점 | 23.4 FLOP/byte |
| Tensor Core 교차점 | 133.5 FLOP/byte |

원형 표시는 계산한 교차점이며 응용 프로그램 실측 점이 아니다. 대역폭 점선은 선형 대역폭 항 B×I이고 수평 연산 천장을 적용하기 전의 선이다. 본측정은 조건별 같은 세션 내 별도 프로세스 5회이며 크기별 오차 막대는 관측 최소–최대 범위다. 전체 정확성·부하 클록·반복 검사와 누적 전력 카운터 해석의 한계는 상위 폴더 `report.md`에 보존했다. 그림 편집으로 새로운 측정 근거나 독립 세션 재현성을 추가한 것은 아니다.

## 재현 및 점검

```bash
python3 -B plot_publication.py
```

기존 matplotlib와 numpy를 사용한다. 새 라이브러리나 GPU 실행은 필요하지 않다. `source_hashes.json`으로 입력 3파일을 확인하고, 검산한 원값과 경로별 곡선의 일치, 최종 크기, 글자 최소 10 pt, 페이지 밖 잘림을 검사한다. `layout_checks.json`은 그 검사 결과다. `check_pdf.py`는 기존 pypdf와 reportlab이 있는 Python에서 PDF 점검과 A4 교정본 생성을 재현한다. PDF의 페이지 크기·글꼴 포함 여부는 `publication_validation.json`에 기록한다.

`proof_actual_size.pdf`는 A4 용지에 실제 폭으로 배치한 흑백 교정본이다. 인쇄할 때 페이지 맞춤을 끄고 100%로 출력한다. 교정용 페이지 문구는 제출 그림에 포함되지 않는다.

폭은 IEEE의 일반적인 한 단·두 단 크기를 참고해 89 mm와 182 mm로 정했다. 특정 투고처의 모든 규정을 확인한 최종 접수 파일이라는 뜻은 아니며, 확정된 템플릿에서는 배치와 번호를 맞춘다. 출처: [IEEE Resolution and Size](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/).
