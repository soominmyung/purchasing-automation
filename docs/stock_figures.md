# Stock figures: how code works them out

Korean translation: [below](#한국어).

Every figure the agents receive about an item is worked out in code (`services/item_grouping.py`)
from three values in the stock CSV: the snapshot date, `AvailableStock` and `WksToOOS`. No model
calculates them. These are the `items` fields of the analysis input (see `docs/sft_gemma4.md`).

## Snapshot date and run date

| Date | What it is | Used for |
|---|---|---|
| Snapshot date | The day the stock was exported from SAP: the `SnapshotDate` line of the CSV, or else a DDMMYY date in the file name | **Every calculation**: order and delivery dates, overdue, the 12 months of history retrieved |
| Run date | The day the pipeline runs | Two things only: a snapshot more than three days old is refused (`max_snapshot_age_days`), and the report states the snapshot's age ("Snapshot date: 2025-04-05 (1 day before this report)") |

The figures are as of the snapshot date because `WksToOOS` is true only on that day: counted from
the run date, a stock that lasts four weeks would be given dates as if it still had four weeks left
days later. The three-day limit keeps that difference small. The sample data fixes the run date
(`run_date` in `config.py`) so the 2025 sample still passes; in production it is the actual date.

## Inputs and constants

| | Value |
|---|---|
| `AvailableStock` | On hand + on order − committed to customers (SAP Business One `OITW`) |
| `WksToOOS` | Available stock ÷ average weekly sales over the previous four months |
| Lead time | 18 weeks: 16 weeks import + 2 weeks internal |
| Cover after delivery | 26 weeks of sales |

A row whose supplier, item code or item name is unusable, whose item code appears twice for the
supplier, or whose `AvailableStock` or `WksToOOS` is missing, negative, not a number or implausibly
large, is left out and listed at the top of the report; it never reaches the model.

## The fields

| Field | How it is worked out |
|---|---|
| `available_stock`, `wks_to_oos` | The CSV values, checked |
| `risk_level` | From `WksToOOS` against the lead time: **Critical** under 18 weeks (a stockout is certain even if ordered today); **High** under 26 (six months of cover); **Medium** under 35 (eight months); **Low** otherwise |
| `expected_stockout_weeks` (the gap) | 18 − `WksToOOS`, or 0 if negative: the weeks without stock even if ordered today |
| `recommended_latest_delivery_date` | Snapshot date + `WksToOOS` × 7 days (whole days, rounded down): the day the stock runs out |
| `recommended_latest_po_date` | Latest delivery date − 18 weeks. If that is before the snapshot date, the snapshot date, with the timing "overdue" |
| `recommended_latest_po_timing`, `recommended_latest_delivery_timing` | Weeks from the snapshot date to each date, as a label: immediately (≤ 0.5), within 1 week, within 2 weeks, within 2–4 weeks, within 4–8 weeks, then "within N weeks" |
| `suggested_quantity` | Weekly sales (`AvailableStock` ÷ `WksToOOS`) × 26, rounded up |
| `system_notes` | Fixed sentences for a Critical item, e.g. "Critical: even if ordered today, a stockout of about 14.1 weeks is expected (lead time 18 weeks)." |
| `data_issues` | Problems in the item's values (rows with unusable values are left out, so this is normally empty) |

## Examples (snapshot date 2026-03-02)

| | Low | High | Critical |
|---|---|---|---|
| `AvailableStock`, `WksToOOS` | 6,720, 56 | 2,200, 22 | 390, 3.9 |
| Risk level | Low (56 ≥ 35) | High (18 ≤ 22 < 26) | Critical (3.9 < 18) |
| Gap | 0 | 0 | 18 − 3.9 = 14.1 weeks |
| Latest delivery | + 392 days = 2027-03-29 | + 154 days = 2026-08-03 | + 27 days = 2026-03-29 |
| Latest order | − 126 days = 2026-11-23, within 38 weeks | − 126 days = 2026-03-30, within 2–4 weeks | − 126 days = 2025-11-23, past: 2026-03-02, overdue |
| Weekly sales, suggested quantity | 120 × 26 = 3,120 | 100 × 26 = 2,600 | 100 × 26 = 2,600 |

## What the dates assume

The latest delivery date is the day the stock runs out: an order that arrives by then avoids a
stockout, one a day later does not. The order date counts the lead time back from it, and the
suggested quantity covers the 26 weeks after it. This rests on four assumptions:

- **Sales continue at the rate of the last four months**, the rate `WksToOOS` is based on. A change
  in demand moves the date; finding one in the history and proposing a quantity change is the
  analysis agent's job.
- **Stock already on order is counted.** `AvailableStock` includes it, so the dates are for what
  must be ordered in addition.
- **No safety stock.** The deadline is the day stock reaches zero, with no margin; the risk of a
  longer lead time is handled by the analysis agent proposing a lead-time change from the history.
- **Part days are dropped** (3.9 weeks × 7 = 27.3 days becomes 27), which moves the date a day
  earlier, never later.

## After a lead-time change

When the analysis proposes a different lead time L for an item (a supplier's notice of longer
lead times, for instance), code works the figures out again (`services/documents.py`,
`with_lead_times`): the risk thresholds become L, L + 8 and L + 17 weeks, the gap L − `WksToOOS`,
and the latest order date the latest delivery date − L weeks (the snapshot date, marked overdue,
if that is on or before it). Weeks to stockout, the latest delivery date and the suggested
quantity do not change. The High example above at L = 22: still High (22 ≤ 22 < 30), gap 0, and
2026-08-03 − 22 weeks = 2026-03-02, the snapshot date itself: overdue.

A proposed quantity change replaces the suggested quantity with the proposed one.

---

# 한국어

## 재고 수치: 코드가 계산하는 방법

에이전트가 받는 품목 수치는 모두 코드(`services/item_grouping.py`)가 재고 CSV의 세 값, 즉 기준일,
`AvailableStock`, `WksToOOS`로 계산합니다. 모델은 이 수치를 계산하지 않습니다. 분석 입력의 `items`
필드가 이것입니다(`docs/sft_gemma4.md` 참조).

### 기준일과 실행일

| 날짜 | 무엇인가 | 쓰이는 곳 |
|---|---|---|
| 기준일 (스냅샷 날짜) | SAP에서 재고를 내보낸 날. CSV의 `SnapshotDate` 줄, 없으면 파일 이름의 DDMMYY 날짜 | **모든 계산**: 발주일과 입고일, 지연 여부, 조회할 이력 12개월 |
| 실행일 | 파이프라인을 돌린 날 | 두 가지뿐: 기준일이 3일보다 오래되면 거부(`max_snapshot_age_days`), 보고서에 기준일이 며칠 전인지 표시("Snapshot date: 2025-04-05 (1 day before this report)") |

수치를 기준일로 계산하는 이유는 `WksToOOS`가 그날에만 맞는 값이기 때문입니다. 실행일로 계산하면, 며칠이
지나 실제로는 덜 남은 재고를 여전히 4주 남은 것처럼 날짜를 잡게 됩니다. 3일 제한은 그 차이를 작게
유지합니다. 샘플 데이터는 2025년 샘플이 통과하도록 실행일을 고정해 두었고(`config.py`의 `run_date`),
운영에서는 실제 날짜입니다.

### 입력과 상수

| | 값 |
|---|---|
| `AvailableStock` | 보유 + 입고 예정 − 고객 출고 확정 (SAP Business One `OITW`) |
| `WksToOOS` | 가용재고 ÷ 직전 4개월 평균 주간 판매량 |
| 리드타임 | 18주: 수입 16주 + 내부 2주 |
| 입고 후 커버 기간 | 26주 판매분 |

공급업체, 품목코드, 품목명을 쓸 수 없는 행, 같은 공급업체에 품목코드가 두 번 나오는 행, `AvailableStock`이나
`WksToOOS`가 비었거나 음수이거나 숫자가 아니거나 비현실적으로 큰 행은 빼고 보고서 맨 위에 적습니다. 이런 행은
모델에 가지 않습니다.

### 필드

| 필드 | 계산 방법 |
|---|---|
| `available_stock`, `wks_to_oos` | CSV 값을 검사한 그대로 |
| `risk_level` | `WksToOOS`를 리드타임과 비교: 18주 미만 **Critical**(오늘 발주해도 품절 확정), 26주 미만 **High**(6개월분), 35주 미만 **Medium**(8개월분), 그 외 **Low** |
| `expected_stockout_weeks` (재고 공백) | 18 − `WksToOOS`, 음수면 0: 오늘 발주해도 재고가 없는 주 |
| `recommended_latest_delivery_date` | 기준일 + `WksToOOS` × 7일(소수점 아래 버림): 재고가 바닥나는 날 |
| `recommended_latest_po_date` | 최종 입고일 − 18주. 기준일보다 앞이면 기준일로 두고 시점을 "overdue"(지연)로 표시 |
| `recommended_latest_po_timing`, `recommended_latest_delivery_timing` | 기준일부터 각 날짜까지의 주 수를 표현으로: immediately(0.5주 이하), within 1 week, within 2 weeks, within 2–4 weeks, within 4–8 weeks, 그 이상은 "within N weeks" |
| `suggested_quantity` | 주간 판매량(`AvailableStock` ÷ `WksToOOS`) × 26, 올림 |
| `system_notes` | Critical 품목에 붙는 고정 문장. 예: "Critical: even if ordered today, a stockout of about 14.1 weeks is expected (lead time 18 weeks)." |
| `data_issues` | 품목 값의 문제(쓸 수 없는 값의 행은 빠지므로 보통 비어 있음) |

### 예시 (기준일 2026-03-02)

| | Low | High | Critical |
|---|---|---|---|
| `AvailableStock`, `WksToOOS` | 6,720, 56 | 2,200, 22 | 390, 3.9 |
| 위험도 | Low (56 ≥ 35) | High (18 ≤ 22 < 26) | Critical (3.9 < 18) |
| 재고 공백 | 0 | 0 | 18 − 3.9 = 14.1주 |
| 최종 입고일 | + 392일 = 2027-03-29 | + 154일 = 2026-08-03 | + 27일 = 2026-03-29 |
| 최종 발주일 | − 126일 = 2026-11-23, within 38 weeks | − 126일 = 2026-03-30, within 2–4 weeks | − 126일 = 2025-11-23, 이미 지남: 2026-03-02, overdue |
| 주간 판매량, 권장 수량 | 120 × 26 = 3,120 | 100 × 26 = 2,600 | 100 × 26 = 2,600 |

### 날짜 계산의 전제

최종 입고일은 재고가 바닥나는 날입니다. 그날까지 들어오면 품절이 없고, 하루라도 늦으면 품절입니다. 발주일은
이 날짜에서 리드타임을 거꾸로 세어 구하고, 권장 수량은 이 날짜 이후 26주를 버티는 양입니다. 여기에는 네 가지
전제가 있습니다.

- **판매가 직전 4개월의 속도로 이어집니다.** `WksToOOS`가 그 속도로 계산된 값입니다. 수요가 바뀌면 날짜도
  바뀝니다. 이력에서 수요 변화를 찾아 수량 변경을 제안하는 것은 분석 에이전트의 일입니다.
- **이미 발주한 물량이 포함됩니다.** `AvailableStock`에 들어 있으므로, 날짜는 추가로 주문해야 할 물량에 대한
  것입니다.
- **안전재고가 없습니다.** 재고가 0이 되는 날을 여유 없이 기한으로 잡습니다. 리드타임이 늘어날 위험은 분석
  에이전트가 이력을 보고 리드타임 변경을 제안하는 방식으로 다룹니다.
- **일 단위 미만은 버립니다**(3.9주 × 7 = 27.3일 → 27일). 날짜가 하루 앞당겨질 수는 있어도 늦춰지지는
  않습니다.

### 리드타임 변경 후

분석이 어떤 품목에 다른 리드타임 L을 제안하면(예: 공급업체의 리드타임 연장 통지), 코드가 수치를 다시
계산합니다(`services/documents.py`, `with_lead_times`). 위험도 경계는 L, L + 8, L + 17주가 되고, 재고 공백은
L − `WksToOOS`, 최종 발주일은 최종 입고일 − L주입니다(기준일 이전이거나 같으면 기준일로 두고 지연 표시).
소진까지 남은 주, 최종 입고일, 권장 수량은 바뀌지 않습니다. 위 High 예시에 L = 22를 적용하면 여전히
High(22 ≤ 22 < 30), 공백 0, 2026-08-03 − 22주 = 2026-03-02로 기준일 당일이므로 지연입니다.

수량 변경을 제안하면 권장 수량이 제안 수량으로 바뀝니다.
