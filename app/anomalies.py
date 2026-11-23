import math
from collections import defaultdict
from datetime import date
import numpy as np
from app.schemas import AnomalyPointRequest, AnomalyResponse, ExpectedRange, FlaggedDay, SkippedSeries


def _poisson_upper_threshold(lam: float, tail_prob: float = 0.005) -> int:
    lam = max(lam, 0.5)
    cdf = 0.0
    k = 0
    term = math.exp(-lam)
    while k < 200:
        cdf += term
        if 1.0 - cdf <= tail_prob:
            return k
        k += 1
        term = term * lam / k
    return k


class AnomalyEngine:
    def __init__(self, window_days: int = 28, min_history: int = 14, z_threshold: float = 3.2, min_absolute_jump: int = 3) -> None:
        self.window_days = window_days
        self.min_history = min_history
        self.z_threshold = z_threshold
        self.min_absolute_jump = min_absolute_jump

    def detect(self, points: list[AnomalyPointRequest]) -> AnomalyResponse:
        grouped: dict[tuple[str, str], list[tuple[date, int]]] = defaultdict(list)
        for pt in points:
            grouped[(pt.departmentId, pt.metric)].append((pt.date, pt.count))
        flagged: list[FlaggedDay] = []
        skipped: list[SkippedSeries] = []
        for (dept_id, metric), series in sorted(grouped.items()):
            series.sort(key=lambda item: item[0])
            if len(series) <= self.min_history:
                skipped.append(SkippedSeries(departmentId=dept_id, metric=metric, reason=f"insufficient_history_under_{self.min_history}_days"))
                continue
            for idx in range(self.min_history, len(series)):
                curr_date, curr_count = series[idx]
                lookback = series[max(0, idx - self.window_days) : idx]
                is_sunday = curr_date.weekday() == 6
                if is_sunday:
                    sunday_lookback = series[max(0, idx - 56) : idx]
                    bucket = [cnt for dt, cnt in sunday_lookback if dt.weekday() == 6]
                    min_bucket = 3
                else:
                    bucket = [cnt for dt, cnt in lookback if dt.weekday() != 6]
                    min_bucket = 7
                if len(bucket) >= min_bucket:
                    vals = np.array(bucket, dtype=float)
                    label = "Sunday" if is_sunday else "weekday"
                else:
                    vals = np.array([cnt for _, cnt in lookback], dtype=float)
                    label = "rolling"
                median = float(np.median(vals))
                mad = float(np.median(np.abs(vals - median)))
                sigma = 1.4826 * max(mad, 1.0)
                low = round(max(0.0, median - self.z_threshold * sigma), 1)
                high = round(median + self.z_threshold * sigma, 1)
                is_spike = curr_count > high and curr_count >= (median + self.min_absolute_jump)
                if is_spike and median < 5.0:
                    is_spike = curr_count > _poisson_upper_threshold(float(np.mean(vals)))
                if is_spike:
                    ratio_str = f"{curr_count / median:.1f}x" if median > 0 else f"+{curr_count}"
                    reason = f"Count of {curr_count} is {ratio_str} the {len(vals)}-day {label} baseline median ({median:.1f}), exceeding expected upper bound of {high:.1f}."
                    flagged.append(
                        FlaggedDay(
                            date=curr_date,
                            departmentId=dept_id,
                            metric=metric,
                            actualCount=curr_count,
                            expectedRange=ExpectedRange(low=low, high=high),
                            reason=reason,
                        )
                    )
        flagged.sort(key=lambda f: (f.date, f.departmentId, f.metric))
        return AnomalyResponse(flaggedDays=flagged, skippedSeries=skipped)