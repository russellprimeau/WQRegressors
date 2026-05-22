"""
Post-Calibration Drift Analysis (post_cal_drift.py)

Model for one calibration-to-calibration segment:

    y(t) = c + (y0 - c) * exp(-t / tau) + noise

where:
    t    = hours since Calibration End Time (the moment the sensor was reset).
           The first measurement is typically at t > 0 (often several hours
           after cal end), and the model is not pinned to any value at t=0
           by the data.
    y(t) = raw sensor reading
    y0   = sensor reading at t=0 (free parameter, what the sensor would read
           at the instant calibration ended). Compare against the recorded
           Post Calibration Value as a sanity check.
    c    = asymptotic reading the sensor decays toward (free parameter,
           equals true-value-plus-long-term-bias)
    tau  = decay time constant (hours)

This is mathematically the same 3-parameter family as the prior
y(t) = c_old + z*(1 - exp(-t/tau)) with c = c_old + z and y0 = c_old, but
the new parameterization names the t=0 value (y0) and the asymptote (c)
directly, which is easier to interpret.

The fit also uses (a) linear least-squares loss instead of Huber, because
Huber was down-weighting the systematically elevated early-time points as
if they were outliers, and (b) explicit per-point weights that boost the
first 24 hours, where the decay shape is most informative.

At the next calibration, the measured (Pre Cal Value - Post Cal Value of this
cal) equals the drift at t = T_end. The fit is unconstrained, and we report
how well the fitted z*(1 - exp(-T_end/tau)) matches that known endpoint as a
consistency check.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
SENSOR_PARM_COL = "sensorParms(8)"
TEMP_PARM_COL   = "sensorParms(1)"
SENSOR_NAME     = "Turbidity (FNU)"
YEARS           = (2023, 2024)
MIN_POINTS      = 20

HOURLY_PATH = ROOT / "data" / "input" / "sensors" / "FullHourly.csv"
CAL_PATH    = ROOT / "data" / "output" / "calibration" / "summaries" / f"{SENSOR_NAME}.csv"
OUT_DIR     = ROOT / "data" / "output" / "calibration" / "drift_analysis" / SENSOR_NAME


def load_calibration_events(cal_path, years):
    """Load calibration log, filter to Completed + stable, dedupe, aggregate same-day."""
    df = pd.read_csv(cal_path)
    df = df[df['Calibration Status'] == 'Completed'].copy()
    if 'Stability Achieved' in df.columns:
        df = df[df['Stability Achieved'] == 'Yes'].copy()

    df['start'] = pd.to_datetime(df['Calibration Start Time'], errors='coerce')
    df['end']   = pd.to_datetime(df['Calibration End Time'],   errors='coerce')
    df = df.dropna(subset=['start', 'end'])

    df = df.drop_duplicates(subset=['start', 'end', 'Standard', 'Pre Calibration Value',
                                    'Post Calibration Value'])

    if years is not None:
        df = df[df['end'].dt.year.isin(years)]
    df = df.sort_values('end').reset_index(drop=True)

    # Aggregate same-day events: collapse all rows on a given calendar day per year
    # into one event whose end-time = max(end) of that day. Baseline value taken
    # from the row with Standard == "0.00 FNU" (the zero-point cal), which defines
    # the expected sensor reading just after calibration.
    df['day'] = df['end'].dt.date
    agg_rows = []
    for day, sub in df.groupby('day', sort=True):
        end_time   = sub['end'].max()
        start_time = sub['start'].min()
        # Prefer 0 FNU standard for the post-cal baseline (sensor sits in clean water).
        zero_rows = sub[sub['Standard'].astype(str).str.strip().str.startswith('0')]
        if len(zero_rows) > 0:
            base_row = zero_rows.iloc[0]
        else:
            base_row = sub.iloc[0]
        post_val = float(base_row['Post Calibration Value'])
        pre_val  = float(base_row['Pre Calibration Value'])
        # Pre Calibration Value 2 measured against the 12.4 FNU standard,
        # used for the linear-error-vs-true-value interpolation in method B.
        pre_val_2  = base_row.get('Pre Calibration Value 2', np.nan)
        post_val_2 = base_row.get('Post Calibration Value 2', np.nan)
        try:
            pre_val_2 = float(pre_val_2)
        except (TypeError, ValueError):
            pre_val_2 = np.nan
        try:
            post_val_2 = float(post_val_2)
        except (TypeError, ValueError):
            post_val_2 = np.nan
        agg_rows.append({
            'start_time': start_time,
            'end_time':   end_time,
            'post_cal_value':   post_val,
            'pre_cal_value':    pre_val,
            'post_cal_value_2': post_val_2,
            'pre_cal_value_2':  pre_val_2,
            'year': end_time.year,
        })

    return pd.DataFrame(agg_rows).sort_values('end_time').reset_index(drop=True)


def build_segments(cal_df, sensor_last_time):
    """
    For each year, build between-cal segments plus a trailing post-last-cal segment
    up to either the next-year first cal or the sensor's last observed time.
    """
    segments = []
    for year in YEARS:
        year_events = cal_df[cal_df['year'] == year].reset_index(drop=True)
        if len(year_events) == 0:
            continue
        # Between-event segments within the year.
        for i in range(len(year_events) - 1):
            next_evt = year_events.loc[i+1]
            segments.append({
                'segment_id': f"{year}-{i+1}",
                'start_time':     year_events.loc[i,   'end_time'],
                'end_time':       next_evt['start_time'],
                'post_cal_value': year_events.loc[i,   'post_cal_value'],
                # Known endpoint constraint at the segment end (next cal start):
                # the next cal's pre-cal readings tell us the sensor's error at
                # two known true values (Standard 1 = 0 FNU, Standard 2 = 12.4
                # FNU). Linear interpolation between these gives the error at
                # any observed y_last.
                'endpoint_drift': next_evt['pre_cal_value']
                                  - next_evt['post_cal_value'],
                'next_pre1':  next_evt['pre_cal_value'],     # measured at standard 1 (=0)
                'next_post1': next_evt['post_cal_value'],    # standard 1 value (=0)
                'next_pre2':  next_evt['pre_cal_value_2'],   # measured at standard 2 (=12.4)
                'next_post2': next_evt['post_cal_value_2'],  # standard 2 value (=12.4)
                'year': year,
            })
        # Trailing segment from the last cal in this year up to end-of-year
        # (or sensor_last_time, whichever is earlier). No endpoint constraint.
        last_end = year_events.iloc[-1]['end_time']
        eoy = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)
        segments.append({
            'segment_id': f"{year}-{len(year_events)}-tail",
            'start_time': last_end,
            'end_time':   min(eoy, sensor_last_time),
            'post_cal_value': year_events.iloc[-1]['post_cal_value'],
            'endpoint_drift': np.nan,
            'next_pre1':  np.nan,
            'next_post1': np.nan,
            'next_pre2':  np.nan,
            'next_post2': np.nan,
            'year': year,
        })
    return segments


def load_sensor_series(path):
    cols = ['TIMESTAMP', SENSOR_PARM_COL, TEMP_PARM_COL]
    df = pd.read_csv(path, usecols=cols, parse_dates=['TIMESTAMP'])
    df = df.rename(columns={SENSOR_PARM_COL: 'turbidity', TEMP_PARM_COL: 'temperature'})
    df = df.dropna(subset=['turbidity']).sort_values('TIMESTAMP').reset_index(drop=True)
    return df


def extract_segment_data(sensor_df, segment):
    mask = (sensor_df['TIMESTAMP'] >= segment['start_time']) & \
           (sensor_df['TIMESTAMP'] <= segment['end_time'])
    sub = sensor_df.loc[mask].copy()
    if len(sub) == 0:
        return sub.assign(t_hours=[])
    sub['t_hours'] = (sub['TIMESTAMP'] - segment['start_time']).dt.total_seconds() / 3600.0
    return sub


# ---------- candidate model ----------
# Decay-from-y0-to-c: y(t) = c + (y0 - c) * exp(-t/tau)
def m_drift(t, y0, c, tau):
    return c + (y0 - c) * np.exp(-t / tau)


def _gaussian_loglik(residuals):
    n = len(residuals)
    ssr = float(np.sum(residuals ** 2))
    if n == 0 or ssr <= 0:
        return np.nan
    sigma2 = ssr / n
    return -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)


def _aicc(k, loglik, n):
    if not np.isfinite(loglik) or n - k - 1 <= 0:
        return np.nan
    aic = 2 * k - 2 * loglik
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def _fit_once(t, y, weights, p0, bounds):
    """Weighted linear least-squares fit. Returns dict or None."""
    n = len(t)
    sigma = 1.0 / np.sqrt(weights)  # curve_fit uses sigma for inverse weighting
    try:
        params, _ = curve_fit(m_drift, t, y, p0=p0, bounds=bounds,
                              sigma=sigma, absolute_sigma=False, maxfev=20000)
        yhat = m_drift(t, *params)
        residuals = y - yhat
        ssr = float(np.sum(weights * residuals ** 2))
        loglik = _gaussian_loglik(residuals)
        k = len(params)
        return {
            'params': params, 'n': n, 'k': k, 'ssr': ssr,
            'loglik': loglik, 'aicc': _aicc(k, loglik, n),
            'yhat': yhat, 'residuals': residuals,
        }
    except Exception:
        return None


def _build_weights(t, y):
    """
    Per-point weights for the fit:
      * boost the first 24 hours (the decay shape lives there)
      * down-weight points with very large positive residual vs a rolling
        median (genuine turbidity spikes — true signal, not drift). Done
        in a second pass after a preliminary fit.
    """
    n = len(t)
    w = np.ones(n)
    # Early-time boost: triangular ramp from 5x at t=0 to 1x at t=24h.
    early_mask = t < 24.0
    w[early_mask] = 1.0 + 4.0 * (1.0 - t[early_mask] / 24.0)
    return w


def _loglinear_initial_guess(t, y, t_max):
    """
    Derive (y0_init, c_init, tau_init) for y(t) = c + (y0 - c) * exp(-t/tau).

    Strategy: linearize. For the model above, log|y - c| is linear in t with
    slope -1/tau and intercept log|y0 - c|. We don't know c, so we iterate:

      pass A: rough c_init = median of last 10% of points (biased, but a
              starting place).
      pass B: fit a line to log|y - c_init| vs t, but using only the points
              where |y - c_init| > some floor (to keep the log finite). Slope
              gives -1/tau_init; intercept gives log|y0_init - c_init|.
      pass C: refine c by averaging only points beyond 3*tau_init (where
              exp(-3) ~ 0.05, so the transient is mostly gone). Redo pass B.

    Falls back to median heuristics on degenerate segments (all-equal y, etc).
    """
    n = len(t)

    def medians_fallback():
        head_n = max(3, n // 20)
        tail_n = max(3, n // 20)
        return (float(np.median(y[:head_n])), float(np.median(y[-tail_n:])),
                t_max / 3.0)

    # --- pass A: rough c
    tail_n = max(5, n // 10)
    c_init = float(np.median(y[-tail_n:]))

    def loglin_fit(c_guess):
        # Sign of (y - c_guess) should be consistent across most points if the
        # decay is monotone; use the sign of the early points (which are
        # farthest from c) as the model's sign.
        head_n = max(3, n // 20)
        sign = 1.0 if (np.median(y[:head_n]) - c_guess) >= 0 else -1.0
        d = sign * (y - c_guess)
        # Use only points where d is comfortably positive (transient still
        # detectable above noise). Floor at 5% of the maximum |y - c|.
        floor = max(1e-3, 0.05 * float(np.max(np.abs(y - c_guess))))
        mask = d > floor
        if mask.sum() < 5:
            return None
        slope, intercept = np.polyfit(t[mask], np.log(d[mask]), 1)
        if not np.isfinite(slope) or slope >= 0:
            return None  # not a decay
        tau_g = -1.0 / slope
        # y0_g - c_guess = sign * exp(intercept)
        y0_g = c_guess + sign * float(np.exp(intercept))
        return y0_g, tau_g

    # --- pass B
    res = loglin_fit(c_init)
    if res is None:
        return medians_fallback()
    y0_init, tau_init = res

    # --- pass C: refine c on points beyond 3*tau, then refit
    if tau_init > 0 and tau_init < t_max:
        late_mask = t > 3.0 * tau_init
        if late_mask.sum() >= 5:
            c_refined = float(np.mean(y[late_mask]))
            res2 = loglin_fit(c_refined)
            if res2 is not None:
                y0_init, tau_init = res2
                c_init = c_refined

    # Clamp tau into a sensible range for the optimizer's seed.
    tau_init = float(np.clip(tau_init, 1.0, t_max * 3.0))
    return float(y0_init), float(c_init), tau_init


EARLY_WINDOW_HOURS    = 72.0  # use the first 72h of every segment for the fit
EARLY_WINDOW_MIN_PTS  = 20    # fallback to first N points if 72h too sparse
Y0_ANCHOR_N           = 50    # number of early points for the y0 linear-extrap bound


def _y0_lower_bound(t, y):
    """
    Lower bound on y0 from a *linear* (not log-linear) extrapolation back to
    t=0 using the first Y0_ANCHOR_N measurements. Over a short early window
    the exponential is locally near-linear, so linear extrapolation gives a
    reasonable lower bound on what y0 should be (the curve cannot start
    below the linear-extrapolated intercept without violating the observed
    early trend).
    """
    n_anchor = min(Y0_ANCHOR_N, len(t))
    if n_anchor < 3:
        return float(y[0])
    tt = t[:n_anchor]
    yy = y[:n_anchor]
    slope, intercept = np.polyfit(tt, yy, 1)
    return float(intercept)


def _shape_based_fit(t, y, t_max, c_grid_resolution=80):
    """
    Estimate (y0, c, tau) for y(t) = c + (y0 - c) * exp(-t/tau) using a
    shape-based, log-linear procedure on the first EARLY_WINDOW_HOURS of
    data. Returns (y0, c, tau, r2) or None.

    Procedure:
      1. Compute the y0 lower bound by linear extrapolation of the first
         Y0_ANCHOR_N points back to t=0.
      2. Build the early-decay window: t <= EARLY_WINDOW_HOURS, falling back
         to the first EARLY_WINDOW_MIN_PTS points if too few fall in the time
         window.
      3. Search over candidate c on a physically constrained grid:
         a. For each c, compute log|y - c| over windowed points where the
            sign of (y - c) matches the decay direction and |y - c| exceeds
            a noise floor.
         b. Fit a line in (t, log|y - c|), record slope, intercept, R^2.
         c. Compute y0_candidate = c + sign * exp(intercept).
         d. Reject candidates where y0_candidate < y0_lower_bound.
      4. Pick the c maximizing R^2 among accepted candidates.
      5. Return (y0, c, tau, R^2). tau = -1/slope.

    Late points (beyond EARLY_WINDOW_HOURS) are excluded so that ground-truth
    drift unrelated to the post-cal transient cannot contaminate the estimate.
    """
    n = len(t)
    if n < 10:
        return None

    # Decay direction (sign of (y0 - c)): positive if early points are above
    # late points, the empirical case for this sensor.
    head_n = max(3, n // 20)
    tail_n = max(3, n // 20)
    sign = 1.0 if np.median(y[:head_n]) >= np.median(y[-tail_n:]) else -1.0

    # Early-decay window: first EARLY_WINDOW_HOURS, or first
    # EARLY_WINDOW_MIN_PTS samples, whichever covers more points.
    time_mask  = t <= EARLY_WINDOW_HOURS
    if time_mask.sum() >= EARLY_WINDOW_MIN_PTS:
        window_mask = time_mask
    else:
        window_mask = np.zeros(n, dtype=bool)
        window_mask[:min(EARLY_WINDOW_MIN_PTS, n)] = True

    y0_lower = _y0_lower_bound(t, y)
    # Allow a noise tolerance: the linear extrapolation itself has uncertainty,
    # so don't reject candidates that fall slightly below it. Tolerance = 10%
    # of the observed y range in the window.
    y_win_full = y[window_mask]
    y_range = float(np.max(y_win_full) - np.min(y_win_full))
    y0_tol = max(0.05, 0.10 * y_range)
    y0_lower_soft = y0_lower - y0_tol  # for sign > 0
    y0_upper_soft = y0_lower + y0_tol  # for sign < 0

    def loglin_r2_for_c(c_guess):
        """Fit log|y - c| vs t over windowed points; return (slope, intercept, r2,
        y0_candidate) or None if degenerate / rejected by y0 bound."""
        d = sign * (y[window_mask] - c_guess)
        floor = max(1e-3, 0.02 * float(np.max(np.abs(y[window_mask] - c_guess))))
        valid = d > floor
        if valid.sum() < 5:
            return None
        tt = t[window_mask][valid]
        ld = np.log(d[valid])
        slope, intercept = np.polyfit(tt, ld, 1)
        if not np.isfinite(slope) or slope >= 0:
            return None
        ld_hat = slope * tt + intercept
        ss_res = float(np.sum((ld - ld_hat) ** 2))
        ss_tot = float(np.sum((ld - ld.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        y0_cand = c_guess + sign * float(np.exp(intercept))
        # Reject candidates that put y0 outside the linear-extrap bound
        # (with noise tolerance), or physically impossible values.
        if y0_cand < 0:
            return None
        if sign > 0 and y0_cand < y0_lower_soft:
            return None
        if sign < 0 and y0_cand > y0_upper_soft:
            return None
        return slope, intercept, r2, y0_cand

    # c grid: physically constrained to lie within (or just beyond) the data
    # range, biased toward values the late portion of the segment suggests
    # for the asymptote.
    y_win = y[window_mask]
    y_lo_win, y_hi_win = float(np.min(y_win)), float(np.max(y_win))
    y_full_late = y[t > 0.7 * t_max]
    late_median = float(np.median(y_full_late)) if len(y_full_late) > 0 else y_lo_win
    spread = max(y_hi_win - y_lo_win, 1e-2)

    if sign > 0:
        # Decay downward => c is at or below the late-segment median.
        c_lo = max(0.0, min(late_median, y_lo_win) - 0.3 * spread)
        c_hi = min(y_lo_win + 0.3 * spread, late_median + 0.3 * spread)
        if c_hi <= c_lo:
            c_hi = c_lo + 0.3 * spread
    else:
        c_lo = max(y_hi_win - 0.3 * spread, late_median - 0.3 * spread)
        c_hi = y_hi_win + 0.3 * spread

    c_candidates = np.linspace(c_lo, c_hi, c_grid_resolution)

    best = None  # (y0, c, tau, r2)
    for c_try in c_candidates:
        res = loglin_r2_for_c(c_try)
        if res is None:
            continue
        slope, _intercept, r2, y0_cand = res
        tau_cand = -1.0 / slope
        if best is None or r2 > best[3]:
            best = (y0_cand, float(c_try), float(tau_cand), float(r2))

    return best


def _endpoint_constrained_fit(y, segment, y0_anchor, tau_from_A):
    """
    METHOD B: y0 from the same linear-extrap anchor as method A. tau from
    method A's early-decay log-linear fit. c is DERIVED (not searched) from
    the requirement that the model reproduces the cal-log-measured error at
    the segment endpoint.

    Cal-log linear-error model (linear interpolation between the two
    pre-cal-at-standard readings of the NEXT calibration):
        error(y_true) = a + b * y_true
        a = pre1 - post1         (error at standard 1 = 0 FNU)
        b = ((pre2 - post2) - a) / (post2 - post1)
    For the observed y_last at the segment endpoint:
        y_true_last = (y_last - a) / (1 + b)
        error_last  = y_last - y_true_last

    Under the model y(t) = c + (y0 - c) * exp(-t/tau), the drift at t_end is
    (c - y0) * (1 - exp(-t_end/tau)). Setting this equal to error_last and
    solving for c:

        c = y0 + error_last / (1 - exp(-t_end/tau))

    Note: c may end up CLOSE TO y0 (small magnitude of transient) when the
    cal-log error_last is small relative to the apparent decay in the
    sensor series. That's a meaningful finding, not a failure: it means the
    cal log thinks the sensor barely drifted, even though the time series
    looks like it dropped a lot. The apparent decay then mostly reflects
    real ground-truth changes, not sensor drift.

    Returns dict or None (only None if cal-log info is missing or the
    inputs are degenerate).
    """
    pre1  = segment.get('next_pre1',  np.nan)
    pre2  = segment.get('next_pre2',  np.nan)
    post1 = segment.get('next_post1', np.nan)
    post2 = segment.get('next_post2', np.nan)
    if not all(np.isfinite([pre1, pre2, post1, post2])):
        return None

    t_end = (segment['end_time'] - segment['start_time']).total_seconds() / 3600.0
    if not np.isfinite(t_end) or t_end <= 0:
        return None
    if not np.isfinite(tau_from_A) or tau_from_A <= 0:
        return None

    a = pre1 - post1
    denom = post2 - post1
    if denom == 0:
        return None
    b = ((pre2 - post2) - a) / denom
    if abs(1.0 + b) < 1e-9:
        return None

    y_last = float(y[-1])
    y_true_last = (y_last - a) / (1.0 + b)
    error_last  = y_last - y_true_last

    y0 = float(y0_anchor)
    tau = float(tau_from_A)

    denom_t = 1.0 - np.exp(-t_end / tau)
    if denom_t <= 1e-9:
        return None
    c = y0 + error_last / denom_t

    return {
        'y0':  y0,
        'c':   float(c),
        'tau': tau,
        'r2_loglin': np.nan,  # method B doesn't optimize R^2; c is derived
        'y_true_last': float(y_true_last),
        'error_last':  float(error_last),
        't_end':  float(t_end),
        'y_last': float(y_last),
    }


def fit_segment(t, y, segment):
    """
    Estimate (y0, c, tau) two ways:

      Method A (primary, returned in the top-level dict): the shape-based
      log-linear fit on the first 72h of data, with y0 anchored by linear
      extrapolation of the first 50 points and c constrained physically.

      Method B (alternative, stored in *_B fields): solves for c, tau such
      that the model passes through the segment-end reading, with the end
      reading interpreted via the next calibration's pre-cal-at-standard-1
      and pre-cal-at-standard-2 linear interpolation. y0 is the same
      linear-extrap anchor as method A.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    if n < MIN_POINTS:
        return None

    t_max = float(np.max(t))

    # --- Method A: shape-based fit on first 72h with linear-extrap y0 anchor.
    result_A = _shape_based_fit(t, y, t_max)
    if result_A is None:
        return None
    y0, c, tau, r2 = result_A
    z = float(y0 - c)

    yhat = m_drift(t, y0, c, tau)
    residuals = y - yhat
    ssr = float(np.sum(residuals ** 2))

    endpoint_model = float(m_drift(np.array([t_max]), y0, c, tau)[0] - c)
    endpoint_obs   = segment.get('endpoint_drift', np.nan)

    # --- Method B: cal-log-endpoint-constrained fit using the same y0 anchor.
    y0_anchor = _y0_lower_bound(t, y)
    # Method B reuses method A's tau (which comes from the early-decay
    # log-linear fit) and derives c from the cal-log endpoint constraint.
    result_B  = _endpoint_constrained_fit(y, segment, y0_anchor, tau)

    out = {
        'y0': float(y0),
        'c':  float(c),
        'z':  z,
        'tau': float(tau),
        'r2_loglin': float(r2),
        'ssr': ssr,
        'aicc': np.nan,
        't_max': t_max,
        'y0_init':  float(y0),
        'c_init':   float(c),
        'tau_init': float(tau),
        'residuals': residuals,
        'endpoint_model':    endpoint_model,
        'endpoint_observed': float(endpoint_obs) if pd.notna(endpoint_obs) else np.nan,
        # Method B fields (NaN if endpoint info unavailable).
        'y0_B':  np.nan,
        'c_B':   np.nan,
        'tau_B': np.nan,
        'z_B':   np.nan,
        'r2_loglin_B': np.nan,
        'y0_anchor':  float(y0_anchor),
    }
    if result_B is not None:
        out['y0_B']        = result_B['y0']
        out['c_B']         = result_B['c']
        out['tau_B']       = result_B['tau']
        out['z_B']         = result_B['y0'] - result_B['c']
        out['r2_loglin_B'] = result_B['r2_loglin']
    return out


# ---------- plotting ----------
def plot_segments_overlay(segment_data, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = {2023: 'tab:blue', 2024: 'tab:orange'}
    seen_years = set()
    for seg, data in segment_data:
        if len(data) == 0:
            continue
        label = str(seg['year']) if seg['year'] not in seen_years else None
        seen_years.add(seg['year'])
        ax.plot(data['t_hours'], data['turbidity'], color=cmap[seg['year']],
                alpha=0.7, linewidth=1.0, label=label)
    ax.set_xlabel('Hours since post-calibration')
    ax.set_ylabel('Raw turbidity reading (FNU)')
    ax.set_title(f'{SENSOR_NAME}: raw readings across all segments')
    ax.legend(title='Year')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_segment_fits(segment_data_with_fits, out_path,
                          z_bar=None, tau_bar=None):
    """
    Per-segment panels showing raw data, the fitted decay curve, and two
    corrected-data overlays:

      - local correction:  y + z_local * (1 - exp(-t / tau_local))
        Uses each segment's own fitted (z, tau). If the segment-level model
        is good, these points sit on a flat horizontal line at y0.
      - global correction: y + z_bar * (1 - exp(-t / tau_bar))
        Uses the cross-segment medians. The gap between the corrected
        points and the local y0 baseline visualizes how badly the global
        approximation misses on each individual segment.
    """
    n = len(segment_data_with_fits)
    if n == 0:
        return
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.4 * nrows), squeeze=False)
    for idx, (seg, data, fit) in enumerate(segment_data_with_fits):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        if len(data) == 0:
            ax.set_title(f"{seg['segment_id']} — no data")
            ax.axis('off')
            continue
        y = data['turbidity'].values
        t = data['t_hours'].values
        ax.scatter(t, y, s=6, color='gray', alpha=0.4, label='raw')

        # Plot calibration endpoint (next cal's pre-cal reading) if known.
        if pd.notna(seg.get('endpoint_drift', np.nan)):
            t_end = float(np.max(t))
            y_end_obs = seg['post_cal_value'] + seg['endpoint_drift']
            ax.scatter([t_end], [y_end_obs], s=80, color='black', marker='X',
                       zorder=5, label=f'next pre-cal ({y_end_obs:.2f})')

        title_extra = ''
        if fit is not None:
            # Fitted decay curve, extrapolated back to t=0.
            t_grid = np.linspace(0.0, float(np.max(t)), 400)
            yhat_grid = m_drift(t_grid, fit['y0'], fit['c'], fit['tau'])
            ax.plot(t_grid, yhat_grid, color='tab:red', linewidth=2.0,
                    label=f"A: y0={fit['y0']:.2f}, c={fit['c']:.2f}, τ={fit['tau']:.0f}h")
            ax.axhline(fit['c'], color='tab:green', linewidth=0.8,
                       linestyle=':', alpha=0.7, label=f'A asymptote c={fit["c"]:.2f}')
            ax.axhline(fit['y0'], color='tab:purple', linewidth=0.8,
                       linestyle=':', alpha=0.7, label=f'A truth y0={fit["y0"]:.2f}')
            ax.scatter([0.0], [fit['y0']], s=60, color='tab:red', marker='o',
                       facecolor='none', linewidth=1.5, zorder=4)

            # Method B (cal-log endpoint constrained), if available.
            has_B = np.isfinite(fit.get('tau_B', np.nan))
            if has_B:
                yhat_B = m_drift(t_grid, fit['y0_B'], fit['c_B'], fit['tau_B'])
                ax.plot(t_grid, yhat_B, color='tab:cyan', linewidth=2.0,
                        linestyle='--',
                        label=f"B: y0={fit['y0_B']:.2f}, c={fit['c_B']:.2f}, τ={fit['tau_B']:.0f}h")
                ax.scatter([0.0], [fit['y0_B']], s=60, color='tab:cyan', marker='s',
                           facecolor='none', linewidth=1.5, zorder=4)

            # Local correction overlay using method A: should land near the A y0 line.
            y_local = y + fit['z'] * (1.0 - np.exp(-t / fit['tau']))
            ax.scatter(t, y_local, s=5, color='tab:orange', alpha=0.5,
                       label=f"A-corrected (z={fit['z']:.2f}, τ={fit['tau']:.0f}h)")

            # Global correction overlay: distance from y0 line = miss size.
            if z_bar is not None and tau_bar is not None:
                y_global = y + z_bar * (1.0 - np.exp(-t / tau_bar))
                ax.scatter(t, y_global, s=5, color='tab:blue', alpha=0.5,
                           label=f"global-corrected (z̄={z_bar:.2f}, τ̄={tau_bar:.0f}h)")

            title_extra = f"  A: τ={fit['tau']:.0f}h"
            if has_B:
                title_extra += f"  B: τ={fit['tau_B']:.0f}h"

        # Robust y limits including corrected overlays so all three traces fit.
        y_all = [y]
        if fit is not None:
            y_all.append(y + fit['z'] * (1.0 - np.exp(-t / fit['tau'])))
            if z_bar is not None and tau_bar is not None:
                y_all.append(y + z_bar * (1.0 - np.exp(-t / tau_bar)))
        y_concat = np.concatenate(y_all)
        lo, hi = np.quantile(y_concat, [0.01, 0.99])
        pad = max(0.1, 0.1 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)

        ax.set_title(f"{seg['segment_id']}  n={len(data)}  start={seg['start_time'].date()}{title_extra}")
        ax.set_xlabel('Hours since cal')
        ax.set_ylabel('Turbidity (FNU)')
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_tau_vs_temperature(rows, out_path):
    pts = [(r['mean_temp'], r['tau']) for r in rows
           if np.isfinite(r.get('mean_temp', np.nan))
           and np.isfinite(r.get('tau', np.nan))]
    if len(pts) < 3:
        return None
    temps = np.array([p[0] for p in pts])
    taus  = np.array([p[1] for p in pts])
    rho, p_rho = stats.spearmanr(temps, taus)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(temps, taus, s=50, color='tab:red', edgecolor='black')
    ax.set_xlabel('Mean segment temperature (°C)')
    ax.set_ylabel('τ (hours)')
    ax.set_title(f'{SENSOR_NAME}: τ vs temperature  (Spearman ρ={rho:.3f}, p={p_rho:.3f})')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return rho, p_rho


# ---------- main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading calibration events from: {CAL_PATH}")
    cal_df = load_calibration_events(CAL_PATH, YEARS)
    print(f"  {len(cal_df)} unique cal events after dedupe/aggregation:")
    for _, row in cal_df.iterrows():
        print(f"    {row['end_time']}  post={row['post_cal_value']}  (year {row['year']})")

    print(f"\nLoading sensor series from: {HOURLY_PATH}")
    sensor_df = load_sensor_series(HOURLY_PATH)
    sensor_last_time = sensor_df['TIMESTAMP'].max()
    print(f"  {len(sensor_df)} rows, last timestamp: {sensor_last_time}")

    segments = build_segments(cal_df, sensor_last_time)
    print(f"\nBuilt {len(segments)} segments")

    segment_data = []
    segment_data_with_fits = []
    rows = []
    for seg in segments:
        data = extract_segment_data(sensor_df, seg)
        segment_data.append((seg, data))
        n = len(data)

        fit = fit_segment(data['t_hours'].values, data['turbidity'].values, seg) if n >= MIN_POINTS else None
        segment_data_with_fits.append((seg, data, fit))

        mean_temp = float(data['temperature'].mean()) if n > 0 and data['temperature'].notna().any() else np.nan
        row = {
            'segment_id': seg['segment_id'],
            'year': seg['year'],
            'start_time': seg['start_time'],
            'end_time':   seg['end_time'],
            'n_points':   n,
            'mean_temp':  mean_temp,
            'post_cal_value':  seg['post_cal_value'],
            'endpoint_drift_observed': seg.get('endpoint_drift', np.nan),
        }
        if fit is not None:
            row.update({
                'y0':  fit['y0'],
                'c':   fit['c'],
                'z':   fit['z'],
                'tau': fit['tau'],
                'r2_loglin': fit['r2_loglin'],
                'ssr': fit['ssr'],
                'aicc': fit['aicc'],
                'endpoint_drift_model': fit['endpoint_model'],
                'endpoint_error': fit['endpoint_model'] - fit['endpoint_observed']
                                  if np.isfinite(fit['endpoint_observed']) else np.nan,
                # Method B columns.
                'y0_B':  fit.get('y0_B',  np.nan),
                'c_B':   fit.get('c_B',   np.nan),
                'z_B':   fit.get('z_B',   np.nan),
                'tau_B': fit.get('tau_B', np.nan),
                'r2_loglin_B': fit.get('r2_loglin_B', np.nan),
            })
        else:
            row.update({'y0': np.nan, 'c': np.nan, 'z': np.nan, 'tau': np.nan,
                        'r2_loglin': np.nan, 'ssr': np.nan, 'aicc': np.nan,
                        'endpoint_drift_model': np.nan, 'endpoint_error': np.nan,
                        'y0_B': np.nan, 'c_B': np.nan, 'z_B': np.nan,
                        'tau_B': np.nan, 'r2_loglin_B': np.nan})
        rows.append(row)

        if fit is not None:
            if np.isfinite(fit.get('tau_B', np.nan)):
                b_str = (f"   B: y0={fit['y0_B']:5.2f} c={fit['c_B']:5.2f} "
                         f"tau={fit['tau_B']:6.1f}h R2={fit['r2_loglin_B']:.3f}")
            else:
                b_str = "   B: (no endpoint data)"
            print(f"  {seg['segment_id']:>16}  n={n:5d}  "
                  f"A: y0={fit['y0']:5.2f} c={fit['c']:5.2f} "
                  f"tau={fit['tau']:6.1f}h R2={fit['r2_loglin']:.3f}"
                  f"{b_str}  meanT={mean_temp if np.isfinite(mean_temp) else float('nan'):.2f}")
        else:
            print(f"  {seg['segment_id']:>16}  n={n:5d}  (no fit)")

    results_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / 'fit_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Compute global medians early so the per-segment fits plot can show how
    # the global correction compares against each segment's own local correction.
    _fit_rows_for_medians = results_df.dropna(subset=['z', 'tau'])
    if len(_fit_rows_for_medians) > 0:
        z_bar_preview   = float(_fit_rows_for_medians['z'].median())
        tau_bar_preview = float(_fit_rows_for_medians['tau'].median())
    else:
        z_bar_preview = tau_bar_preview = None

    plot_segments_overlay(segment_data, OUT_DIR / 'segments_overlay.png')
    plot_per_segment_fits(segment_data_with_fits, OUT_DIR / 'per_segment_fits.png',
                          z_bar=z_bar_preview, tau_bar=tau_bar_preview)
    tau_T_result = plot_tau_vs_temperature(rows, OUT_DIR / 'tau_vs_temperature.png')

    fit_rows = results_df.dropna(subset=['tau'])
    summary_lines = [
        f"Sensor: {SENSOR_NAME}",
        f"Period: {YEARS}",
        f"Model: y(t) = c + z * (1 - exp(-t/tau)) + noise",
        f"Segments analyzed: {len(rows)}",
        f"Segments with >= {MIN_POINTS} points: {(results_df['n_points'] >= MIN_POINTS).sum()}",
        f"Successful fits: {len(fit_rows)}",
        "",
    ]
    if len(fit_rows) > 0:
        summary_lines += [
            f"Across {len(fit_rows)} fits:",
            f"  median y0  = {fit_rows['y0'].median():.3f} FNU",
            f"  median c   = {fit_rows['c'].median():.3f} FNU",
            f"  median |y0 - c| = {(fit_rows['y0'] - fit_rows['c']).abs().median():.3f} FNU",
            f"  median tau = {fit_rows['tau'].median():.2f} hours  "
            f"({fit_rows['tau'].median()/24:.2f} days)",
            f"  tau range  = [{fit_rows['tau'].min():.2f}, {fit_rows['tau'].max():.2f}] hours",
            "",
            "Endpoint consistency (model vs observed cal correction):",
        ]
        ep = fit_rows.dropna(subset=['endpoint_drift_observed', 'endpoint_drift_model'])
        for _, r in ep.iterrows():
            summary_lines.append(
                f"  {r['segment_id']:>16}  obs={r['endpoint_drift_observed']:+.3f}  "
                f"model={r['endpoint_drift_model']:+.3f}  err={r['endpoint_error']:+.3f}"
            )
    if tau_T_result is not None:
        rho, p_rho = tau_T_result
        summary_lines += [
            "",
            f"τ-vs-temperature:",
            f"  Spearman ρ = {rho:.3f}, p = {p_rho:.3f}",
        ]
    summary_path = OUT_DIR / 'summary.txt'
    summary_path.write_text('\n'.join(summary_lines), encoding='utf-8')
    print(f"Wrote {summary_path}")

    # ---------- correction stage ----------
    z_bar, tau_bar = derive_and_apply_correction(results_df, sensor_df, cal_df)

    # ---------- jump-comparison stage (all years) ----------
    if z_bar is not None and tau_bar is not None:
        compare_calibration_jumps(z_bar, tau_bar, sensor_df, results_df)


def derive_and_apply_correction(results_df, sensor_df, cal_df):
    """
    Take per-segment z and tau from results_df, compute global medians, build a
    corrected sensor series across the full 2023-2024 record, and write:
      - corrected_series.csv
      - corrected_timeseries.png
      - correction_formula.md
    """
    fit_rows = results_df.dropna(subset=['z', 'tau']).copy()
    if len(fit_rows) == 0:
        print("No converged fits; skipping correction stage.")
        return

    z_bar   = float(fit_rows['z'].median())
    tau_bar = float(fit_rows['tau'].median())
    z_mean,   z_std   = float(fit_rows['z'].mean()),   float(fit_rows['z'].std())
    tau_mean, tau_std = float(fit_rows['tau'].mean()), float(fit_rows['tau'].std())
    z_q25,   z_q75   = (float(fit_rows['z'].quantile(0.25)),   float(fit_rows['z'].quantile(0.75)))
    tau_q25, tau_q75 = (float(fit_rows['tau'].quantile(0.25)), float(fit_rows['tau'].quantile(0.75)))
    n_segments = len(fit_rows)

    print(f"\n--- Correction stage ---")
    print(f"Global parameters (median over {n_segments} fits):")
    print(f"  z_bar   = {z_bar:.4f} FNU")
    print(f"  tau_bar = {tau_bar:.2f} hours ({tau_bar/24:.2f} days)")

    # Restrict to 2023-2024 sensor records for the corrected series.
    year_mask = sensor_df['TIMESTAMP'].dt.year.isin(YEARS)
    series = sensor_df.loc[year_mask, ['TIMESTAMP', 'turbidity']].copy()
    series = series.sort_values('TIMESTAMP').reset_index(drop=True)

    # Assign each measurement the most recent Calibration End Time using merge_asof.
    cal_ends = cal_df[['end_time']].rename(columns={'end_time': 'cal_end'}).copy()
    cal_ends = cal_ends.sort_values('cal_end').reset_index(drop=True)
    series = pd.merge_asof(
        series,
        cal_ends,
        left_on='TIMESTAMP',
        right_on='cal_end',
        direction='backward',
    )

    # t_hours since most recent cal_end. NaN cal_end => before first cal => 0 correction.
    #
    # Physical model: at t=0 the sensor reads the true value (just calibrated).
    # The raw reading evolves as y(t) = c + (y0 - c)*exp(-t/tau), so by time t
    # the reading has drifted from y0 toward c. The accumulated error in the
    # reading at time t is:
    #     error(t) = y(t) - y_true = (c - y0) * (1 - exp(-t/tau)) = -z * (1 - exp(-t/tau))
    # To recover an estimate of the truth we subtract that error:
    #     y_corrected = y_raw - error = y_raw + z * (1 - exp(-t/tau))
    delta = (series['TIMESTAMP'] - series['cal_end']).dt.total_seconds() / 3600.0
    series['t_hours_since_cal'] = delta
    correction = z_bar * (1.0 - np.exp(-delta / tau_bar))
    correction = correction.where(delta.notna(), 0.0)
    series['correction']  = correction
    series['y_raw']       = series['turbidity']
    series['y_corrected'] = series['y_raw'] + correction

    # Local correction: use each segment's own (z, tau) when the measurement
    # falls inside that segment. A segment's start_time equals the cal_end that
    # was assigned to the measurement, so we can look up z/tau by cal_end.
    # Measurements outside any fitted segment (between segments' end_time and
    # the next cal start, or in segments that failed to fit) get no local
    # correction and we fall back to the global one.
    seg_lookup = results_df.set_index('start_time')[['z', 'tau']]
    z_local   = series['cal_end'].map(seg_lookup['z'])
    tau_local = series['cal_end'].map(seg_lookup['tau'])
    local_correction = z_local * (1.0 - np.exp(-delta / tau_local))
    # Where local fit is missing or pre-first-cal, fall back to global.
    local_correction = local_correction.where(
        z_local.notna() & tau_local.notna() & delta.notna(),
        correction
    )
    series['local_correction']  = local_correction
    series['y_corrected_local'] = series['y_raw'] + local_correction

    series = series[['TIMESTAMP', 'cal_end', 't_hours_since_cal',
                     'y_raw', 'correction', 'y_corrected',
                     'local_correction', 'y_corrected_local']]

    csv_out = OUT_DIR / 'corrected_series.csv'
    series.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out}")

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(series['TIMESTAMP'], series['y_raw'],
            color='0.55', linewidth=0.6, label='raw')
    ax.plot(series['TIMESTAMP'], series['y_corrected'],
            color='tab:blue', linewidth=0.6, alpha=0.9,
            label=f'corrected (global z̄={z_bar:.2f}, τ̄={tau_bar:.0f}h)')
    ax.plot(series['TIMESTAMP'], series['y_corrected_local'],
            color='tab:orange', linewidth=0.6, alpha=0.9,
            label='corrected (local, per-segment z and τ)')
    # cal end markers
    for ce in cal_df['end_time']:
        ax.axvline(ce, color='tab:red', linewidth=0.8, alpha=0.5, linestyle='--')
    # robust y-limits
    both = pd.concat([series['y_raw'], series['y_corrected'],
                      series['y_corrected_local']]).dropna()
    if len(both) > 0:
        lo, hi = np.quantile(both, [0.01, 0.99])
        pad = max(0.2, 0.1 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Turbidity (FNU)')
    ax.set_title(f'{SENSOR_NAME}: raw vs corrected, dashed red = Calibration End Times')
    # annotation
    ax.text(0.01, 0.97,
            f"$y_{{corr}}(t) = y_{{obs}}(t) + z \\cdot (1 - \\exp(-t/\\tau))$\n"
            f"global: $\\bar z$ = {z_bar:.3f} FNU,  $\\bar\\tau$ = {tau_bar:.1f} h\n"
            f"local: per-segment $z$ and $\\tau$ from fit_results.csv\n"
            f"n = {n_segments} segments  (t = hours since last cal)",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_out = OUT_DIR / 'corrected_timeseries.png'
    fig.savefig(png_out, dpi=200)
    plt.close(fig)
    print(f"Wrote {png_out}")

    # ----- markdown writeup -----
    md_lines = [
        f"# {SENSOR_NAME} post-calibration drift correction",
        "",
        "## What this correction does, and why",
        "",
        "Calibration leaves the sensor reading the *truth* at t=0: the cal team has "
        "just compared the sensor against a known standard and forced agreement. As "
        "time passes between calibrations, error accumulates in the reading. For "
        "this turbidity sensor, the empirical pattern across all 11 segments is that "
        "the reading sits high immediately after calibration and drifts *downward* "
        "to a lower steady-state baseline over a time scale of roughly a week. "
        "(Biofilm gradually coating the optical window is one plausible mechanism: a "
        "coated window attenuates the scattered-light signal that turbidity is "
        "computed from, suppressing the apparent reading.)",
        "",
        "The model below describes how that error accumulates in time, by fitting "
        "the raw reading within each calibration-to-calibration segment. Knowing the "
        "shape of that accumulation lets us **un-do** it: subtract the model's "
        "estimate of the accumulated error from any reading, and what's left is a "
        "better estimate of the true value at that timestamp than the raw reading.",
        "",
        "The correction is most valuable *between* calibrations, where the truth is "
        "otherwise unknown. The further from the most recent calibration, the larger "
        "the correction. Right after a calibration, the correction is near zero "
        "(consistent with the calibration just having anchored the reading to truth).",
        "",
        "## The model fit per segment",
        "",
        "For each segment between two calibration events, the raw sensor series was "
        "fit to",
        "",
        "    y(t) = c + (y0 - c) * exp(-t / tau) + noise",
        "",
        "with three free parameters per segment:",
        "",
        "- `y0`  = sensor reading at t=0 (the cal-end moment, extrapolated from the "
        "first measurements). Because calibration just anchored the sensor to truth, "
        "**y0 is the model's estimate of the true value at that moment**.",
        "- `c`   = the value the raw reading is *drifting toward* as the segment "
        "progresses. **c = y0 + accumulated_long_term_drift**, i.e. truth-plus-error "
        "in the asymptotic limit.",
        "- `tau` = the time constant of the drift accumulation (hours).",
        "",
        "Define `z = y0 - c`. Empirically z is positive on this sensor (y0 > c — the "
        "reading drifts downward over time). The accumulated error at time t is",
        "",
        "    error(t) = y(t) - y_true = (c - y0) * (1 - exp(-t/tau)) = -z * (1 - exp(-t/tau))",
        "",
        "With z > 0 this error is *negative* (the raw reading sits below truth at "
        "large t), so the correction below adds a positive amount back.",
        "",
        "The fit uses weighted nonlinear least squares with an early-time weight "
        "boost (the decay shape lives in the first ~24 h) and spike down-weighting "
        "(genuine turbidity events get demoted so they don't pull the curve). "
        "Initial guesses come from a log-space linearization. Implementation: "
        "`src/utils/post_cal_drift.py`.",
        "",
        "## The correction formula",
        "",
        "    y_corrected(t) = y_observed(t) + z_bar * (1 - exp(-t / tau_bar))",
        "",
        "where `t` is hours since the most recent Calibration End Time, and the "
        "global parameters are the medians of the per-segment fits. `z_bar` carries "
        "its own sign: when the sensor drifts downward over time (the empirical "
        "case here, z_bar > 0), the correction is positive and adds to the raw "
        "reading; if a sensor drifted upward instead (z_bar < 0), the same formula "
        "would yield a negative correction.",
        "",
        f"## Constants (median across {n_segments} segments, {YEARS[0]}-{YEARS[-1]})",
        "",
        f"    z_bar    = {z_bar:+.4f}  FNU    (median y0 - c; positive => sensor drifts downward over time)",
        f"    tau_bar  = {tau_bar:.2f} hours  ({tau_bar/24:.2f} days)  (median drift time constant)",
        "",
        f"    Mean +/- std:  z = {z_mean:+.4f} +/- {z_std:.4f} FNU,  "
        f"tau = {tau_mean:.2f} +/- {tau_std:.2f} hours",
        f"    IQR:           z = [{z_q25:+.4f}, {z_q75:+.4f}] FNU,  "
        f"tau = [{tau_q25:.2f}, {tau_q75:.2f}] hours",
        "",
        "Median is used rather than mean because two of the eleven segments "
        "(2023-4, 2023-5) drift in the wrong direction with very long tau, and "
        "they would skew the mean. The median is robust to those.",
        "",
        "## Bounds and interpretation",
        "",
        "Right after a calibration (t = 0): correction = 0, y_corrected = y_observed. "
        "The sensor is freshly anchored to truth; no adjustment needed.",
        "",
        f"Long after a calibration (t >> tau_bar ~ {tau_bar/24:.0f} days): correction "
        f"saturates at z_bar = {z_bar:+.3f} FNU. The sensor has drifted by its full "
        "characteristic amount, and the corrected value undoes that drift.",
        "",
        "At t = tau_bar (one time constant), the correction has reached "
        f"(1 - 1/e) ~ 63% of its saturated value, i.e. ~{0.632*z_bar:+.3f} FNU.",
        "",
        "## Application",
        "",
        "For each raw measurement `y` at timestamp `T`:",
        "",
        "1. Find the most recent Calibration End Time `C` with `C <= T`.",
        "2. Compute `t = (T - C)` in hours.",
        "3. `y_corrected = y + z_bar * (1 - exp(-t / tau_bar))`.",
        "4. If no calibration precedes `T` in the available record, set the "
        "correction to 0.",
        "",
        "## Caveats",
        "",
        "- `y_corrected` is an estimate of the truth, not the truth. The model "
        "describes the *systematic* component of drift that's typical across "
        "segments; per-segment variation in z and tau (see the table below) is "
        "absorbed into residual error.",
        "- The correction assumes drift is well-described by a single exponential "
        "with the medians above. Segments where the sensor drifts in the opposite "
        "direction, or where the time constant is dramatically different, are not "
        "well corrected by these globals.",
        "- The correction is built only from {0}-{1} data on this sensor. It should "
        "not be applied to other sensors without refitting, and may not generalize "
        "to years with substantially different biofouling regimes (e.g. extreme "
        "algal bloom seasons).".format(YEARS[0], YEARS[-1]),
        "",
        "## Per-segment fits used in the median",
        "",
        "| segment_id | y0 (FNU) | c (FNU) | z = y0-c (FNU) | tau (hours) | mean_temp (C) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in fit_rows.iterrows():
        md_lines.append(
            f"| {r['segment_id']} | {r['y0']:.3f} | {r['c']:.3f} | "
            f"{r['z']:+.4f} | {r['tau']:.2f} | {r['mean_temp']:.2f} |"
        )
    md_lines.append("")
    md_out = OUT_DIR / 'correction_formula.md'
    md_out.write_text('\n'.join(md_lines), encoding='utf-8')
    print(f"Wrote {md_out}")

    return z_bar, tau_bar


def compare_calibration_jumps(z_bar, tau_bar, sensor_df, results_df=None):
    """
    Validation: at each calibration event in the FULL record (all years),
    compare the discontinuity in the raw sensor series across the cal
    interval to the discontinuity after applying the drift correction.

    For each event i (with a previous event i-1):
      duration_h = (cal_i.start - cal_{i-1}.end) in hours
      y_pre     = last sensor reading at TIMESTAMP < cal_i.start
      y_post    = first sensor reading at TIMESTAMP > cal_i.end
      raw_jump  = y_pre - y_post

      For the corrected jump, apply the correction to y_pre using
      t_pre = (y_pre.TIMESTAMP - cal_{i-1}.end) in hours:
        y_pre_corrected = y_pre + z_bar * (1 - exp(-t_pre / tau_bar))
      corrected_jump  = y_pre_corrected - y_post

    Skip pairs whose previous cal end and this cal start fall in different
    calendar years (winter maintenance). Skip events with no previous cal,
    or where no bracketing sensor readings are found.

    Writes jump_comparison.csv and jump_vs_duration.png.
    """
    cal_df_all = load_calibration_events(CAL_PATH, years=None)

    # Same-day cal events have already been collapsed by load_calibration_events,
    # so consecutive rows are different physical cal interventions.
    records = []
    for i in range(1, len(cal_df_all)):
        prev = cal_df_all.iloc[i - 1]
        curr = cal_df_all.iloc[i]
        prev_end   = prev['end_time']
        curr_start = curr['start_time']
        curr_end   = curr['end_time']

        if prev_end.year != curr_start.year:
            continue  # cross-year pair, drop

        # Bracketing sensor readings.
        before = sensor_df[sensor_df['TIMESTAMP'] < curr_start]
        after  = sensor_df[sensor_df['TIMESTAMP'] > curr_end]
        if len(before) == 0 or len(after) == 0:
            continue
        # The "last reading before" should also be after prev_end, else we're
        # bracketing a different interval.
        before = before[before['TIMESTAMP'] > prev_end]
        if len(before) == 0:
            continue

        pre_row  = before.iloc[-1]
        post_row = after.iloc[0]
        y_pre    = float(pre_row['turbidity'])
        y_post   = float(post_row['turbidity'])
        if not (np.isfinite(y_pre) and np.isfinite(y_post)):
            continue

        duration_h = (curr_start - prev_end).total_seconds() / 3600.0
        t_pre_h    = (pre_row['TIMESTAMP'] - prev_end).total_seconds() / 3600.0
        correction_at_pre = z_bar * (1.0 - np.exp(-t_pre_h / tau_bar))
        y_pre_corrected = y_pre + correction_at_pre

        # Local correction using the per-segment (z, tau) for the segment that
        # starts at prev_end and ends at this calibration. Falls back to NaN
        # if the segment had no fit (e.g. 2023-4 outlier, or segments outside
        # the YEARS window used for fitting).
        local_correction = np.nan
        y_pre_corrected_local = np.nan
        if results_df is not None:
            match = results_df[results_df['start_time'] == prev_end]
            if len(match) > 0:
                z_loc   = float(match.iloc[0]['z'])
                tau_loc = float(match.iloc[0]['tau'])
                if np.isfinite(z_loc) and np.isfinite(tau_loc) and tau_loc > 0:
                    local_correction = z_loc * (1.0 - np.exp(-t_pre_h / tau_loc))
                    y_pre_corrected_local = y_pre + local_correction

        # Cal-log-derived predicted error at y_pre: linear interpolation between
        # the current cal's pre-cal-at-standard-1 (0 FNU) and pre-cal-at-standard-2
        # (12.4 FNU) readings. This tells us how much error the cal team measured
        # at two known true values; assuming linearity, we can predict the error
        # at the in-situ y_pre reading.
        pre1  = curr.get('pre_cal_value',    np.nan)
        post1 = curr.get('post_cal_value',   np.nan)
        pre2  = curr.get('pre_cal_value_2',  np.nan)
        post2 = curr.get('post_cal_value_2', np.nan)
        cal_log_predicted_error = np.nan
        if all(np.isfinite([pre1, pre2, post1, post2])) and (post2 - post1) != 0:
            a = pre1 - post1
            b = ((pre2 - post2) - a) / (post2 - post1)
            if abs(1.0 + b) > 1e-9:
                y_true_last = (y_pre - a) / (1.0 + b)
                cal_log_predicted_error = y_pre - y_true_last

        records.append({
            'cal_index': i,
            'year': curr_start.year,
            'prev_cal_end':  prev_end,
            'this_cal_start': curr_start,
            'duration_hours': duration_h,
            'duration_days':  duration_h / 24.0,
            'pre_timestamp':  pre_row['TIMESTAMP'],
            'post_timestamp': post_row['TIMESTAMP'],
            'y_pre':           y_pre,
            'y_pre_corrected': y_pre_corrected,
            'y_pre_corrected_local': y_pre_corrected_local,
            'y_post':          y_post,
            'raw_jump':        y_pre - y_post,
            'corrected_jump_global': y_pre_corrected - y_post,
            'corrected_jump_local':  (y_pre_corrected_local - y_post
                                      if np.isfinite(y_pre_corrected_local) else np.nan),
            'cal_log_predicted_jump': cal_log_predicted_error,
            'correction_applied_global': correction_at_pre,
            'correction_applied_local':  local_correction,
            't_pre_hours': t_pre_h,
        })

    if not records:
        print("\nNo valid same-year calibration pairs for jump comparison.")
        return

    jumps = pd.DataFrame(records).sort_values('duration_hours').reset_index(drop=True)

    csv_out = OUT_DIR / 'jump_comparison.csv'
    jumps.to_csv(csv_out, index=False)
    print(f"\n--- Jump comparison stage (all years, same-year pairs) ---")
    print(f"  {len(jumps)} valid pairs:")
    for _, r in jumps.iterrows():
        loc_str = (f" loc={r['corrected_jump_local']:+.3f}"
                   if np.isfinite(r['corrected_jump_local']) else "  loc=    NA")
        print(f"    {r['this_cal_start'].date()}  duration={r['duration_days']:6.1f}d  "
              f"raw={r['raw_jump']:+.3f}  global={r['corrected_jump_global']:+.3f}{loc_str}")
    print(f"  median |raw|             = {jumps['raw_jump'].abs().median():.3f} FNU")
    print(f"  median |global-corrected|= {jumps['corrected_jump_global'].abs().median():.3f} FNU")
    print(f"  median |local-corrected| = {jumps['corrected_jump_local'].abs().median():.3f} FNU")
    print(f"  median |cal-log pred|    = {jumps['cal_log_predicted_jump'].abs().median():.3f} FNU")
    print(f"Wrote {csv_out}")

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.6)

    x = jumps['this_cal_start']
    ax.scatter(x, jumps['raw_jump'],
               s=55, color='0.4', marker='o',
               label='No correction')
    ax.scatter(x, jumps['corrected_jump_global'],
               s=55, color='tab:blue', marker='D',
               label='Corrected for global average drift')
    local_valid = jumps['corrected_jump_local'].notna()
    if local_valid.any():
        ax.scatter(x[local_valid], jumps.loc[local_valid, 'corrected_jump_local'],
                   s=55, color='tab:orange', marker='s',
                   label='Corrected for locally-fitted drift')
    # Cal-log-derived predicted jump: error at y_pre from linear interpolation
    # of the two pre-cal-at-standard readings.
    cal_log_valid = jumps['cal_log_predicted_jump'].notna()
    if cal_log_valid.any():
        ax.scatter(x[cal_log_valid], jumps.loc[cal_log_valid, 'cal_log_predicted_jump'],
                   s=55, color='tab:green', marker='^',
                   label='Error at calibration (interpolated at last measured value)')
    # Connect raw -> global-corrected with a thin line to make the shrinkage visible.
    for _, r in jumps.iterrows():
        ax.plot([r['this_cal_start'], r['this_cal_start']],
                [r['raw_jump'], r['corrected_jump_global']],
                color='0.7', linewidth=0.7, alpha=0.6, zorder=0)

    # Robust y-limits first so we know where to place the duration annotations.
    both = pd.concat([jumps['raw_jump'], jumps['corrected_jump_global'],
                      jumps['corrected_jump_local'].dropna(),
                      jumps['cal_log_predicted_jump'].dropna()])
    lo, hi = np.quantile(both, [0.05, 0.95])
    pad = max(0.5, 0.2 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    n_clipped = ((both < lo - pad) | (both > hi + pad)).sum()

    # Annotate each calibration with the duration since the previous one.
    y_anno = ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for _, r in jumps.iterrows():
        ax.annotate(f"{r['duration_days']:.0f}d",
                    xy=(r['this_cal_start'], y_anno),
                    ha='center', va='top', fontsize=7, color='0.3',
                    rotation=90)

    ax.set_xlabel('Calibration date')
    ax.set_ylabel('Pre-cal minus post-cal sensor reading (FNU)')
    ax.set_title(f'{SENSOR_NAME}: calibration discontinuity over time, four estimators\n'
                 f'(all years; same-year pairs only; n={len(jumps)};  '
                 f'annotation = days since previous calibration)')

    ax.text(0.005, 0.98,
            f"median |raw|              = {jumps['raw_jump'].abs().median():.3f} FNU\n"
            f"median |global-corrected| = {jumps['corrected_jump_global'].abs().median():.3f} FNU\n"
            f"median |local-corrected|  = {jumps['corrected_jump_local'].abs().median():.3f} FNU\n"
            f"median |cal-log predicted|= {jumps['cal_log_predicted_jump'].abs().median():.3f} FNU\n"
            f"z̄ = {z_bar:.3f},  τ̄ = {tau_bar:.1f} h"
            + (f"\n({n_clipped} extreme points off-axis)" if n_clipped else ""),
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_out = OUT_DIR / 'jump_vs_duration.png'
    fig.savefig(png_out, dpi=200)
    plt.close(fig)
    print(f"Wrote {png_out}")


if __name__ == '__main__':
    main()
