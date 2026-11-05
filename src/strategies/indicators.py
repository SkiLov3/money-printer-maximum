from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_indicator_frame(df: pd.DataFrame, tick_size: float, limit: int | None = None) -> pd.DataFrame:
    """
    FIX: Added 'limit' parameter to optimize performance by only calculating
    indicators for the last N bars when possible.
    """
    if df.empty:
        return df.copy()
    if limit and len(df) > limit:
        # Include >=1,500 bars (over 1 full 1,380-min Globex session) so 18:00 ET session VWAP and EMA-200 are exact
        calculation_df = df.tail(limit + 1500).copy()
    else:
        calculation_df = df.copy()

    res = calculation_df

    if "timestamp" not in res.columns:
        res = res.assign(timestamp=pd.RangeIndex(start=0, stop=len(res), step=1))

    if "price" not in res.columns and "close" in res.columns:
        res = res.assign(price=res["close"])

    if "high" not in res.columns:
        res = res.assign(
            high=res["price"] + (tick_size * 2),
            low=res["price"] - (tick_size * 2),
            close=res["price"],
        )

    if "open" not in res.columns:
        res = res.assign(open=res["close"])

    if "volume" not in res.columns:
        res = res.assign(volume=1.0)

    res = res.assign(
        ema_14=res["price"].ewm(span=14, adjust=False).mean(),
        ema_18=res["price"].ewm(span=18, adjust=False).mean(),
        ema_24=res["price"].ewm(span=24, adjust=False).mean(),
        ema_50=res["price"].ewm(span=50, adjust=False).mean(),
        ema_100=res["price"].ewm(span=100, adjust=False).mean(),
        ema_200=res["price"].ewm(span=200, adjust=False).mean(),
        ema_9=res["price"].ewm(span=9, adjust=False).mean(),
        ema_21=res["price"].ewm(span=21, adjust=False).mean(),
    )

    tenkan = (res["high"].rolling(9).max() + res["low"].rolling(9).min()) / 2
    kijun = (res["high"].rolling(26).max() + res["low"].rolling(26).min()) / 2
    res = res.assign(
        tenkan=tenkan,
        kijun=kijun,
        senkou_a=((tenkan + kijun) / 2).shift(26),
        senkou_b=((res["high"].rolling(52).max() + res["low"].rolling(52).min()) / 2).shift(26),
        chikou_span=res["close"].shift(26),
    )
    res = res.assign(cloud_thickness=(res["senkou_a"] - res["senkou_b"]).abs())

    tr1 = res["high"] - res["low"]
    tr2 = abs(res["high"] - res["close"].shift(1))
    tr3 = abs(res["low"] - res["close"].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    res = res.assign(atr=tr.rolling(14).mean())

    up_move = res["high"] - res["high"].shift(1)
    down_move = res["low"].shift(1) - res["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr_s = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdm_s = pd.Series(plus_dm, index=res.index).ewm(alpha=1 / 14, adjust=False).mean()
    mdm_s = pd.Series(minus_dm, index=res.index).ewm(alpha=1 / 14, adjust=False).mean()

    plus_di = 100 * (pdm_s / (tr_s + 1e-9))
    minus_di = 100 * (mdm_s / (tr_s + 1e-9))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    res = res.assign(adx=dx.ewm(alpha=1 / 14, adjust=False).mean())

    delta = res["price"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where((gain > 0) | (loss > 0), 50.0)
    res = res.assign(rsi=rsi)

    if "timestamp" in res.columns and "volume" in res.columns:
        ts = pd.to_datetime(res["timestamp"], errors="coerce")
        vol = res["volume"].fillna(0).astype(float).replace(0, 1)
        pv = res["price"] * vol
        try:
            if hasattr(ts, "dt"):
                if ts.dt.tz is None:
                    ts = ts.dt.tz_localize("UTC")
                et = ts.dt.tz_convert("America/New_York")
                # CME Globex session begins at 18:00 ET. Bars at/after 18:00 roll to next session date.
                dates = (et + pd.Timedelta(hours=6)).dt.date
            else:
                dates = ts
        except Exception:
            dates = ts.dt.date if hasattr(ts, "dt") else ts
        cum_pv = pv.groupby(dates).cumsum()
        cum_vol = vol.groupby(dates).cumsum()
        res = res.assign(vwap=cum_pv / cum_vol)
    else:
        res = res.assign(vwap=res["price"])

    res = res.assign(
        rolling_high_20=res["high"].rolling(20).max(),
        rolling_low_20=res["low"].rolling(20).min(),
        rolling_high_50=res["high"].rolling(50).max(),
        rolling_low_50=res["low"].rolling(50).min(),
        close_lag_26=res["close"].shift(26),
        body_size=(res["close"] - res["open"]).abs(),
    )

    res = _append_simple_market_metrics(res)

    if limit and len(res) > limit:
        return res.tail(limit)

    return res


def _append_simple_market_metrics(res: pd.DataFrame) -> pd.DataFrame:
    frame = res.copy()
    n = len(frame)
    open_arr = frame["open"].to_numpy(dtype=float)
    close_arr = frame["close"].to_numpy(dtype=float)
    ha_close = (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
    ha_close_arr = ha_close.to_numpy(dtype=float)
    ha_open_values = np.empty(n, dtype=float)
    if n > 0:
        ha_open_values[0] = (open_arr[0] + close_arr[0]) / 2.0
        for idx in range(1, n):
            ha_open_values[idx] = (ha_open_values[idx - 1] + ha_close_arr[idx - 1]) / 2.0
    ha_open = pd.Series(ha_open_values, index=frame.index)
    ha_high = pd.concat([frame["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([frame["low"], ha_open, ha_close], axis=1).min(axis=1)

    ha_tr1 = ha_high - ha_low
    ha_tr2 = (ha_high - ha_close.shift(1)).abs()
    ha_tr3 = (ha_low - ha_close.shift(1)).abs()
    ha_tr = pd.concat([ha_tr1, ha_tr2, ha_tr3], axis=1).max(axis=1)
    atr8 = ha_tr.ewm(alpha=1 / 8, adjust=False).mean()
    up = ((ha_high + ha_low) / 2.0) - (1.3 * atr8)
    down = ((ha_high + ha_low) / 2.0) + (1.3 * atr8)

    up_arr = up.to_numpy(dtype=float)
    down_arr = down.to_numpy(dtype=float)
    trend_up = np.empty(n, dtype=float)
    trend_down = np.empty(n, dtype=float)
    switches = np.ones(n, dtype=int)
    if n > 0:
        trend_up[0] = up_arr[0]
        trend_down[0] = down_arr[0]
        for idx in range(1, n):
            prev_up = trend_up[idx - 1]
            prev_down = trend_down[idx - 1]
            prev_close = ha_close_arr[idx - 1]
            current_close = ha_close_arr[idx]
            current_up = up_arr[idx]
            current_down = down_arr[idx]
            trend_up[idx] = max(current_up, prev_up) if prev_close > prev_up else current_up
            trend_down[idx] = min(current_down, prev_down) if prev_close < prev_down else current_down
            if current_close > prev_down:
                switches[idx] = 1
            elif current_close < prev_up:
                switches[idx] = -1
            else:
                switches[idx] = switches[idx - 1]

    up_move = ha_high - ha_high.shift(1)
    down_move = ha_low.shift(1) - ha_low
    smm_plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    smm_minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    smm_tr_s = ha_tr.ewm(alpha=1 / 14, adjust=False).mean()
    smm_pdm_s = pd.Series(smm_plus_dm, index=frame.index).ewm(alpha=1 / 14, adjust=False).mean()
    smm_mdm_s = pd.Series(smm_minus_dm, index=frame.index).ewm(alpha=1 / 14, adjust=False).mean()
    smm_plus_di = 100.0 * smm_pdm_s / (smm_tr_s + 1e-9)
    smm_minus_di = 100.0 * smm_mdm_s / (smm_tr_s + 1e-9)

    typical = (ha_high + ha_low + ha_close) / 3.0
    money_flow = typical * frame["volume"].fillna(0).astype(float).replace(0, 1)
    positive_flow = money_flow.where(typical > typical.shift(1), 0.0)
    negative_flow = money_flow.where(typical < typical.shift(1), 0.0)
    positive_sum = positive_flow.rolling(10).sum()
    negative_sum = negative_flow.rolling(10).sum()
    mfi = 100.0 * positive_sum / (positive_sum + negative_sum + 1e-9)
    mfi = mfi.where((positive_sum > 0) | (negative_sum > 0), 50.0)

    frame = frame.assign(
        smm_ha_open=ha_open,
        smm_ha_high=ha_high,
        smm_ha_low=ha_low,
        smm_ha_close=ha_close,
        smm_ema8=ha_close.ewm(span=8, adjust=False).mean(),
        smm_ema13=ha_close.ewm(span=13, adjust=False).mean(),
        smm_ema21=ha_close.ewm(span=21, adjust=False).mean(),
        smm_ema200=ha_close.ewm(span=200, adjust=False).mean(),
        smm_trend_switch=pd.Series(switches, index=frame.index),
        smm_plus_di=smm_plus_di,
        smm_minus_di=smm_minus_di,
        smm_mfi=mfi,
    )
    return frame
