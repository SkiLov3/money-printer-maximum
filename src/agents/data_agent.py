from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

import pandas as pd

from src.charting import normalize_frame, resample_timeframe


def parse_market_timestamp(raw_ts) -> datetime:
    """Robustly parses market timestamps from diverse shapes (ISO, epoch s/ms/us/ns, datetime) to naive UTC datetime."""
    if raw_ts is None or (isinstance(raw_ts, str) and not raw_ts.strip()):
        return datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        val = float(raw_ts)
        import math
        if math.isnan(val) or math.isinf(val) or val <= 0:
            return datetime.now(timezone.utc).replace(tzinfo=None)
        if val > 1e16:
            return pd.to_datetime(val, unit='ns', utc=True).tz_localize(None).to_pydatetime()
        elif val > 1e13:
            return pd.to_datetime(val, unit='us', utc=True).tz_localize(None).to_pydatetime()
        elif val > 1e10:
            return pd.to_datetime(val, unit='ms', utc=True).tz_localize(None).to_pydatetime()
        elif val > 1e8:
            return pd.to_datetime(val, unit='s', utc=True).tz_localize(None).to_pydatetime()
        return datetime.now(timezone.utc).replace(tzinfo=None)
    except (ValueError, TypeError, OverflowError):
        pass

    try:
        ts = pd.to_datetime(raw_ts)
        if pd.isna(ts):
            return datetime.now(timezone.utc).replace(tzinfo=None)
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            return ts.tz_convert('UTC').tz_localize(None).to_pydatetime()
        return pd.to_datetime(ts).to_pydatetime()
    except Exception:
        return datetime.now(timezone.utc).replace(tzinfo=None)


class DataAgent:
    """
    FIX H1: Aggregates streaming tick data into proper OHLCV 1-minute bars.
    The strategy agent requires high, low, close, volume columns — not just price.
    """
    def __init__(self, symbol: str, bar_interval_seconds=60, max_bars: int | None = None):
        self.symbol = symbol
        self.logger = logging.getLogger(f"DataAgent-{symbol}")
        self.lock = threading.RLock()
        
        # Completed OHLCV bars for the strategy
        # FIX: Explicitly set dtypes to avoid 'object' type issues during math/indicators
        self.bars = pd.DataFrame({
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'open': pd.Series(dtype='float64'),
            'high': pd.Series(dtype='float64'),
            'low': pd.Series(dtype='float64'),
            'close': pd.Series(dtype='float64'),
            'volume': pd.Series(dtype='float64'),
            'price': pd.Series(dtype='float64')
        })
        self.max_bars = int(max_bars or 12000)  # Enough 1m history for higher-timeframe learning review.
        
        # Current bar being built from ticks
        self.bar_interval = bar_interval_seconds
        self.current_bar = None  # {open, high, low, close, volume, start_time}
        self.tick_count = 0
        self.total_bars_generated = 0

    def _matches_symbol(self, incoming: str) -> bool:
        if not incoming:
            return False
        if incoming == self.symbol:
            return True
        import re
        def _norm(s: str) -> str:
            clean = str(s).lstrip("/").upper().strip()
            m_con = re.search(r'\b([A-Z0-9]+)\.([FGHJKMNQUVXZ]\d{1,2})\b', clean)
            if m_con:
                root, my = m_con.group(1), m_con.group(2)
                if root.endswith("E") and len(root) > 3:
                    root = root[:-1]
                elif root == "ENQ":
                    root = "NQ"
                elif root == "EP":
                    root = "ES"
                return f"{root}{my}"
            return clean

        s1 = _norm(self.symbol)
        s2 = _norm(incoming)
        if s1 == s2:
            return True
        m1 = re.match(r"^([A-Z0-9]+)([FGHJKMNQUVXZ])(\d{1,2})$", s1)
        m2 = re.match(r"^([A-Z0-9]+)([FGHJKMNQUVXZ])(\d{1,2})$", s2)
        base1 = m1.group(1) if m1 else s1
        base2 = m2.group(1) if m2 else s2
        if s1 == base2 or s2 == base1:
            return True
        if m1 and m2:
            root1, month1, yr1 = m1.groups()
            root2, month2, yr2 = m2.groups()
            if root1 == root2 and month1 == month2 and yr1[-1] == yr2[-1]:
                return True
        return False

    def handle_market_data(self, data):
        """Processes a market tick if it matches this agent's symbol."""
        incoming = data.get('symbol')
        if incoming != self.symbol and not self._matches_symbol(incoming):
            return

        try:
            tick_price = float(data.get('price', 0))
            tick_volume = float(data.get('volume', 0))
            if tick_price <= 0:
                self.logger.warning(f"Invalid tick price ({tick_price}) received for {self.symbol}. Data: {data}")
                return # Skip invalid ticks

            # Prefer incoming tick timestamp with epoch protection
            raw_ts = data.get('timestamp') or data.get('time') or data.get('t')
            tick_dt = parse_market_timestamp(raw_ts)

            # Bucket start_time strictly to the top of the calendar minute (e.g. 09:30:00)
            minute_start = tick_dt.replace(second=0, microsecond=0)

            with self.lock:
                # Reject late ticks arriving after current bar has advanced
                if self.current_bar and minute_start < self.current_bar['start_time']:
                    self.logger.debug(f"Ignoring late tick {tick_dt} for earlier bar {minute_start}")
                    return

                # Reject ticks belonging to earlier closed historical bars
                if not self.bars.empty and minute_start < self.bars['timestamp'].iloc[-1]:
                    self.logger.debug(f"Ignoring stale tick {tick_dt} for already closed bar {minute_start}")
                    return

                self.check_bar_expiry(tick_dt)
                if self.current_bar is None:
                    # Start a new bar
                    self.current_bar = {
                        'start_time': minute_start,
                        'open': tick_price,
                        'high': tick_price,
                        'low': tick_price,
                        'close': tick_price,
                        'volume': tick_volume
                    }
                    self.tick_count = 1
                else:
                    if minute_start > self.current_bar['start_time']:
                        self._close_bar(tick_dt)
                        self.current_bar = {
                            'start_time': minute_start,
                            'open': tick_price,
                            'high': tick_price,
                            'low': tick_price,
                            'close': tick_price,
                            'volume': tick_volume
                        }
                        self.tick_count = 1
                        return

                    # Update current bar
                    self.current_bar['high'] = max(self.current_bar['high'], tick_price)
                    self.current_bar['low'] = min(self.current_bar['low'], tick_price)
                    self.current_bar['close'] = tick_price
                    self.current_bar['volume'] += tick_volume
                    self.tick_count += 1
                    
                self.logger.debug(f"Tick: {tick_price} (bar ticks: {self.tick_count})")
                
        except Exception as e:
            self.logger.error(f"Error processing market data for {self.symbol}: {e}")

    def check_bar_expiry(self, now=None):
        """Checks if the current bar has expired based on time and closes it if so."""
        if now is None:
            now = datetime.now(timezone.utc).astimezone()
        
        # FIX: Ensure 'now' is naive UTC for internal comparison and storage
        if hasattr(now, 'tzinfo') and now.tzinfo is not None:
            now = now.astimezone(timezone.utc).replace(tzinfo=None)
            
        with self.lock:
            if self.current_bar:
                elapsed = (now - self.current_bar['start_time']).total_seconds()
                if elapsed >= self.bar_interval:
                    self.logger.debug(f"Bar interval {self.bar_interval}s elapsed ({elapsed:.1f}s). Closing bar.")
                    self._close_bar(now)

    def _close_bar(self, timestamp):
        """Closes the current bar and adds it to the bar history."""
        if self.current_bar is None:
            return
        
        # FIX: Ensure timestamp being stored is naive
        start_time = self.current_bar['start_time']
        if hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
            start_time = start_time.astimezone(timezone.utc).replace(tzinfo=None)

        new_bar = {
            'timestamp': start_time,
            'open': self.current_bar['open'],
            'high': self.current_bar['high'],
            'low': self.current_bar['low'],
            'close': self.current_bar['close'],
            'volume': self.current_bar['volume'],
            'price': self.current_bar['close']  # Strategy uses 'price' column
        }
        
        new_bar_df = pd.DataFrame([new_bar])
        
        # FIX: Explicitly strip timezone from timestamp BEFORE astype()
        # This prevents the TypeError: Cannot use .astype to convert from aware to naive
        if 'timestamp' in new_bar_df.columns:
            new_bar_df.loc[:, 'timestamp'] = pd.to_datetime(new_bar_df['timestamp']).dt.tz_localize(None)

        # Force other dtypes to match exactly
        type_map = {col: self.bars[col].dtype for col in self.bars.columns if col in new_bar_df.columns}
        if type_map:
            new_bar_df = new_bar_df.astype(type_map)

        self.bars = pd.concat([self.bars, new_bar_df], ignore_index=True)
        self.bars = self.bars.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp').reset_index(drop=True)
        
        # Trim to max size
        if len(self.bars) > self.max_bars:
            self.bars = self.bars.iloc[-self.max_bars:].reset_index(drop=True)
        
        self.logger.debug(
            f"Bar closed: O={new_bar['open']:.2f} H={new_bar['high']:.2f} "
            f"L={new_bar['low']:.2f} C={new_bar['close']:.2f} V={new_bar['volume']}"
        )
        
        # Reset for next bar
        self.current_bar = None
        self.tick_count = 0
        self.total_bars_generated += 1

    def get_latest_data(self):
        """Returns the completed OHLCV bars as a DataFrame for strategy analysis."""
        with self.lock:
            return self.bars.copy()

    def get_chart_frame(self, frame: str = "1m", limit: int = 500):
        """Returns visual-only bars for dashboard charting."""
        normalized = normalize_frame(frame)
        with self.lock:
            bars = self._bars_with_current_locked()
        return resample_timeframe(bars, normalized).tail(int(limit or 500)).reset_index(drop=True)

    def _bars_with_current_locked(self):
        bars = self.bars.copy()
        if self.current_bar is None:
            return bars
        start_time = self.current_bar["start_time"]
        if hasattr(start_time, "tzinfo") and start_time.tzinfo is not None:
            start_time = start_time.astimezone(timezone.utc).replace(tzinfo=None)
        current = pd.DataFrame(
            [
                {
                    "timestamp": start_time,
                    "open": self.current_bar["open"],
                    "high": self.current_bar["high"],
                    "low": self.current_bar["low"],
                    "close": self.current_bar["close"],
                    "volume": self.current_bar["volume"],
                    "price": self.current_bar["close"],
                }
            ]
        )
        combined = pd.concat([bars, current], ignore_index=True)
        return combined.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    
    def get_bar_count(self):
        """Returns the total number of bars generated across the lifetime of the agent."""
        with self.lock:
            return self.total_bars_generated

    def load_historical_bars(self, bars_data):
        """
        Pre-loads historical bars into the agent from API data.
        Handles various API field naming conventions (startTime, openPrice, etc).
        """
        if not bars_data:
            return
            
        new_rows = []
        for b in bars_data:
            # Map API fields to internal DataFrame format using flexible lookup
            ts_raw = b.get('timestamp') or b.get('startTime') or b.get('t')
            ts = parse_market_timestamp(ts_raw)

            # Map OHLCV with fallbacks for camelCase or short keys
            o = b.get('open') or b.get('openPrice') or b.get('o') or 0
            h = b.get('high') or b.get('highPrice') or b.get('h') or 0
            l = b.get('low') or b.get('lowPrice') or b.get('l') or 0
            c = b.get('close') or b.get('closePrice') or b.get('c') or 0
            v = b.get('volume') or b.get('v') or 0

            new_rows.append({
                'timestamp': ts,
                'open': float(o),
                'high': float(h),
                'low': float(l),
                'close': float(c),
                'volume': float(v),
                'price': float(c)
            })
            
        with self.lock:
            history_df = pd.DataFrame(new_rows)
            
            # FIX: Explicitly strip timezone from timestamp BEFORE astype()
            if 'timestamp' in history_df.columns:
                history_df.loc[:, 'timestamp'] = pd.to_datetime(history_df['timestamp']).dt.tz_localize(None)

            # Avoid overwriting or racing with the currently forming live bar
            if self.current_bar is not None and 'timestamp' in history_df.columns:
                cur_start = pd.Timestamp(self.current_bar['start_time']).tz_localize(None)
                history_df = history_df[history_df['timestamp'] < cur_start]

            # FIX: Ensure dtypes match to avoid FutureWarnings during concat
            type_map = {col: self.bars[col].dtype for col in self.bars.columns if col in history_df.columns}
            if type_map and not history_df.empty:
                history_df = history_df.astype(type_map)
            
            existing_ts = set(self.bars['timestamp']) if not self.bars.empty else set()
            new_unique_count = sum(1 for ts in history_df['timestamp'] if ts not in existing_ts) if not history_df.empty else 0

            # Combine with existing (if any) and trim
            if not history_df.empty:
                self.bars = pd.concat([self.bars, history_df], ignore_index=True)
                self.bars = self.bars.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp').reset_index(drop=True)
                if len(self.bars) > self.max_bars:
                    self.bars = self.bars.iloc[-self.max_bars:].reset_index(drop=True)

            self.total_bars_generated = max(
                getattr(self, "total_bars_generated", 0) + max(new_unique_count, 0),
                len(self.bars),
            )
                
        self.logger.info(f"Pre-loaded {len(new_rows)} historical bars into state.")
