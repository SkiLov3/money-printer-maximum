import logging
import os
from datetime import datetime, time, timedelta

import pytz


class NewsAgent:
    """
    Manages news stand-down windows and US market hours.
    Relying on a robust recurring schedule + fixed major dates 
    to avoid premium API costs.
    """
    def __init__(self):
        self.logger = logging.getLogger("NewsAgent")
        self.stand_down_pre = int(os.getenv("NEWS_STAND_DOWN_PRE", "10"))
        self.stand_down_post = int(os.getenv("NEWS_STAND_DOWN_POST", "20"))
        self._logged_standdowns = set()
        
        # High Impact Recurring Calendar (ET Timezone)
        # Format: (DayOfWeek, Time_ET, EventName)
        # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
        # Only truly weekly events belong here; monthly/FOMC releases live in fixed_events.
        self.weekly_schedule = [
            (3, "08:30", "Initial Jobless Claims"),  # Every Thursday
        ]
        
        # Major High-Impact Economic Releases (ET timezone)
        self.fixed_events = [
            # September 2026
            (2026, 9, 4,   "08:30", "US Non-Farm Payrolls (NFP)"),
            (2026, 9, 10,  "08:30", "US CPI Inflation"),
            (2026, 9, 16,  "14:00", "FOMC Rate Decision"),
            # October 2026
            (2026, 10, 1,  "10:00", "US ISM Manufacturing PMI"),
            (2026, 10, 2,  "08:30", "US Non-Farm Payrolls (NFP)"),
            (2026, 10, 6,  "10:00", "US JOLTS Job Openings"),
            (2026, 10, 7,  "14:00", "FOMC Meeting Minutes"),
            (2026, 10, 8,  "08:30", "Initial Jobless Claims"),
            (2026, 10, 13, "08:30", "US CPI Inflation"),
            (2026, 10, 14, "08:30", "US PPI Inflation"),
            (2026, 10, 15, "08:30", "Initial Jobless Claims"),
            (2026, 10, 22, "08:30", "Initial Jobless Claims"),
            (2026, 10, 29, "08:30", "Initial Jobless Claims"),
            # November 2026
            (2026, 11, 4,  "14:00", "FOMC Rate Decision"),
            (2026, 11, 6,  "08:30", "US Non-Farm Payrolls (NFP)"),
            (2026, 11, 12, "08:30", "US CPI Inflation"),
            (2026, 11, 13, "08:30", "US PPI Inflation"),
            (2026, 11, 25, "14:00", "FOMC Meeting Minutes"),
            # December 2026
            (2026, 12, 4,  "08:30", "US Non-Farm Payrolls (NFP)"),
            (2026, 12, 10, "08:30", "US CPI Inflation"),
            (2026, 12, 11, "08:30", "US PPI Inflation"),
            (2026, 12, 16, "14:00", "FOMC Rate Decision"),
        ]
        
        self.logger.info("NewsAgent initialized with Local Protection Schedule (Free Tier).")

    def is_stand_down(self, dt: datetime) -> bool:
        """Returns True if within a news window."""
        et_tz = pytz.timezone("US/Eastern")
        if hasattr(dt, 'to_pydatetime'): dt = dt.to_pydatetime()
        
        # FIX: If naive, assume UTC (matching DataAgent standard)
        if dt.tzinfo is None: 
            dt = pytz.utc.localize(dt)
            
        dt_et = dt.astimezone(et_tz)
        
        # 1. Check Fixed Events 
        for y, m, d, t_str, name in self.fixed_events:
            event_dt = et_tz.localize(datetime(y, m, d, int(t_str[:2]), int(t_str[3:])))  # noqa: DTZ001
            if self._in_window(dt_et, event_dt):
                self._log_standdown_once(f"{name}:{event_dt.isoformat()}", f"News Stand-Down: {name} at {t_str}")
                return True
                
        # 2. Check Weekly Recurring
        weekday = dt_et.weekday()
        for s_day, s_time, s_name in self.weekly_schedule:
            if weekday == s_day:
                h, m = map(int, s_time.split(":"))
                event_dt = dt_et.replace(hour=h, minute=m, second=0, microsecond=0)
                if self._in_window(dt_et, event_dt):
                    self._log_standdown_once(
                        f"{s_name}:{event_dt.date().isoformat()}:{s_time}",
                        f"Weekly News Stand-Down: {s_name} at {s_time}",
                    )
                    return True
        return False

    def _log_standdown_once(self, key: str, message: str) -> None:
        if key in self._logged_standdowns:
            return
        self._logged_standdowns.add(key)
        self.logger.warning(message)

    def is_news_approaching(self, dt: datetime, lead_minutes=2) -> str:
        """Checks if a news event is imminent."""
        et_tz = pytz.timezone("US/Eastern")
        if hasattr(dt, 'to_pydatetime'): dt = dt.to_pydatetime()
        
        # FIX: If naive, assume UTC
        if dt.tzinfo is None: dt = pytz.utc.localize(dt)
        
        dt_et = dt.astimezone(et_tz)

        # Fixed
        for y, m, d, t_str, name in self.fixed_events:
            event_dt = et_tz.localize(datetime(y, m, d, int(t_str[:2]), int(t_str[3:])))  # noqa: DTZ001
            if event_dt - timedelta(minutes=lead_minutes) <= dt_et < event_dt:
                return name
        # Weekly
        weekday = dt_et.weekday()
        for s_day, s_time, s_name in self.weekly_schedule:
            if weekday == s_day:
                h, m = map(int, s_time.split(":"))
                event_dt = dt_et.replace(hour=h, minute=m, second=0, microsecond=0)
                if event_dt - timedelta(minutes=lead_minutes) <= dt_et < event_dt:
                    return s_name
        return None

    def _in_window(self, current: datetime, event: datetime) -> bool:
        start = event - timedelta(minutes=self.stand_down_pre)
        end = event + timedelta(minutes=self.stand_down_post)
        return start <= current <= end

    def is_market_hours(self, dt: datetime) -> bool:
        """
        ACTIVE TRADING WINDOW: Full Globex Hours (Sun 18:00 ET - Fri 17:00 ET).
        
        The NewsAgent now serves as a high-level Exchange Gate. It only blocks 
        the daily settlement hour (17:00-18:00 ET) and the weekend closure.
        Individual strategies (like PriceActionBSStrategy) are now responsible 
        for managing their own intraday session filters (Asian, NY, Dead Zones).
        """
        et_tz = pytz.timezone("US/Eastern")
        if hasattr(dt, 'to_pydatetime'): dt = dt.to_pydatetime()

        # FIX: If naive, assume UTC
        if dt.tzinfo is None: dt = pytz.utc.localize(dt)

        dt_et = dt.astimezone(et_tz)
        weekday = dt_et.weekday()  # 0=Mon, 6=Sun
        time_et = dt_et.time()

        # 1. Weekend Closure
        # Friday 17:00 ET through Sunday 18:00 ET
        if weekday == 4 and time_et >= time(17, 0): # Friday after 5pm
            return False
        if weekday == 5: # All day Saturday
            return False
        if weekday == 6 and time_et < time(18, 0): # Sunday before 6pm
            return False

        # 2. Daily Maintenance Closure
        # Monday-Thursday 17:00 ET to 18:00 ET
        if weekday in (0, 1, 2, 3) and time(17, 0) <= time_et < time(18, 0):
            return False

        return True


    def is_near_close(self, dt: datetime, lead_minutes=20) -> bool:
        """
        Check if we are within the Topstep/Globex safety window.
        Topstep deadline is 15:50 CT (4:50 PM ET).
        Globex hard close is 17:00 ET.
        """
        et_tz = pytz.timezone("US/Eastern")
        if hasattr(dt, 'to_pydatetime'): dt = dt.to_pydatetime()
        if dt.tzinfo is None: dt = pytz.utc.localize(dt)
        dt_et = dt.astimezone(et_tz)
        
        # [EXPERT FIX] Daily Topstep Dead Zone (Settlement and Rollover)
        # We MUST be flat by 16:50 ET (4:50 PM ET).
        # We can resume trading at 18:00 ET (6:00 PM ET).
        ts_limit = dt_et.replace(hour=16, minute=50, second=0, microsecond=0)
        ts_reset = dt_et.replace(hour=18, minute=0, second=0, microsecond=0)
        
        # Lockout active between 4:50 PM ET and 6:00 PM ET
        if ts_limit <= dt_et < ts_reset:
            return True
            
        # Or within lead_minutes before the 4:50 PM ET deadline:
        safety_start = ts_limit - timedelta(minutes=lead_minutes)
        if safety_start <= dt_et < ts_limit:
            return True
            
        # Friday end-of-week hard stop after 16:50 ET
        if dt_et.weekday() == 4 and dt_et >= ts_limit:
            return True

        return False
