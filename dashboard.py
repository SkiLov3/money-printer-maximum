#!/usr/bin/env python3
"""
TopstepX Trading Swarm — Live Dashboard
Run alongside the bot: python dashboard.py
Then open: http://localhost:8765  (or http://<droplet-ip>:8765 remotely)

ENV VARS:
  DASHBOARD_PORT      Port to bind (default: 8765)
  DASHBOARD_HOST      Host to bind (default: 0.0.0.0 — accessible remotely)
  DASHBOARD_PASSWORD  Set this on a public droplet! Browser will prompt for password.
"""
import base64
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("dashboard")

STATUS_FILE = Path("logs/bot_status.json")
try:
    from src.version import VERSION
except ImportError:
    VERSION = "2.5.3"
PORT        = int(os.getenv("DASHBOARD_PORT", 8765))
HOST        = os.getenv("DASHBOARD_HOST", "127.0.0.1")
PASSWORD    = os.getenv("DASHBOARD_PASSWORD", "")       # empty = no auth (local dev)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"/>
<meta http-equiv="Pragma" content="no-cache"/>
<meta http-equiv="Expires" content="0"/>
<title>TopstepX Swarm — Live Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0a0d14;
    --surface:  #111827;
    --surface2: #1a2235;
    --border:   #1e2d45;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --green:    #10b981;
    --red:      #ef4444;
    --amber:    #f59e0b;
    --blue:     #3b82f6;
    --red-hover: #dc2626;
  }

  body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; overflow-x: hidden; }

  /* HEADER */
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 32px; background: var(--surface);
    border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 100;
  }
  .logo { display: flex; align-items: center; gap: 10px; }
  .logo-icon {
    width: 36px; height: 36px; border-radius: 8px;
    background: linear-gradient(135deg, var(--green), var(--blue));
    display: flex; align-items: center; justify-content: center; font-size: 18px;
  }
  .logo-text { font-size: 16px; font-weight: 700; letter-spacing: -0.3px; }
  .logo-sub  { font-size: 11px; color: var(--muted); margin-top: 1px; }
  .header-badges { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
  .badge {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
    border: 1px solid transparent; transition: all .3s;
  }
  .badge-green { background: rgba(16,185,129,.15); border-color: rgba(16,185,129,.4); color: var(--green); }
  .badge-red   { background: rgba(239,68,68,.15);  border-color: rgba(239,68,68,.4);  color: var(--red); }
  .badge-amber { background: rgba(245,158,11,.15); border-color: rgba(245,158,11,.4); color: var(--amber); }
  .badge-blue  { background: rgba(59,130,246,.15); border-color: rgba(59,130,246,.4); color: var(--blue); }
  .badge-muted { background: rgba(100,116,139,.15);border-color: rgba(100,116,139,.4);color: var(--muted); }
  .dot { width: 7px; height: 7px; border-radius: 50%; }
  .dot-green { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse 2s infinite; }
  .dot-red   { background: var(--red);   box-shadow: 0 0 6px var(--red); }
  .dot-amber { background: var(--amber); box-shadow: 0 0 6px var(--amber); animation: pulse 1.5s infinite; }
  .dot-muted { background: var(--muted); }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.4; } }
  .last-update { font-size: 11px; color: var(--muted); font-family: 'JetBrains Mono', monospace; }

  /* MAIN */
  main { padding: 28px 32px; max-width: 1600px; margin: 0 auto; }
  .tabs { display:flex; gap:8px; margin-bottom:20px; border-bottom:1px solid var(--border); }
  .tab-btn { background: transparent; color: var(--muted); border: none; border-bottom: 2px solid transparent; padding: 0 4px 12px; font-weight: 700; cursor: pointer; }
  .tab-btn.active { color: var(--text); border-bottom-color: var(--blue); }
  .view { display:none; }
  .view.active { display:block; }

  /* STAT CARDS */
  .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  @media (max-width: 1100px) { .cards { grid-template-columns: repeat(2,1fr); } }
  @media (max-width:  600px)  { .cards { grid-template-columns: 1fr; } }
  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 22px 24px; position: relative; overflow: hidden; transition: border-color .3s, box-shadow .3s;
  }
  .card:hover { border-color: #2d4165; box-shadow: 0 4px 32px rgba(0,0,0,.3); }
  .card::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at top right, rgba(59,130,246,.05), transparent 60%);
    pointer-events: none;
  }
  .card-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 10px; }
  .card-value { font-size: 32px; font-weight: 700; line-height: 1; letter-spacing: -1px; font-family: 'JetBrains Mono', monospace; }
  .card-sub   { font-size: 12px; color: var(--muted); margin-top: 8px; }
  .card-icon  { position: absolute; right: 20px; top: 20px; font-size: 28px; opacity: .15; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }
  .neutral  { color: var(--text); }

  /* DRAWDOWN BAR */
  .dd-bar-wrap { margin-top: 12px; }
  .dd-bar-track { height: 6px; border-radius: 99px; background: var(--surface2); overflow: hidden; }
  .dd-bar-fill  { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--green), var(--amber), var(--red)); transition: width .8s ease; }
  .dd-labels { display: flex; justify-content: space-between; margin-top: 4px; font-size: 10px; color: var(--muted); }

  /* BOTTOM GRID */
  .bottom-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  @media (max-width: 1450px) { .bottom-grid { grid-template-columns: 1fr 1fr; } }
  @media (max-width: 950px) { .bottom-grid { grid-template-columns: 1fr; } }
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; }
  .panel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 16px; border-bottom: 1px solid var(--border); font-size: 13px; font-weight: 600;
  }
  .panel-header .count { background: var(--surface2); border-radius: 99px; padding: 2px 10px; font-size: 11px; color: var(--muted); }

  /* POSITIONS & TRADE LOG */
  .pos-empty { padding: 48px 20px; text-align: center; color: var(--muted); font-size: 13px; }
  .pos-empty-icon { font-size: 36px; margin-bottom: 8px; }
  .panel-table-wrap { overflow-x: auto; overflow-y: hidden; -webkit-overflow-scrolling: touch; scrollbar-width: thin; width: 100%; box-sizing: border-box; }
  .grid-row { display: grid; padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 12px; align-items: center; transition: background .2s; box-sizing: border-box; }
  .grid-row:hover { background: var(--surface2); }
  .grid-row:last-child { border-bottom: none; }
  .grid-hdr { display: grid; padding: 8px 12px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); border-bottom: 1px solid var(--border); background: var(--surface2); box-sizing: border-box; align-items: center; }
  .pos-cols { display: grid; grid-template-columns: minmax(80px, 1.2fr) minmax(60px, 0.8fr) minmax(40px, 0.6fr) minmax(80px, 1fr) minmax(80px, 1fr) minmax(90px, 1.1fr); gap: 8px; width: 100%; box-sizing: border-box; }
  .trade-cols { display: grid; grid-template-columns: minmax(110px, 1.2fr) minmax(90px, 1fr) minmax(80px, 1fr) minmax(60px, 0.8fr) minmax(70px, 0.9fr) minmax(90px, 1.1fr); gap: 8px; width: 100%; align-items: center; box-sizing: border-box; }
  .badge-session { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; white-space: nowrap; line-height: 1.3; }
  .session-rth { background: rgba(59, 130, 246, 0.18); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.35); }
  .session-pre { background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
  .session-asian { background: rgba(168, 85, 247, 0.18); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35); }
  .session-london { background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
  .session-globex { background: rgba(107, 114, 128, 0.18); color: #9ca3af; border: 1px solid rgba(107, 114, 128, 0.35); }

  .grid-cell-truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
  .side-buy  { color: var(--green); font-weight: 700; }
  .side-sell { color: var(--red);   font-weight: 700; }
  .mono { font-family: 'JetBrains Mono', monospace; font-size: 11px; }
  .trade-pnl-pos { color: var(--green); font-weight: 600; font-family: 'JetBrains Mono', monospace; white-space: nowrap; text-align: right; }
  .trade-pnl-neg { color: var(--red);   font-weight: 600; font-family: 'JetBrains Mono', monospace; white-space: nowrap; text-align: right; }
  .no-trades { padding: 48px 20px; text-align: center; color: var(--muted); font-size: 13px; }

  /* CHART */
  .chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; margin-top: 20px; }
  .chart-canvas-wrap { padding: 8px 16px 16px; height: 200px; }
  .chart-toolbar { display:flex; flex-wrap:wrap; gap:12px; align-items:center; padding:14px 16px; background: var(--surface); border:1px solid var(--border); border-radius: 16px; margin-bottom: 14px; }
  .chart-select, .frame-btn, .chart-action-btn, .chart-icon-btn { background:#0f172a; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:8px 10px; font-weight:700; }
  .frame-btn, .chart-action-btn, .chart-icon-btn { cursor:pointer; font-size:12px; }
  .frame-btn.active { border-color: var(--blue); color: var(--blue); background: rgba(59,130,246,.12); }
  .chart-actions { display:flex; gap:8px; align-items:center; }
  .chart-action-btn:hover, .chart-icon-btn:hover { border-color: var(--blue); color: var(--blue); }
  .chart-zoom { display:flex; gap:6px; align-items:center; }
  .chart-zoom-select { min-width:84px; }
  .chart-icon-btn { width:36px; height:36px; padding:0; display:inline-flex; align-items:center; justify-content:center; }
  .chart-icon-btn svg { width:18px; height:18px; stroke:currentColor; fill:none; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }
  .chart-grid { display:grid; grid-template-columns: minmax(0, 1fr) 360px; gap:16px; }
  @media (max-width: 1100px) { .chart-grid { grid-template-columns: 1fr; } }
  .tv-chart { height: 680px; background: var(--surface); border:1px solid var(--border); border-radius:16px; overflow:hidden; touch-action:none; }
  .diagnostics-body { max-height: 680px; overflow:auto; }
  .event-row { padding:10px 12px; border-bottom:1px solid rgba(255,255,255,.05); display:grid; gap:4px; }
  .event-row:last-child { border-bottom:none; }
  .event-title { display:grid; grid-template-columns:minmax(0, 1fr) auto; align-items:center; gap:10px; font-size:12px; font-weight:700; }
  .event-title-text { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .event-meta { display:flex; align-items:center; justify-content:flex-end; gap:6px; min-width:0; }
  .event-time { color:var(--muted); font-family:'JetBrains Mono', monospace; font-size:10px; white-space:nowrap; }
  .event-detail { color: var(--muted); font-size:11px; line-height:1.4; }
  .ops-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
  @media (max-width: 900px) { .ops-grid { grid-template-columns: 1fr; } }
  .ops-body { padding: 16px 20px; font-size: 13px; }
  .ops-stack { display: grid; gap: 12px; }
  .ops-item { background: var(--surface2); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }
  .ops-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 6px; }
  .ops-title { font-weight: 700; color: var(--text); }
  .ops-sub { color: var(--muted); font-size: 11px; line-height: 1.5; }
  .ops-pill { padding: 2px 8px; border-radius: 999px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
  .pill-green { background: rgba(16,185,129,.15); color: var(--green); }
  .pill-red { background: rgba(239,68,68,.15); color: var(--red); }
  .pill-amber { background: rgba(245,158,11,.15); color: var(--amber); }
  .pill-blue { background: rgba(59,130,246,.15); color: var(--blue); }
  .kv { display: flex; justify-content: space-between; gap: 12px; font-size: 12px; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,.04); }
  .kv:last-child { border-bottom: none; }
  .kv-label { color: var(--muted); }
  .kv-value { color: var(--text); font-family: 'JetBrains Mono', monospace; text-align: right; }
  .sat-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 10px; }
  .sat-table th { background: var(--surface2); color: var(--muted); padding: 8px 10px; text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); }
  .sat-table td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
  .sat-table tr:hover { background: rgba(255,255,255,.02); }
  .sat-winner-row { background: rgba(16,185,129,.08) !important; border-left: 3px solid var(--green); }

  /* OFFLINE */
  #offline { display: none; position: fixed; inset: 0; z-index: 9999; background: rgba(10,13,20,.92); align-items: center; justify-content: center; flex-direction: column; gap: 12px; }
  #offline.show { display: flex; }
  #offline h2 { color: var(--red); font-size: 22px; }
  #offline p  { color: var(--muted); font-size: 14px; }
</style>
</head>
<body>

<div id="offline">
  <div style="font-size:48px">⚠️</div>
  <h2>Bot Status File Not Found</h2>
  <p>Start the bot with <code style="color:var(--amber)">python main.py</code> — dashboard will connect automatically.</p>
</div>

<header>
  <div class="logo">
    <div class="logo-icon">⚡</div>
    <div>
      <div class="logo-text">TopstepX Swarm <span style="font-size: 11px; font-weight: normal; color: var(--muted); vertical-align: middle; margin-left: 4px;" id="dashboard-version">v__APP_VERSION__</span></div>
      <div class="logo-sub">Live Monitoring Dashboard</div>
    </div>
  </div>
  <div class="header-badges">
    <div class="badge" id="badge-running"><div class="dot dot-muted"></div>Offline</div>
    <div class="badge" style="font-size: 14px; padding: 6px 16px; border-radius: 6px;" id="badge-mode">--</div>
    <div class="badge" id="badge-market">--</div>
    <div class="badge" id="badge-news">--</div>
    <div class="badge" id="badge-signalr">--</div>
    <div class="badge" id="badge-sat-opt"><div class="dot dot-blue"></div>SAT OPT: IDLE</div>
    <button id="flatten-btn" onclick="confirmFlatten()" style="background: var(--red); color: white; border: none; padding: 6px 16px; border-radius: 6px; font-weight: 700; cursor: pointer; display: flex; align-items: center; gap: 8px; font-size: 13px; transition: background .2s, transform .1s;">
      <span style="font-size:16px">🔥</span> FLATTEN ALL
    </button>
  </div>
  <div class="last-update" id="last-update">Connecting…</div>
</header>

<main>
  <div class="tabs">
    <button class="tab-btn active" id="overview-tab" onclick="showTab('overview')">Overview</button>
    <button class="tab-btn" id="charts-tab" onclick="showTab('charts')">Charts</button>
    <button class="tab-btn" id="optimizer-tab" onclick="showTab('optimizer')">🤖 Saturday Optimizer</button>
  </div>

  <section id="overview-view" class="view active">
  <div class="cards">
    <div class="card" style="border-left: 4px solid var(--blue);">
      <div class="card-icon">💎</div>
      <div class="card-label">Live Open P&L</div>
      <div class="card-value" id="card-unrealized">--</div>
      <div class="card-sub" id="card-unrealized-sub">Floating P&L (Unrealized)</div>
    </div>
    <div class="card">
      <div class="card-icon">💰</div>
      <div class="card-label">Session P&L</div>
      <div class="card-value" id="card-pnl">--</div>
      <div class="card-sub" id="card-pnl-sub">Realized + Unrealized: --</div>
    </div>
    <div class="card">
      <div class="card-icon">🏦</div>
      <div class="card-label">Account Equity / Balance</div>
      <div class="card-value neutral" id="card-balance">--</div>
      <div class="card-sub" id="card-balance-sub">Cash: -- | Peak: --</div>
    </div>
    <div class="card">
      <div class="card-icon">📉</div>
      <div class="card-label">Drawdown from Peak (MLL)</div>
      <div class="card-value" id="card-dd">--</div>
      <div class="card-sub" id="card-dd-sub">Floor: -- | Buffer: --</div>
      <div class="dd-bar-wrap">
        <div class="dd-bar-track"><div class="dd-bar-fill" id="dd-bar" style="width:0%"></div></div>
        <div class="dd-labels"><span>$0</span><span id="dd-limit">--</span></div>
      </div>
    </div>
    <div class="card">
      <div class="card-icon">📊</div>
      <div class="card-label">Open Positions</div>
      <div class="card-value neutral" id="card-pos">0</div>
      <div class="card-sub" id="card-sym">No active symbols</div>
    </div>
  </div>

  <!-- ACTIVE POSITIONS (FULL WIDTH) -->
  <div class="panel" style="margin-bottom: 20px;">
    <div class="panel-header">
      <div style="display:flex; align-items:center; gap:8px;">
        <span>🟢 Active Positions</span>
        <span class="count" id="pos-count">0</span>
      </div>
      <div style="font-size: 11px; color: var(--muted);">Live Market Positions</div>
    </div>
    <div id="positions-body" class="panel-table-wrap">
      <div class="pos-empty"><div class="pos-empty-icon">📋</div>No open positions</div>
    </div>
  </div>

  <!-- STRATEGY COMMAND CENTER -->
  <div class="panel" style="margin-bottom: 20px;">
    <div class="panel-header">
      <div style="display:flex; align-items:center; gap:8px;">
        <span>🧠 Live Strategy Intel (Active Setups)</span>
      </div>
      <div style="font-size: 11px; color: var(--muted);">Regime, Momentum & Volatility Filters</div>
    </div>
    <div id="strategy-body" style="padding: 16px 20px;">
      <div class="no-trades">Awaiting strategy initialization...</div>
    </div>
  </div>

  <!-- SATURDAY AUTONOMOUS OPTIMIZER WIDGET -->
  <div class="panel" style="margin-bottom: 20px;">
    <div class="panel-header">
      <div style="display:flex; align-items:center; gap:10px;">
        <span>🤖 Saturday Autonomous Optimizer (Qwen 3.5 122B)</span>
        <span class="ops-pill pill-blue" id="sat-phase-pill">IDLE</span>
      </div>
      <div style="display:flex; align-items:center; gap:10px;">
        <span class="count" id="sat-last-run" style="font-family: var(--mono); font-size: 11px;">LAST: -- | NEXT: Sat 08:00 ET</span>
        <button id="sat-run-btn" onclick="triggerSaturdayOptimizer()" style="background: linear-gradient(135deg, var(--green), var(--blue)); color: white; border: none; padding: 6px 14px; border-radius: 6px; font-weight: 700; cursor: pointer; font-size: 11px; transition: opacity .2s;">⚡ Run Dry-Run</button>
      </div>
    </div>
    <div class="ops-body" id="sat-body" style="padding:16px 20px;">
      <div class="no-trades">Optimizer is idle. It runs autonomously on Saturday at 08:00 ET when flat, or click "Run Dry-Run" to trigger an audit now.</div>
    </div>
  </div>

  <!-- CONSOLIDATED CLOSED TRADES LOG -->
  <div class="panel" style="margin-bottom: 20px;">
    <div class="panel-header">
      <div style="display:flex; align-items:center; gap:8px;">
        <span>📜 Trade Execution History</span>
        <span class="count" id="trade-count">0 trades</span>
      </div>
      <div style="font-size: 11px; color: var(--muted);">Closed Session & Recent Fills</div>
    </div>
    <div id="trades-body" class="panel-table-wrap">
      <div class="no-trades">No historical trades stored</div>
    </div>
  </div>
  </section>

  <section id="charts-view" class="view">
    <div class="chart-toolbar">
      <select class="chart-select" id="chart-symbol" onchange="refreshChart(true)"></select>
      <div id="frame-buttons" style="display:flex; flex-wrap:wrap; gap:8px;"></div>
      <div class="chart-actions">
        <button class="chart-action-btn" onclick="fitChartContent()" title="Fit all loaded bars">Fit</button>
        <button class="chart-action-btn" onclick="jumpChartLatest()" title="Jump to latest bar">Latest</button>
      </div>
      <div class="chart-zoom">
        <button class="chart-icon-btn" onclick="stepChartZoom(-1)" aria-label="Zoom out price action" title="Zoom out price action">
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10" cy="10" r="6"></circle><path d="m14.5 14.5 5 5"></path><path d="M7.5 10h5"></path></svg>
        </button>
        <select class="chart-select chart-zoom-select" id="chart-zoom" onchange="setChartZoomPercent(this.value)" title="Price action zoom percentage">
          <option value="25">25%</option>
          <option value="50">50%</option>
          <option value="75">75%</option>
          <option value="100" selected>100%</option>
          <option value="125">125%</option>
          <option value="150">150%</option>
          <option value="200">200%</option>
          <option value="300">300%</option>
          <option value="400">400%</option>
          <option value="600">600%</option>
          <option value="800">800%</option>
          <option value="1000">1000%</option>
          <option value="2000">2000%</option>
        </select>
        <button class="chart-icon-btn" onclick="stepChartZoom(1)" aria-label="Zoom in price action" title="Zoom in price action">
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10" cy="10" r="6"></circle><path d="m14.5 14.5 5 5"></path><path d="M7.5 10h5"></path><path d="M10 7.5v5"></path></svg>
        </button>
      </div>
      <span class="last-update" id="chart-status">Chart waiting for bot data</span>
    </div>
    <div class="chart-grid">
      <div class="tv-chart" id="tv-chart"></div>
      <div class="panel">
        <div class="panel-header"><span>Price Action Diagnostics</span><span class="count" id="diag-count">0</span></div>
        <div class="ops-body diagnostics-body" id="chart-diagnostics"><div class="no-trades">Open the Charts tab to load visual diagnostics.</div></div>
      </div>
    </div>
  </section>

  <section id="optimizer-view" class="view">
    <div class="panel" style="margin-bottom: 20px;">
      <div class="panel-header">
        <div style="display:flex; align-items:center; gap:10px;">
          <span>🤖 Saturday Autonomous Optimizer (Qwen 3.5 122B)</span>
          <span class="ops-pill pill-blue" id="sat-phase-pill-tab">IDLE</span>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
          <span class="count" id="sat-last-run-tab" style="font-family: var(--mono); font-size: 11px;">LAST: -- | NEXT: Sat 08:00 ET</span>
          <button id="sat-run-btn-tab" onclick="triggerSaturdayOptimizer()" style="background: linear-gradient(135deg, var(--green), var(--blue)); color: white; border: none; padding: 6px 14px; border-radius: 6px; font-weight: 700; cursor: pointer; font-size: 11px; transition: opacity .2s;">⚡ Run Dry-Run</button>
        </div>
      </div>
      <div class="ops-body" id="sat-body-tab" style="padding:16px 20px;">
        <div class="no-trades">Optimizer is idle. It runs autonomously on Saturday at 08:00 ET when flat, or click "Run Dry-Run" to trigger an audit now.</div>
      </div>
    </div>
  </section>
</main>

<script>
const $ = id => document.getElementById(id);
const esc = s => (s === null || s === undefined) ? '' : String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
const fmt$ = v => (v === null || v === undefined || isNaN(v)) ? '--' : (Number(v) < 0 ? '-$' : '+$') + Math.abs(Number(v)).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
const fmtB = v => (v === null || v === undefined || isNaN(v)) ? '--' : (Number(v) < 0 ? '-$' : '$') + Math.abs(Number(v)).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
const fmtPx = v => (v === null || v === undefined) ? '--' : Number(v).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
const cleanTradeType = t => {
  if (!t) return '--';
  const s = String(t).toLowerCase();
  if (s === 'sl' || s === 'stop' || s.includes('stop_loss')) return 'Stop';
  if (s === 'tp' || s === 'target' || s.includes('take_profit')) return 'Target';
  if (s.includes('reconcile') || s.includes('flip')) return 'Reconcile';
  if (s.includes('flatten')) return 'Flatten';
  return s.charAt(0).toUpperCase() + s.slice(1);
};
const sessionBadge = sess => {
  if (!sess) return '<span class="badge-session session-globex">--</span>';
  const s = String(sess).trim().toUpperCase();
  let cls = 'session-globex';
  if (s.includes('RTH') || s === 'NY') cls = 'session-rth';
  else if (s.includes('PRE')) cls = 'session-pre';
  else if (s.includes('ASIAN') || s.includes('ASIA')) cls = 'session-asian';
  else if (s.includes('LONDON') || s.includes('LDN')) cls = 'session-london';
  return `<span class="badge-session ${cls}">${esc(s)}</span>`;
};

const chartFrames = ['1m','5m','15m','30m','1h','4h','1d'];
const chartZoomPercents = [25, 50, 75, 100, 125, 150, 200, 300, 400, 600, 800, 1000, 2000];
let activeTab = 'overview';
let activeChartFrame = '1m';
let activeChartSymbol = '';
let tvChart = null, candleSeries = null;
let overlaySeries = [], priceLines = [];
let chartContextKey = '';
let chartHadData = false;
let chartLastBarCount = 0;
let chartZoomBaseBars = 0;
let activeChartZoomPercent = 100;
let chartLastBarTime = 0;
let chartFullRefreshInFlight = false;
let chartLiveInFlight = false;
let chartLastFullRefreshAt = 0;
let chartUserInteractingUntil = 0;
const chartOverlayRefreshMs = 10000;

function setBadge(id, cls, dotCls, text) {
  const el = $(id); el.className = 'badge ' + cls;
  el.innerHTML = `<div class="dot ${dotCls}"></div>${text}`;
}

function showTab(name) {
  activeTab = name;
  $('overview-tab')?.classList.toggle('active', name === 'overview');
  $('charts-tab')?.classList.toggle('active', name === 'charts');
  $('optimizer-tab')?.classList.toggle('active', name === 'optimizer');
  $('overview-view')?.classList.toggle('active', name === 'overview');
  $('charts-view')?.classList.toggle('active', name === 'charts');
  $('optimizer-view')?.classList.toggle('active', name === 'optimizer');
  if (name === 'charts') {
    ensureChart();
    refreshChart(true);
    setTimeout(() => tvChart && tvChart.resize($('tv-chart').clientWidth, $('tv-chart').clientHeight), 50);
  }
}

function setupFrameButtons() {
  const wrap = $('frame-buttons');
  if (wrap.dataset.ready === '1') return;
  wrap.dataset.ready = '1';
  wrap.innerHTML = chartFrames.map(frame => `<button class="frame-btn ${frame === activeChartFrame ? 'active' : ''}" id="frame-${frame}" onclick="setChartFrame('${frame}')">${frame.toUpperCase()}</button>`).join('');
}

function setChartFrame(frame) {
  activeChartFrame = frame;
  for (const item of chartFrames) $('frame-' + item)?.classList.toggle('active', item === frame);
  refreshChart(true);
}

function updateChartSymbols(symbols) {
  const select = $('chart-symbol');
  if (!Array.isArray(symbols) || symbols.length === 0) return;
  const current = select.value || activeChartSymbol || symbols[0];
  const nextHtml = symbols.map(sym => `<option value="${sym}">${sym}</option>`).join('');
  if (select.innerHTML !== nextHtml) select.innerHTML = nextHtml;
  activeChartSymbol = symbols.includes(current) ? current : symbols[0];
  select.value = activeChartSymbol;
  setupFrameButtons();
}

function ensureChart() {
  setupFrameButtons();
  if (tvChart || !window.LightweightCharts) return;
  tvChart = LightweightCharts.createChart($('tv-chart'), {
    layout: { background: { color: '#111827' }, textColor: '#cbd5e1' },
    grid: { vertLines: { color: 'rgba(148,163,184,.08)' }, horzLines: { color: 'rgba(148,163,184,.08)' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
    handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    kineticScroll: { mouse: true, touch: true },
    rightPriceScale: { borderColor: '#334155', autoScale: true, scaleMargins: { top: 0.12, bottom: 0.12 } },
    timeScale: { borderColor: '#334155', timeVisible: true, secondsVisible: false, rightOffset: 8, barSpacing: 8, minBarSpacing: 0.1, lockVisibleTimeRangeOnResize: true, fixLeftEdge: false, fixRightEdge: false },
  });
  candleSeries = tvChart.addCandlestickSeries({
    upColor: '#10b981',
    downColor: '#ef4444',
    borderUpColor: '#10b981',
    borderDownColor: '#ef4444',
    wickUpColor: '#10b981',
    wickDownColor: '#ef4444',
  });
  window.addEventListener('resize', () => {
    if (tvChart) tvChart.resize($('tv-chart').clientWidth, $('tv-chart').clientHeight);
  });
  const chartEl = $('tv-chart');
  const noteInteraction = () => { chartUserInteractingUntil = Date.now() + 1500; };
  ['wheel', 'pointerdown', 'pointermove', 'touchstart', 'touchmove'].forEach(eventName => {
    chartEl.addEventListener(eventName, noteInteraction, { passive: true });
  });
}

function applyPriceAutoscale() {
  try { tvChart.priceScale('right').applyOptions({ autoScale: true }); } catch(e) {}
}

function syncChartZoomSelect() {
  const select = $('chart-zoom');
  if (select) select.value = String(activeChartZoomPercent);
}

function fitChartContent() {
  if (!tvChart) return;
  activeChartZoomPercent = 100;
  chartZoomBaseBars = Math.max(chartLastBarCount, 1);
  syncChartZoomSelect();
  tvChart.timeScale().fitContent();
  applyPriceAutoscale();
}

function jumpChartLatest() {
  if (!tvChart) return;
  const scale = tvChart.timeScale();
  if (typeof scale.scrollToRealTime === 'function') {
    scale.scrollToRealTime();
  } else {
    scale.scrollToPosition(0, false);
  }
  applyPriceAutoscale();
}

function setChartZoomPercent(value, anchor='center') {
  const percent = Number(value);
  if (!Number.isFinite(percent) || percent <= 0) return;
  activeChartZoomPercent = percent;
  syncChartZoomSelect();
  if (!tvChart || !chartLastBarCount) return;
  const scale = tvChart.timeScale();
  const range = scale.getVisibleLogicalRange();
  const baseBars = Math.max(chartZoomBaseBars || chartLastBarCount, 1);
  const visibleBars = Math.max(2, Math.round(baseBars * (100 / percent)));
  const maxIndex = Math.max(chartLastBarCount - 1, 0);
  const rightPadding = 8;
  let center = anchor === 'latest' || !range
    ? maxIndex - (Math.min(visibleBars, chartLastBarCount) / 2) + (rightPadding / 2)
    : (Number(range.from) + Number(range.to)) / 2;
  if (!Number.isFinite(center)) center = maxIndex;
  let from = center - (visibleBars / 2);
  let to = center + (visibleBars / 2);
  const maxTo = maxIndex + rightPadding;
  if (to > maxTo) {
    from -= to - maxTo;
    to = maxTo;
  }
  if (from < 0) {
    to += -from;
    from = 0;
  }
  scale.setVisibleLogicalRange({ from, to });
  applyPriceAutoscale();
}

function stepChartZoom(direction) {
  const nearest = chartZoomPercents.reduce((best, value, index) => (
    Math.abs(value - activeChartZoomPercent) < Math.abs(chartZoomPercents[best] - activeChartZoomPercent) ? index : best
  ), 0);
  const nextIndex = Math.max(0, Math.min(chartZoomPercents.length - 1, nearest + direction));
  setChartZoomPercent(chartZoomPercents[nextIndex]);
}

function clearChartOverlays() {
  for (const series of overlaySeries) {
    try { tvChart.removeSeries(series); } catch(e) {}
  }
  overlaySeries = [];
  for (const line of priceLines) {
    try { candleSeries.removePriceLine(line); } catch(e) {}
  }
  priceLines = [];
}

function lineStyle(style) {
  if (style === 'dashed') return LightweightCharts.LineStyle.Dashed;
  if (style === 'dotted') return LightweightCharts.LineStyle.Dotted;
  return LightweightCharts.LineStyle.Solid;
}

function drawLine(points, color, style='solid', width=2) {
  if (!points || points.length < 2) return;
  const byTime = new Map();
  for (const p of points) {
    if (p && p.time && p.value !== null && p.value !== undefined && Number.isFinite(Number(p.value))) {
      byTime.set(Number(p.time), { time: Number(p.time), value: Number(p.value) });
    }
  }
  const cleanPoints = Array.from(byTime.values()).sort((a, b) => a.time - b.time);
  if (cleanPoints.length < 2) return;
  const series = tvChart.addLineSeries({ color, lineWidth: width, lineStyle: lineStyle(style), priceLineVisible: false, lastValueVisible: false });
  series.setData(cleanPoints);
  overlaySeries.push(series);
}

function markerFromItem(item) {
  const isBuy = item.side === 'Buy';
  const kind = item.kind || '';
  if (kind === 'swing_high') return { time: item.time, position: 'aboveBar', color: '#ef4444', shape: 'circle', text: item.label || 'SH' };
  if (kind === 'swing_low') return { time: item.time, position: 'belowBar', color: '#10b981', shape: 'circle', text: item.label || 'SL' };
  if (kind === 'failed') return { time: item.time, position: 'aboveBar', color: '#f59e0b', shape: 'square', text: item.setup_id || 'FAIL' };
  if (kind === 'pending') return { time: item.time, position: isBuy ? 'belowBar' : 'aboveBar', color: '#3b82f6', shape: isBuy ? 'arrowUp' : 'arrowDown', text: item.setup_id || 'ARM' };
  if (kind === 'signal' || kind === 'entry' || kind === 'active_position') return { time: item.time, position: isBuy ? 'belowBar' : 'aboveBar', color: isBuy ? '#10b981' : '#ef4444', shape: isBuy ? 'arrowUp' : 'arrowDown', text: item.setup_id || item.label || 'ENTRY' };
  if (kind === 'exit') return { time: item.time, position: 'aboveBar', color: (item.pnl ?? 0) >= 0 ? '#10b981' : '#ef4444', shape: 'square', text: `${item.order_type || 'EXIT'} ${fmt$(item.pnl ?? 0)}` };
  return null;
}

function formatChartTimestamp(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds <= 0) return '--';
  return new Date(seconds * 1000).toLocaleString(undefined, {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    timeZoneName: 'short',
  });
}

function diagnosticRows(overlays) {
  const eventRows = (overlays.events || []).map(item => ({...item, row_type: 'event'}));
  const swingRows = (overlays.markers || [])
    .filter(item => item.kind === 'swing_high' || item.kind === 'swing_low')
    .map(item => ({
      ...item,
      row_type: 'marker',
      label: item.kind === 'swing_high' ? 'Swing High' : 'Swing Low',
      setup_id: item.kind === 'swing_high' ? 'SH' : 'SL',
      detail: 'chart marker',
    }));
  return [...eventRows, ...swingRows]
    .filter(item => item.time)
    .sort((a, b) => Number(b.time || 0) - Number(a.time || 0))
    .slice(0, 60);
}

async function refreshChart(force=false) {
  if (activeTab !== 'charts' && !force) return;
  ensureChart();
  if (!tvChart || !candleSeries) {
    $('chart-status').textContent = 'Chart library unavailable';
    return;
  }
  if (chartFullRefreshInFlight) return;
  if (!force && Date.now() < chartUserInteractingUntil) return;
  const symbol = $('chart-symbol').value || activeChartSymbol;
  if (!symbol) return;
  chartFullRefreshInFlight = true;
  try {
    const scale = tvChart.timeScale();
    const requestedContextKey = `${symbol}|${activeChartFrame}`;
    const contextChanged = chartContextKey !== requestedContextKey;
    const previousRange = contextChanged || !chartHadData ? null : scale.getVisibleLogicalRange();
    const wasAtLatest = typeof scale.scrollPosition === 'function' ? Math.abs(scale.scrollPosition()) < 0.5 : false;
    const r = await fetch(`/api/chart?symbol=${encodeURIComponent(symbol)}&frame=${encodeURIComponent(activeChartFrame)}&limit=800&ts=${Date.now()}`);
    if (!r.ok) throw new Error(await r.text());
    const payload = await r.json();
    const bars = payload.bars || [];
    activeChartSymbol = payload.symbol;
    const nextContextKey = `${payload.symbol}|${payload.frame}`;
    chartLastBarCount = bars.length;
    chartLastBarTime = bars.length ? Number(bars[bars.length - 1].time || 0) : 0;
    if (contextChanged || !chartZoomBaseBars) chartZoomBaseBars = Math.max(bars.length, 1);
    candleSeries.setData(bars);
    clearChartOverlays();
    const overlays = payload.overlays || {};
    for (const line of overlays.lines || []) drawLine(line.points || [], line.color || '#3b82f6', line.style || 'solid', line.width || 2);
    for (const level of overlays.levels || []) {
      const levelWidth = level.width || 1;
      drawLine([{time: level.start_time, value: level.price}, {time: level.end_time, value: level.price}], level.color || '#94a3b8', level.style || 'dashed', levelWidth);
      try {
        priceLines.push(candleSeries.createPriceLine({ price: level.price, color: level.color || '#94a3b8', lineWidth: levelWidth, lineStyle: lineStyle(level.style), axisLabelVisible: true, title: level.label || '' }));
      } catch(e) {}
    }
    const markerItems = [...(overlays.markers || []), ...(overlays.events || [])];
    const markers = markerItems.map(markerFromItem).filter(Boolean).filter(m => m.time).sort((a, b) => a.time - b.time);
    candleSeries.setMarkers(markers);
    if (bars.length) {
      if (contextChanged || !chartHadData) {
        if (activeChartZoomPercent === 100) {
          scale.fitContent();
        } else {
          setChartZoomPercent(activeChartZoomPercent, 'latest');
        }
      } else if (previousRange && !wasAtLatest) {
        scale.setVisibleLogicalRange(previousRange);
      }
      if (contextChanged || !chartHadData) applyPriceAutoscale();
    }
    chartContextKey = nextContextKey;
    chartHadData = bars.length > 0;
    chartLastFullRefreshAt = Date.now();
    $('chart-status').textContent = `${payload.symbol} ${payload.frame.toUpperCase()} / ${bars.length} bars / ${markerItems.length} annotations`;
    renderChartDiagnostics(payload);
  } catch(e) {
    $('chart-status').textContent = 'Chart load failed: ' + e.message;
  } finally {
    chartFullRefreshInFlight = false;
  }
}

async function refreshLiveChart() {
  if (activeTab !== 'charts' || chartLiveInFlight || chartFullRefreshInFlight) return;
  ensureChart();
  if (!tvChart || !candleSeries) return;
  const symbol = $('chart-symbol').value || activeChartSymbol;
  if (!symbol || !chartHadData) return;
  chartLiveInFlight = true;
  try {
    const r = await fetch(`/api/chart/live?symbol=${encodeURIComponent(symbol)}&frame=${encodeURIComponent(activeChartFrame)}&ts=${Date.now()}`);
    if (!r.ok) throw new Error(await r.text());
    const payload = await r.json();
    const bar = payload.bar;
    if (!bar || payload.symbol !== activeChartSymbol || payload.frame !== activeChartFrame) return;
    const nextTime = Number(bar.time || 0);
    const isNewBar = chartLastBarTime > 0 && nextTime > chartLastBarTime;
    candleSeries.update(bar);
    if (isNewBar) chartLastBarCount = Math.min(chartLastBarCount + 1, 800);
    chartLastBarTime = Math.max(chartLastBarTime, nextTime);
    const livePrice = Number(payload.last_price ?? bar.close);
    const priceText = Number.isFinite(livePrice) ? ` / live ${livePrice.toFixed(2)}` : '';
    $('chart-status').textContent = `${payload.symbol} ${payload.frame.toUpperCase()} / ${chartLastBarCount} bars${priceText}`;
    if (isNewBar || Date.now() - chartLastFullRefreshAt > chartOverlayRefreshMs) {
      refreshChart(false);
    }
  } catch(e) {
    $('chart-status').textContent = 'Live chart update failed: ' + e.message;
  } finally {
    chartLiveInFlight = false;
  }
}

function renderChartDiagnostics(payload) {
  const body = $('chart-diagnostics');
  const overlays = payload.overlays || {};
  const rows = diagnosticRows(overlays);
  $('diag-count').textContent = rows.length.toString();
  const state = overlays.state?.price_action_bs || {};
  let html = `<div class="ops-item" style="margin-bottom:12px;">
    <div class="ops-head"><div class="ops-title">${esc(payload.symbol)} ${esc(String(payload.frame || '').toUpperCase())}</div><div class="ops-pill pill-blue">${esc(state.main_bias || 'neutral')}</div></div>
    <div class="kv"><div class="kv-label">Pending</div><div class="kv-value">${esc(state.pending_candidates ?? 0)}</div></div>
    <div class="kv"><div class="kv-label">Vol Cooldown</div><div class="kv-value">${state.volatility_cooldown_active ? 'yes' : 'no'}</div></div>
    <div class="kv"><div class="kv-label">Range</div><div class="kv-value">${state.range_state?.high ? `${Number(state.range_state.low).toFixed(2)}-${Number(state.range_state.high).toFixed(2)}` : '--'}</div></div>
  </div>`;
  if (!rows.length) {
    body.innerHTML = html + '<div class="no-trades">No strategy diagnostics or swing markers yet</div>';
    return;
  }
  for (const event of rows) {
    const pill = event.kind === 'failed' ? 'pill-amber'
      : event.kind === 'signal' || event.kind === 'entry' ? 'pill-green'
      : event.kind === 'swing_high' ? 'pill-red'
      : event.kind === 'swing_low' ? 'pill-green'
      : 'pill-blue';
    html += `<div class="event-row">
      <div class="event-title">
        <span class="event-title-text">${esc(event.label || event.kind)}</span>
        <span class="event-meta"><span class="event-time">${esc(formatChartTimestamp(event.time))}</span><span class="ops-pill ${pill}">${esc(event.setup_id || event.kind)}</span></span>
      </div>
      <div class="event-detail">${esc(event.side || '')} ${event.price ? '@ ' + Number(event.price).toFixed(2) : ''}</div>
      <div class="event-detail">${esc(event.detail || event.reason || '')}</div>
    </div>`;
  }
  body.innerHTML = html;
}

async function confirmFlatten() {
  const btn = $('flatten-btn');
  const originalHtml = btn.innerHTML;
  if (confirm("🚨 EMERGENCY FLATTEN: Close all positions and cancel all pending orders immediately?")) {
    try {
      btn.disabled = true;
      btn.style.opacity = '0.5';
      btn.innerHTML = 'FLATTENING...';
      const r = await fetch('/api/flatten', {method: 'POST'});
      if (r.ok) {
        alert("Command Sent: All positions are being flattened.");
      } else {
        alert("Error: Command failed to send. Check bot logs.");
      }
    } catch(e) {
      alert("Error: Website could not connect to bot.");
    } finally {
      btn.disabled = false;
      btn.style.opacity = '1';
      btn.innerHTML = originalHtml;
    }
  }
}

async function triggerSaturdayOptimizer() {
  const btn = $('sat-run-btn');
  const btnTab = $('sat-run-btn-tab');
  const originalText = btn?.textContent || '⚡ Run Dry-Run';
  const activePos = latestStatus?.active_positions ?? {};
  const hasPositions = Object.values(activePos).some(arr => Array.isArray(arr) && arr.length > 0);
  if (hasPositions) {
    alert("Cannot run optimizer: open positions currently detected.");
    return;
  }
  if (!confirm("Run Saturday Autonomous Optimizer now in dry-run mode? (Diagnoses trades via Qwen 3.5 122B and backtests candidate hypotheses)")) return;
  try {
    if (btn) { btn.disabled = true; btn.textContent = 'Starting...'; btn.style.opacity = '0.5'; }
    if (btnTab) { btnTab.disabled = true; btnTab.textContent = 'Starting...'; btnTab.style.opacity = '0.5'; }
    const r = await fetch('/api/saturday-optimizer/run', {method: 'POST'});
    if (r.ok) {
      alert("Saturday Optimizer dry-run started! Watch progress in the widget below.");
    } else {
      const err = await r.json().catch(() => ({}));
      alert("Failed to start: " + (err.error || "Unknown error"));
    }
  } catch (e) {
    alert("Could not connect to server: " + e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = originalText; btn.style.opacity = '1'; }
    if (btnTab) { btnTab.disabled = false; btnTab.textContent = originalText; btnTab.style.opacity = '1'; }
  }
}

async function refresh() {
  try {
    const r = await fetch('/api/status?' + Date.now());
    if (!r.ok) throw new Error();
    const d = await r.json();
    $('offline').classList.remove('show');

    setBadge('badge-running', d.running ? 'badge-green' : 'badge-red', d.running ? 'dot-green' : 'dot-red', d.running ? 'RUNNING' : 'STOPPED');

    // Enhanced Mode Badge
    const isLive = d.mode === 'LIVE';
    const modeBadge = $('badge-mode');
    modeBadge.className = 'badge ' + (isLive ? 'badge-red' : 'badge-blue');
    modeBadge.innerHTML = `<div class="dot ${isLive ? 'dot-red' : 'dot-muted'}"></div>${isLive ? 'LIVE TRADING' : 'SIMULATION'}`;
    modeBadge.style.boxShadow = isLive ? '0 0 15px rgba(239, 68, 68, 0.4)' : 'none';
    modeBadge.style.borderWidth = isLive ? '2px' : '1px';

    setBadge('badge-market',  d.market_open  ? 'badge-green' : 'badge-amber', d.market_open  ? 'dot-green' : 'dot-amber', d.market_open  ? 'RTH OPEN' : 'MARKET CLOSED');
    setBadge('badge-news',    d.news_standdown ? 'badge-amber' : 'badge-muted', d.news_standdown ? 'dot-amber' : 'dot-muted', d.news_standdown ? 'NEWS STAND-DOWN' : 'No News');
    setBadge('badge-signalr', d.signalr_connected ? 'badge-green' : 'badge-red', d.signalr_connected ? 'dot-green' : 'dot-red', d.signalr_connected ? 'SignalR ✓' : 'SignalR ✗');
    $('last-update').textContent = 'Updated: ' + d.last_updated;

    const pnl = d.session_pnl ?? 0;
    const unrealized = d.unrealized_pnl ?? 0;
    const totalPnl = pnl + unrealized;

    const pnlEl = $('card-pnl');
    pnlEl.textContent = fmt$(pnl);
    pnlEl.className = 'card-value ' + (pnl >= 0 ? 'positive' : 'negative');
    $('card-pnl-sub').textContent = 'Total Session: ' + fmt$(totalPnl);

    const unrEl = $('card-unrealized');
    unrEl.textContent = fmt$(unrealized);
    unrEl.className = 'card-value ' + (unrealized > 0 ? 'positive' : unrealized < 0 ? 'negative' : 'neutral');
    $('card-unrealized-sub').textContent = unrealized !== 0 ? '🟢 Market moving...' : '⚪ No active floating P&L';

    const netEquity = d.net_equity !== undefined ? d.net_equity : ((d.balance ?? 0) + unrealized);
    $('card-balance').textContent    = d.balance ? fmtB(netEquity) : '--';
    $('card-balance-sub').textContent = 'Cash: ' + (d.balance ? fmtB(d.balance) : '--') + ' | Peak: ' + (d.peak_balance ? fmtB(d.peak_balance) : '--');

    const dd = d.drawdown ?? 0, ddMax = d.max_drawdown ?? 1300; // Use a default or fallthrough
    const ddPct = Math.min(100, (dd / ddMax) * 100);
    $('card-dd').textContent = fmtB(dd);
    $('card-dd').className   = 'card-value ' + (dd > 0 ? 'negative' : 'neutral');
    $('dd-bar').style.width  = ddPct + '%';
    $('dd-limit').textContent = fmtB(ddMax);
    if ($('card-dd-sub')) {
      const minBal = d.minimum_balance;
      const buffer = Math.max(0, ddMax - dd);
      $('card-dd-sub').textContent = (minBal ? 'Floor: ' + fmtB(minBal) : 'MLL limit') + ' | Buffer: ' + fmtB(buffer);
    }

    const allPos = d.active_positions ?? {};
    let totalPos = 0; const syms = [];
    for (const [s, list] of Object.entries(allPos)) { totalPos += list.length; if (list.length) syms.push(s); }
    $('card-pos').textContent  = totalPos;
    $('card-sym').textContent  = syms.length ? syms.join(', ') : 'No active symbols';
    $('pos-count').textContent = totalPos;
    updateChartSymbols(d.symbols || Object.keys(allPos));

    const posBody = $('positions-body');
    if (totalPos === 0) {
      posBody.innerHTML = '<div class="pos-empty"><div class="pos-empty-icon">📋</div>No open positions</div>';
    } else {
      let h = '<div class="grid-hdr pos-cols"><span>Symbol</span><span>Side</span><span>Qty</span><span>Entry</span><span>Last</span><span style="text-align:right">P&L</span></div>';
      for (const [s, list] of Object.entries(allPos)) {
        for (const p of list) {
          const timeStr = p.time && p.time !== '(broker)' ? p.time : (p.time === '(broker)' ? 'broker' : '');
          h += `<div class="grid-row pos-cols">` +
               `<div style="min-width:0; overflow:hidden;"><div class="grid-cell-truncate" style="font-weight:600">${esc(s)}</div>${timeStr ? `<div class="mono" style="font-size:10px; color:var(--muted); line-height:1.1;">${esc(timeStr)}</div>` : ''}</div>` +
               `<span class="${p.side==='Buy'?'side-buy':'side-sell'}" style="white-space:nowrap;">${esc(p.side ? p.side.toUpperCase() : '')}</span>` +
               `<span>${esc(p.qty)}</span>` +
               `<span class="mono">${fmtPx(p.entry)}</span>` +
               `<span class="mono">${fmtPx(p.last_price)}</span>` +
               `<span class="${(p.open_pnl ?? 0)>=0?'trade-pnl-pos':'trade-pnl-neg'}" style="text-align:right; justify-self:end; font-variant-numeric:tabular-nums;">${p.open_pnl === undefined ? '--' : fmt$(p.open_pnl)}</span>` +
               `</div>`;
        }
      }
      posBody.innerHTML = h;
    }

    // Consolidated Trade Execution History
    const seenTrades = new Set();
    const tradeList = [];
    const rawTrades = [...(d.session_trades || []), ...(d.recent_trades || [])];
    for (const t of rawTrades) {
      if (!t) continue;
      const key = `${t.time || ''}|${t.symbol || ''}|${t.side || ''}|${t.type || ''}|${t.pnl ?? ''}`;
      if (!seenTrades.has(key)) {
        seenTrades.add(key);
        tradeList.push(t);
      }
    }
    const tradeCountEl = $('trade-count');
    if (tradeCountEl) tradeCountEl.textContent = tradeList.length + ' trades';
    const tradesBody = $('trades-body');
    if (tradesBody) {
      if (tradeList.length === 0) {
        tradesBody.innerHTML = '<div class="no-trades">No closed trades recorded yet</div>';
      } else {
        let h = '<div class="grid-hdr trade-cols"><span>Date / Time</span><span>Session</span><span>Symbol</span><span>Side</span><span>Type</span><span style="text-align:right; justify-self:end;">P&L</span></div>';
        for (const t of tradeList) {
          h += `<div class="grid-row trade-cols">` +
               `<span class="mono grid-cell-truncate" style="color:var(--muted)">${esc(t.time || '--')}</span>` +
               `<span>${sessionBadge(t.session)}</span>` +
               `<span class="grid-cell-truncate" style="font-weight:600">${esc(t.symbol || '--')}</span>` +
               `<span class="${t.side === 'Buy' ? 'side-buy' : 'side-sell'}" style="white-space:nowrap;">${esc(t.side ? t.side.toUpperCase() : '--')}</span>` +
               `<span class="grid-cell-truncate" style="color:var(--muted); text-transform:capitalize;" title="${esc(t.type || '')}">${esc(cleanTradeType(t.type))}</span>` +
               `<span class="${(t.pnl ?? 0) >= 0 ? 'trade-pnl-pos' : 'trade-pnl-neg'}" style="text-align:right; justify-self:end; font-variant-numeric:tabular-nums;">${fmt$(t.pnl)}</span>` +
               `</div>`;
        }
        tradesBody.innerHTML = h;
      }
    }

    // Strategy Command Center
    const stratBody = $('strategy-body');
    if (stratBody) {
      if (!d.strategies || Object.keys(d.strategies).length === 0) {
        stratBody.innerHTML = '<div class="no-trades">Awaiting strategy engine telemetry...</div>';
      } else {
        let sh = '';
        for (const [sym, data] of Object.entries(d.strategies)) {
          const ind = data.indicators || {};
          const status = ind.status || 'Active';
          const isStale = !!ind.data_stale;
          const barCount = ind.bar_count || 0;
          const price = ind.price;
          const vwap = ind.vwap;
          const ema200 = ind.ema_200;
          const rsi = ind.rsi;
          const adx = ind.adx;
          const atr = ind.atr;
          const sessionRegime = ind.session_regime || 'GLOBEX';

          const f = (val, dec=2) => (val === null || val === undefined || isNaN(val)) ? '--' : Number(val).toLocaleString('en-US', {minimumFractionDigits:dec, maximumFractionDigits:dec});

          let vwapDiffText = '--';
          let vwapColor = 'var(--text)';
          if (price != null && vwap != null) {
            const diff = price - vwap;
            vwapDiffText = (diff >= 0 ? '+' : '') + diff.toFixed(2) + ' (' + (diff >= 0 ? 'Above' : 'Below') + ')';
            vwapColor = diff >= 0 ? 'var(--green)' : 'var(--red)';
          }

          let emaDiffText = '--';
          let emaColor = 'var(--text)';
          if (price != null && ema200 != null) {
            const diff = price - ema200;
            emaDiffText = (diff >= 0 ? 'Bullish' : 'Bearish') + ' (' + (diff >= 0 ? '+' : '') + diff.toFixed(2) + ')';
            emaColor = diff >= 0 ? 'var(--green)' : 'var(--red)';
          }

          let rsiStatus = 'Neutral';
          let rsiColor = 'var(--text)';
          if (rsi != null) {
            if (rsi >= 70) { rsiStatus = 'Overbought'; rsiColor = 'var(--red)'; }
            else if (rsi <= 30) { rsiStatus = 'Oversold'; rsiColor = 'var(--green)'; }
            else { rsiStatus = 'Balanced'; rsiColor = 'var(--blue)'; }
          }

          let adxStatus = 'Consolidating';
          let adxColor = 'var(--muted)';
          if (adx != null) {
            if (adx >= 30) { adxStatus = 'Strong Trend'; adxColor = 'var(--green)'; }
            else if (adx >= 20) { adxStatus = 'Moderate Trend'; adxColor = 'var(--blue)'; }
            else { adxStatus = 'Ranging / Choppy'; adxColor = 'var(--amber)'; }
          }

          sh += `<div style="background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom: 14px; border-bottom: 1px solid var(--border); padding-bottom: 10px;">
              <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size: 16px; font-weight: 700; color: #60a5fa; letter-spacing: 0.5px;">${esc(sym)}</span>
                <span class="ops-pill ${status === 'Active' ? 'pill-green' : 'pill-amber'}">${esc(status.toUpperCase())}</span>
                ${sessionBadge(sessionRegime)}
                ${isStale ? '<span class="ops-pill pill-red">FEED DELAYED</span>' : '<span class="ops-pill pill-blue">FEED LIVE</span>'}
              </div>
              <div style="display:flex; align-items:center; gap:16px;">
                <div style="font-size: 12px; font-family:'JetBrains Mono', monospace;"><span style="color:var(--muted)">Last: </span><span style="font-weight:700; color:var(--text);">${f(price)}</span></div>
                <div style="font-size: 11px; color:var(--muted);"><span style="font-family:'JetBrains Mono', monospace;">${esc(barCount)}</span> bars active</div>
              </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px;">
              <div style="background: var(--surface); padding: 10px 12px; border-radius: 6px; border: 1px solid var(--border);">
                <div style="font-size: 10px; color: var(--muted); text-transform: uppercase;">VWAP Benchmark</div>
                <div style="font-size: 13px; font-weight: 700; margin-top: 4px; font-family:'JetBrains Mono', monospace;">${f(vwap)}</div>
                <div style="font-size: 11px; color: ${vwapColor}; font-weight: 600; margin-top: 2px;">${vwapDiffText}</div>
              </div>

              <div style="background: var(--surface); padding: 10px 12px; border-radius: 6px; border: 1px solid var(--border);">
                <div style="font-size: 10px; color: var(--muted); text-transform: uppercase;">EMA 200 Baseline</div>
                <div style="font-size: 13px; font-weight: 700; margin-top: 4px; font-family:'JetBrains Mono', monospace;">${f(ema200)}</div>
                <div style="font-size: 11px; color: ${emaColor}; font-weight: 600; margin-top: 2px;">${emaDiffText}</div>
              </div>

              <div style="background: var(--surface); padding: 10px 12px; border-radius: 6px; border: 1px solid var(--border);">
                <div style="font-size: 10px; color: var(--muted); text-transform: uppercase;">Momentum ADX (14)</div>
                <div style="font-size: 13px; font-weight: 700; margin-top: 4px; font-family:'JetBrains Mono', monospace;">${f(adx, 1)}</div>
                <div style="font-size: 11px; color: ${adxColor}; font-weight: 600; margin-top: 2px;">${adxStatus}</div>
              </div>

              <div style="background: var(--surface); padding: 10px 12px; border-radius: 6px; border: 1px solid var(--border);">
                <div style="font-size: 10px; color: var(--muted); text-transform: uppercase;">Oscillator RSI (14)</div>
                <div style="font-size: 13px; font-weight: 700; margin-top: 4px; font-family:'JetBrains Mono', monospace;">${f(rsi, 1)}</div>
                <div style="font-size: 11px; color: ${rsiColor}; font-weight: 600; margin-top: 2px;">${rsiStatus}</div>
              </div>

              <div style="background: var(--surface); padding: 10px 12px; border-radius: 6px; border: 1px solid var(--border);">
                <div style="font-size: 10px; color: var(--muted); text-transform: uppercase;">Volatility ATR (14)</div>
                <div style="font-size: 13px; font-weight: 700; margin-top: 4px; font-family:'JetBrains Mono', monospace;">${f(atr)}</div>
                <div style="font-size: 11px; color: var(--muted); margin-top: 2px;">Exp. Range/Bar</div>
              </div>
            </div>
          </div>`;
        }
        stratBody.innerHTML = sh;
      }
    }

    renderSaturdayOptimizer(d.saturday_optimizer);
  } catch(e) { $('offline').classList.add('show'); }
}

function renderSaturdayOptimizer(sat) {
  const body = $('sat-body');
  const bodyTab = $('sat-body-tab');
  const pill = $('sat-phase-pill');
  const pillTab = $('sat-phase-pill-tab');
  const lastRun = $('sat-last-run');
  const lastRunTab = $('sat-last-run-tab');
  const optBadge = $('badge-sat-opt');

  const setPill = (txt, cls) => {
    if (pill) { pill.textContent = txt; pill.className = 'ops-pill ' + cls; }
    if (pillTab) { pillTab.textContent = txt; pillTab.className = 'ops-pill ' + cls; }
  };
  const setLastRun = txt => {
    if (lastRun) lastRun.textContent = txt;
    if (lastRunTab) lastRunTab.textContent = txt;
  };
  const setBody = content => {
    if (body) body.innerHTML = content;
    if (bodyTab) bodyTab.innerHTML = content;
  };

  const getNextSaturdayET = () => {
    try {
      const now = new Date();
      const nyStr = now.toLocaleString("en-US", { timeZone: "America/New_York" });
      const nyDate = new Date(nyStr);
      const day = nyDate.getDay(); // 0 = Sun, ..., 6 = Sat
      let daysUntilSat = (6 - day + 7) % 7;
      if (daysUntilSat === 0 && (nyDate.getHours() > 8 || (nyDate.getHours() === 8 && nyDate.getMinutes() > 0))) {
        daysUntilSat = 7;
      }
      const nextSat = new Date(nyDate);
      nextSat.setDate(nyDate.getDate() + daysUntilSat);
      const yyyy = nextSat.getFullYear();
      const mm = String(nextSat.getMonth() + 1).padStart(2, '0');
      const dd = String(nextSat.getDate()).padStart(2, '0');
      return `${yyyy}-${mm}-${dd} 08:00 ET`;
    } catch(e) {
      return 'Sat 08:00 ET';
    }
  };

  const formatLastRunET = (s) => {
    if (!s) return '--';
    if (s.last_updated_utc) {
      try {
        const d = new Date(s.last_updated_utc);
        const nyDate = new Date(d.toLocaleString("en-US", { timeZone: "America/New_York" }));
        const yyyy = nyDate.getFullYear();
        const mm = String(nyDate.getMonth() + 1).padStart(2, '0');
        const dd = String(nyDate.getDate()).padStart(2, '0');
        const hh = String(nyDate.getHours()).padStart(2, '0');
        const min = String(nyDate.getMinutes()).padStart(2, '0');
        return `${yyyy}-${mm}-${dd} ${hh}:${min} ET`;
      } catch(e) {}
    }
    return s.run_id || '--';
  };

  const nextRunStr = getNextSaturdayET();
  const lastRunStr = formatLastRunET(sat);

  if (!sat || !sat.phase || sat.phase === 'INITIAL') {
    if (optBadge) setBadge('badge-sat-opt', 'badge-blue', 'dot-blue', 'SAT OPT: IDLE');
    setPill('IDLE', 'pill-blue');
    setLastRun(`LAST: ${lastRunStr} | NEXT: ${nextRunStr}`);
    setBody('<div class="no-trades">Optimizer is idle. It runs autonomously on Saturday at 08:00 ET when flat, or click "Run Dry-Run" to trigger an audit now.</div>');
    return;
  }

  const phase = sat.phase;
  const winner = sat.promoted_candidate;
  const isDry = !!sat.dry_run && !sat.promoted_to_production;
  if (phase && phase !== 'COMPLETED') {
    setLastRun(`RUNNING (${phase.replace('_', ' ')}) | NEXT: ${nextRunStr}`);
  } else {
    setLastRun(`LAST: ${lastRunStr} | NEXT: ${nextRunStr}`);
  }

  if (phase === 'COMPLETED') {
    if (winner) {
      if (isDry) {
        if (optBadge) setBadge('badge-sat-opt', 'badge-amber', 'dot-amber', 'SAT OPT: DRY-RUN WINNER');
        setPill('DRY-RUN WINNER', 'pill-amber');
      } else {
        if (optBadge) setBadge('badge-sat-opt', 'badge-green', 'dot-green', 'SAT OPT: PROMOTED');
        setPill('PROMOTED', 'pill-green');
      }
    } else {
      if (optBadge) setBadge('badge-sat-opt', 'badge-blue', 'dot-blue', 'SAT OPT: BASELINE HELD');
      setPill('BASELINE HELD', 'pill-blue');
    }
  } else {
    if (optBadge) setBadge('badge-sat-opt', 'badge-amber', 'dot-amber', 'SAT OPT: ' + phase.replace('_', ' '));
    setPill(phase.replace('_', ' '), 'pill-amber');
  }

  let html = '';

  // Top Metrics Grid
  const base = sat.baseline_metrics || {};
  html += `<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px;">
    <div style="background: var(--surface2); padding: 12px; border-radius: 8px; border: 1px solid var(--border);">
      <div style="font-size: 10px; color: var(--muted); text-transform: uppercase;">Active Phase</div>
      <div style="font-size: 14px; font-weight: 700; color: var(--text); margin-top: 4px;">${esc(phase)}</div>
      <div style="font-size: 11px; color: var(--muted);">${sat.extracted_trades_count ?? 0} trades audited</div>
    </div>
    <div style="background: var(--surface2); padding: 12px; border-radius: 8px; border: 1px solid var(--border);">
      <div style="font-size: 10px; color: var(--muted); text-transform: uppercase;">Decision / Outcome</div>
      <div style="font-size: 14px; font-weight: 700; color: ${winner ? 'var(--green)' : 'var(--blue)'}; margin-top: 4px;">${winner ? (isDry ? 'Dry-Run Winner Found' : 'Promoted to Production') : 'Hold Baseline'}</div>
      <div style="font-size: 11px; color: var(--muted);">${winner ? esc(winner.candidate_id) : 'Baseline outperforms'}</div>
    </div>
    <div style="background: var(--surface2); padding: 12px; border-radius: 8px; border: 1px solid var(--border);">
      <div style="font-size: 10px; color: var(--muted); text-transform: uppercase;">Baseline PF / Holdout</div>
      <div style="font-size: 14px; font-weight: 700; color: var(--text); margin-top: 4px;">PF ${(base.profit_factor ?? 0).toFixed(2)} | Holdout ${fmt$(base.holdout_pnl ?? 0)}</div>
      <div style="font-size: 11px; color: var(--muted);">MDD: ${fmtB(base.max_drawdown ?? 0)} (${base.total_trades ?? 0} trades)</div>
    </div>
    <div style="background: var(--surface2); padding: 12px; border-radius: 8px; border: 1px solid var(--border);">
      <div style="font-size: 10px; color: var(--muted); text-transform: uppercase;">Candidates Tested</div>
      <div style="font-size: 14px; font-weight: 700; color: var(--text); margin-top: 4px;">${(sat.evaluation_results ?? []).length} tested / ${(sat.hypotheses ?? []).length} hypotheses</div>
      <div style="font-size: 11px; color: var(--muted);">Local Qwen 3.5 (122B)</div>
    </div>
  </div>`;

  // Diagnostic Audit Summary
  if (sat.analysis_summary || (sat.identified_flaws && sat.identified_flaws.length)) {
    html += `<div style="background: var(--surface2); padding: 14px; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 16px;">
      <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--blue); margin-bottom: 6px;">🧠 Weekly Trade Loss Diagnosis (Qwen 3.5 122B)</div>
      ${sat.analysis_summary ? `<div style="font-size: 12px; line-height: 1.5; color: var(--text); margin-bottom: 10px;">${esc(sat.analysis_summary)}</div>` : ''}`;
    if (sat.identified_flaws && sat.identified_flaws.length) {
      html += `<div style="display: flex; flex-direction: column; gap: 4px;">`;
      for (const flaw of sat.identified_flaws) {
        html += `<div style="display:flex; align-items:flex-start; gap:8px; font-size:11px; color:var(--muted);"><span style="color:var(--amber);">⚠️</span><span>${esc(flaw)}</span></div>`;
      }
      html += `</div>`;
    }
    html += `</div>`;
  }

  // Candidate Evaluation Table
  const results = sat.evaluation_results || [];
  html += `<div style="overflow-x:auto;">
    <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--muted); margin-bottom: 6px;">Empirical Backtest & 21-Day Holdout Matrix (MES + MCL Concurrent)</div>
    <table class="sat-table">
      <thead>
        <tr>
          <th>Candidate / Parameter</th>
          <th>Strategy</th>
          <th>Value</th>
          <th>Profit Factor</th>
          <th>Total PnL</th>
          <th>21d Holdout PnL</th>
          <th>Max DD</th>
          <th>Trades</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>`;

  if (base && base.total_trades) {
    const isPromoted = winner && !isDry;
    const basePill = isPromoted
      ? '<span class="ops-pill pill-muted">SUPERSEDED</span>'
      : (isDry && winner ? '<span class="ops-pill pill-blue">CURRENT (LIVE)</span>' : '<span class="ops-pill pill-green">ACTIVE (HELD)</span>');
    const baseLabel = isPromoted ? '[BASELINE] Prior Config' : '[BASELINE] Production';

    html += `<tr style="opacity: 0.9;">
      <td><span style="font-weight:700; color:var(--text)">${baseLabel}</span></td>
      <td>MTF + ORB</td>
      <td>v__APP_VERSION__</td>
      <td><span style="font-weight:700; color:${base.profit_factor >= 1.25 ? 'var(--green)' : 'var(--text)'}">${Number(base.profit_factor).toFixed(2)}</span></td>
      <td style="color:${base.total_pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${fmt$(base.total_pnl)}</td>
      <td style="color:${base.holdout_pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${fmt$(base.holdout_pnl)}</td>
      <td>${fmtB(base.max_drawdown ?? 0)}</td>
      <td>${base.total_trades}</td>
      <td>${basePill}</td>
    </tr>`;
  }

  if (results.length === 0) {
    html += `<tr><td colspan="9" style="text-align:center; color:var(--muted); padding:20px;">No candidate backtests evaluated yet.</td></tr>`;
  } else {
    for (const r of results) {
      const isWinner = winner && winner.candidate_id === r.candidate_id;
      const pfCol = r.profit_factor >= 1.25 ? 'var(--green)' : 'var(--text)';
      let candPill = '<span class="ops-pill pill-muted">TESTED</span>';
      if (isWinner) {
        candPill = isDry
          ? '<span class="ops-pill pill-amber">DRY-RUN WINNER</span>'
          : '<span class="ops-pill pill-green">PROMOTED (ACTIVE)</span>';
      }
      html += `<tr class="${isWinner ? 'sat-winner-row' : ''}">
        <td><div style="font-weight:600; font-family:'JetBrains Mono', monospace; font-size:11px;">${esc(r.candidate_id)}</div>${r.rationale ? `<div style="font-size:10px; color:var(--muted);">${esc(r.rationale)}</div>` : ''}</td>
        <td>${esc(r.strategy)}</td>
        <td style="font-family:'JetBrains Mono', monospace;">${esc(JSON.stringify(r.value))}</td>
        <td><span style="font-weight:700; color:${pfCol}">${Number(r.profit_factor).toFixed(2)}</span></td>
        <td style="color:${r.total_pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${fmt$(r.total_pnl)}</td>
        <td style="color:${r.holdout_pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${fmt$(r.holdout_pnl)}</td>
        <td>${fmtB(r.max_drawdown ?? 0)}</td>
        <td>${r.total_trades}</td>
        <td>${candPill}</td>
      </tr>`;
    }
  }

  html += `</tbody></table></div>`;
  setBody(html);
}

refresh();
setInterval(refresh, 1000);
setInterval(() => refreshLiveChart(), 750);
setInterval(() => refreshChart(false), chartOverlayRefreshMs);
</script>
</body>
</html>
"""
HTML = HTML.replace("__APP_VERSION__", VERSION)


def _check_auth(handler) -> bool:
    if not PASSWORD:
        return True
    auth = handler.headers.get("Authorization", "")
    if auth.startswith("Basic "):
        try:
            import hmac
            _, pwd = base64.b64decode(auth[6:]).decode().split(":", 1)
            return hmac.compare_digest(pwd, PASSWORD)
        except Exception as auth_err:
            logger.debug("Auth header decode failed: %s", auth_err)
            return False
    return False


class Handler(BaseHTTPRequestHandler):
    supervisor: Any = None

    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-XSS-Protection", "1; mode=block")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self' https://cdn.jsdelivr.net https://unpkg.com;"
        )
        super().end_headers()

    def _require_auth(self) -> bool:
        if _check_auth(self):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="TopstepX Dashboard"')
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Unauthorized. Set DASHBOARD_PASSWORD env var.")
        return False

    def _check_same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin:
            if not host or urlparse(origin).netloc != host:
                self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Cross-origin POST rejected"}).encode("utf-8"))
                return False
            return True
        referer = self.headers.get("Referer")
        if referer and host and urlparse(referer).netloc != host:
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Cross-origin POST rejected"}).encode("utf-8"))
            return False
        return True

    def do_POST(self):
        if not self._require_auth():
            return
        if not self._check_same_origin():
            return

        if self.path == "/api/flatten":
            import asyncio
            if hasattr(self, 'supervisor') and self.supervisor and self.supervisor.loop:
                self.log_message("Dashboard: Triggering EMERGENCY FLATTEN via API")
                # Thread-safe async call
                asyncio.run_coroutine_threadsafe(
                    self.supervisor._flatten_all_emergency(),
                    self.supervisor.loop
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
            else:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Error: Supervisor loop not initialized")
        elif self.path == "/api/saturday-optimizer/run":
            import subprocess
            import sys
            try:
                if hasattr(self, 'supervisor') and self.supervisor:
                    active_pos = getattr(self.supervisor, 'active_positions', None) or {}
                    has_positions = any(active_pos.values()) if isinstance(active_pos, dict) else bool(active_pos)
                    if has_positions:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "Optimizer cannot run while positions are open."}).encode("utf-8"))
                        return
                    opt_task = getattr(self.supervisor, '_optimizer_task', None)
                    if opt_task and hasattr(opt_task, 'done') and not opt_task.done():
                        self.send_response(409)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "Optimizer is already running."}).encode("utf-8"))
                        return
                    from datetime import datetime, timezone
                    news_agent = getattr(self.supervisor, 'news_agent', None)
                    is_open = bool(news_agent and hasattr(news_agent, 'is_market_hours') and news_agent.is_market_hours(datetime.now(timezone.utc)))
                    if (is_open or getattr(self.supervisor, 'market_open', False)) and hasattr(self.supervisor, 'logger'):
                        self.supervisor.logger.warning("Manual optimizer dry-run triggered via dashboard during market hours.")
                cmd = [sys.executable, "scripts/saturday_autonomous_optimizer.py", "--dry-run", "--force-offline"]
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "started", "mode": "dry-run"}).encode("utf-8"))
            except Exception as exc:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
        elif self.path == "/api/risk-config":
            if not hasattr(self, 'supervisor') or not self.supervisor:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Error: Supervisor not initialized")
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))

                new_dll = payload.get("max_daily_loss")
                if new_dll is not None:
                    import math
                    val_float = float(new_dll)
                    if not math.isfinite(val_float) or val_float < 50.0 or val_float > 100000.0:
                        raise ValueError("max_daily_loss must be between 50 and 100000")
                    # Update active memory
                    from decimal import Decimal
                    val = Decimal(str(val_float))
                    self.supervisor.risk_agent.max_daily_loss = val
                    val_str = str(int(val_float)) if val_float.is_integer() else str(val_float)

                    # Persist to .env
                    import re
                    env_path = ".env"
                    if os.path.exists(env_path):
                        with open(env_path) as f:
                            content = f.read()
                        if "MAX_DAILY_LOSS" in content:
                            content = re.sub(r"MAX_DAILY_LOSS=.*", f"MAX_DAILY_LOSS={val_str}", content)
                        else:
                            content += f"\nMAX_DAILY_LOSS={val_str}\n"
                        with open(env_path, 'w') as f:
                            f.write(content)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "max_daily_loss": new_dll}).encode("utf-8"))
            except Exception as exc:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if not self._require_auth():
            return
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif path == "/api/chart":
            if not hasattr(self, 'supervisor') or not self.supervisor:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Supervisor not initialized"}).encode("utf-8"))
                return
            try:
                params = parse_qs(parsed.query)
                symbol = (params.get("symbol") or [""])[0]
                frame = (params.get("frame") or ["1m"])[0]
                limit = int((params.get("limit") or ["500"])[0])
                payload = self.supervisor.get_chart_payload(symbol=symbol, frame=frame, limit=limit)
                payload = self.supervisor._json_safe(payload) if hasattr(self.supervisor, "_json_safe") else payload
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            except Exception as exc:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
        elif path == "/api/chart/live":
            if not hasattr(self, 'supervisor') or not self.supervisor:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Supervisor not initialized"}).encode("utf-8"))
                return
            try:
                params = parse_qs(parsed.query)
                symbol = (params.get("symbol") or [""])[0]
                frame = (params.get("frame") or ["1m"])[0]
                payload = self.supervisor.get_chart_live_payload(symbol=symbol, frame=frame)
                payload = self.supervisor._json_safe(payload) if hasattr(self.supervisor, "_json_safe") else payload
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            except Exception as exc:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
        elif path == "/api/status":
            if STATUS_FILE.exists():
                try:
                    data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
                sat_state_file = Path("logs/saturday_optimizer_state.json")
                if sat_state_file.exists():
                    try:
                        data["saturday_optimizer"] = json.loads(sat_state_file.read_text(encoding="utf-8"))
                    except Exception as exc:
                        logger.debug("Failed to read saturday optimizer state: %s", exc)
                payload_bytes = json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(payload_bytes)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\n  ⚡ TopstepX Dashboard  →  http://localhost:{PORT}")
    if HOST not in ("127.0.0.1", "localhost"):
        print(f"  Remote access         →  http://<droplet-ip>:{PORT}")
    print(f"  Auth:  {'PASSWORD PROTECTED ✓' if PASSWORD else 'None  ← add DASHBOARD_PASSWORD env var before exposing remotely'}")
    print(f"  File:  {STATUS_FILE.absolute()}")
    print("  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")


if __name__ == "__main__":
    main()
