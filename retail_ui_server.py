"""Local retail insights dashboard with workflow, chat, and interactive plots.

Run with:
    python retail_ui_server.py

Or use START_APP.bat for a one-click start.
"""

from __future__ import annotations

import calendar
import json
import re
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from flask import Flask, Response, jsonify, request, send_file
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from werkzeug.utils import secure_filename

WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
  sys.path.insert(0, str(WORKSPACE_ROOT))

from phase1_retail_insights import format_currency, run_phase1_analysis

APP_TITLE = "Retail Insights Studio"
AUTHOR_NAME = "Nagabhushana Raju S"
AUTHOR_SIGNATURE = f"{AUTHOR_NAME} | Retail Dash"
OUTPUT_DIR = WORKSPACE_ROOT / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
INTERACTIVE_DIR = PLOT_DIR / "interactive"
CLEANED_CSV = OUTPUT_DIR / "cleaned_retail_data.csv"
REPORT_MD = OUTPUT_DIR / "phase1_report.md"
MODEL_PATH = MODEL_DIR / "retail_sales_pipeline.joblib"
MODEL_METRICS_PATH = MODEL_DIR / "retail_model_metrics.json"
PREDICTION_PLOT_PATH = PLOT_DIR / "predicted_vs_actual.png"

for folder in [OUTPUT_DIR, UPLOAD_DIR, MODEL_DIR, PLOT_DIR, INTERACTIVE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

app = Flask(__name__)

STATE_LOCK = threading.Lock()
APP_STATE: dict[str, Any] = {
    "status": "idle",
    "progress": 0,
    "stage": "idle",
    "message": "Ready. Press Start to run the pipeline.",
  "workflow_mode": "automatic",
  "workflow_step": "idle",
  "manual_dataset_name": None,
    "log": [],
    "analysis": {},
    "model": {},
    "artifacts": {},
    "model_ready": False,
    "data_ready": False,
    "last_error": None,
}
MODEL_CONTEXT: dict[str, Any] = {}
CURRENT_JOB: threading.Thread | None = None
MANUAL_DATASET_PATH: Path | None = None
MANUAL_DATASET_NAME: str | None = None

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>__APP_TITLE__</title>
  <style>
    :root {
      --bg1: #05070d;
      --bg2: #0b1220;
      --card: rgba(255, 255, 255, 0.04);
      --card-strong: rgba(255, 255, 255, 0.06);
      --stroke: rgba(255, 255, 255, 0.1);
      --text: #f1f5f9;
      --muted: rgba(241, 245, 249, 0.66);
      --mint: #4cc9c2;
      --sky: #7cc4ff;
      --sun: #d8b46a;
      --rose: #cd8fa0;
      --violet: #7d89ff;
      --lime: #9bbf7a;
      --shadow: 0 18px 50px rgba(1, 4, 10, 0.28);
      --radius: 18px;
    }
    * { box-sizing: border-box; }
    html, body { min-height: 100%; }
    body {
      margin: 0;
      font-family: "Aptos", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 20% 10%, rgba(76, 201, 194, 0.07), transparent 26%),
        radial-gradient(circle at 80% 86%, rgba(125, 137, 255, 0.05), transparent 28%),
        linear-gradient(180deg, var(--bg1), var(--bg2));
      color: var(--text);
      line-height: 1.5;
      overflow-x: hidden;
    }
    .brand-backdrop {
      position: fixed;
      inset: 0;
      z-index: 0;
      pointer-events: none;
      overflow: hidden;
      user-select: none;
    }
    .brand-backdrop span {
      position: absolute;
      white-space: nowrap;
      font-size: clamp(56px, 10vw, 144px);
      font-weight: 800;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: rgba(241, 245, 249, 0.045);
      -webkit-text-stroke: 1px rgba(255, 255, 255, 0.018);
      animation: driftBrand 32s ease-in-out infinite;
      filter: blur(0.15px);
    }
    .brand-backdrop .brand-a { top: 11%; left: -8%; }
    .brand-backdrop .brand-b { bottom: 14%; right: -18%; animation-direction: reverse; animation-duration: 38s; }
    .blob {
      position: fixed;
      z-index: 0;
      border-radius: 999px;
      filter: blur(28px);
      opacity: 0.16;
      animation: drift 28s ease-in-out infinite;
      pointer-events: none;
    }
    .blob.one { width: 240px; height: 240px; background: radial-gradient(circle, rgba(76,201,194,0.35), rgba(76,201,194,0.03)); top: 6%; right: 8%; }
    .blob.two { width: 260px; height: 260px; background: radial-gradient(circle, rgba(125,137,255,0.25), rgba(125,137,255,0.03)); bottom: 10%; left: -36px; animation-delay: -6s; }
    .blob.three { width: 170px; height: 170px; background: radial-gradient(circle, rgba(205,143,160,0.2), rgba(205,143,160,0.02)); top: 34%; left: 36%; animation-delay: -12s; }
    @keyframes drift {
      0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
      50% { transform: translate3d(18px, -16px, 0) scale(1.08); }
    }
    @keyframes driftBrand {
      0%, 100% { transform: translate3d(0, 0, 0) rotate(-8deg); }
      50% { transform: translate3d(48px, -18px, 0) rotate(-6deg); }
    }
    .wrap {
      position: relative;
      z-index: 1;
      max-width: 1380px;
      margin: 0 auto;
      padding: 20px 20px 28px;
    }
    .topbar {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      padding: 12px 16px;
      margin-bottom: 14px;
      border: 1px solid var(--stroke);
      border-radius: 20px;
      background: rgba(255,255,255,0.03);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
      backdrop-filter: blur(10px);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .logo {
      width: 52px;
      height: 52px;
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(76, 201, 194, 0.16), rgba(255, 255, 255, 0.04));
      border: 1px solid rgba(255, 255, 255, 0.12);
      box-shadow: none;
      position: relative;
      overflow: hidden;
    }
    .logo::after {
      content: "";
      position: absolute;
      inset: 11px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.28);
      animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
      0%, 100% { transform: scale(0.95); opacity: 0.65; }
      50% { transform: scale(1.05); opacity: 1; }
    }
    .brand h1 {
      margin: 0;
      font-size: clamp(22px, 2.8vw, 34px);
      letter-spacing: 0.15px;
    }
    .brand p, .subtitle, .muted {
      margin: 4px 0 0;
      color: var(--muted);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(76, 201, 194, 0.08);
      border: 1px solid rgba(76, 201, 194, 0.16);
      font-weight: 600;
      letter-spacing: 0.15px;
    }
    .badge .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #7ef0e6;
      box-shadow: 0 0 0 0 rgba(126, 240, 230, 0.45);
      animation: ping 2.4s infinite;
    }
    @keyframes ping {
      0% { box-shadow: 0 0 0 0 rgba(126, 240, 230, 0.45); }
      70% { box-shadow: 0 0 0 14px rgba(126, 240, 230, 0); }
      100% { box-shadow: 0 0 0 0 rgba(126, 240, 230, 0); }
    }
    .hero {
      display: grid;
      grid-template-columns: 1.25fr 0.85fr;
      gap: 14px;
      margin-bottom: 14px;
    }
    .card {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--stroke);
      border-radius: var(--radius);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.14);
      backdrop-filter: blur(8px);
    }
    .card-inner { padding: 16px; }
    .glass-title {
      margin: 0 0 8px;
      font-size: 18px;
      line-height: 1.15;
    }
    .hero-copy {
      display: grid;
      gap: 14px;
    }
    .hero-copy .headline {
      font-size: clamp(24px, 3.8vw, 42px);
      line-height: 1.05;
      margin: 0;
    }
    .hero-copy .headline span {
      color: var(--mint);
    }
    .hero-copy .lead {
      margin: 0;
      color: var(--muted);
      max-width: 68ch;
      line-height: 1.65;
      font-size: 15px;
    }
    .hero-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }
    .hidden {
      display: none !important;
    }
    .mode-strip {
      display: grid;
      gap: 10px;
      margin-top: 8px;
      padding: 12px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .mode-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: end;
    }
    .mode-field {
      min-width: 220px;
      flex: 0 1 280px;
    }
    .mode-help {
      color: var(--muted);
      font-size: 12.5px;
      line-height: 1.5;
    }
    .workflow-frame {
      display: grid;
      gap: 12px;
      margin-top: 12px;
      padding: 12px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .workflow-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
    }
    .workflow-tools {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
      align-items: flex-start;
    }
    .tool-stack {
      display: grid;
      gap: 4px;
      max-width: 190px;
    }
    .workflow-tool {
      padding: 8px 10px;
      font-size: 12px;
      border-radius: 12px;
    }
    .tool-note {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
    }
    .workflow-note {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12.5px;
      line-height: 1.45;
    }
    .pipeline-label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 800;
    }
    .pipeline-track {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 2px;
    }
    .pipeline-arrow {
      flex: 0 0 auto;
      color: rgba(241, 245, 249, 0.34);
      font-size: 18px;
      line-height: 1;
      user-select: none;
      transition: color .18s ease, opacity .18s ease;
    }
    .pipeline-arrow.is-running {
      color: rgba(76, 201, 194, 0.84);
    }
    .pipeline-arrow.is-done {
      color: rgba(148, 163, 184, 0.78);
    }
    .pipeline-arrow.is-error {
      color: rgba(236, 72, 153, 0.82);
    }
    .pipeline-arrow.is-blocked {
      opacity: 0.28;
    }
    .pipeline-step {
      min-width: 114px;
      border: 1px solid rgba(255, 255, 255, 0.11);
      border-radius: 14px;
      padding: 10px 12px;
      text-align: left;
      display: grid;
      gap: 4px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--text);
      opacity: 1;
      pointer-events: auto;
      cursor: pointer;
      box-shadow: none;
      transition: transform .18s ease, background .18s ease, border-color .18s ease, color .18s ease, opacity .18s ease;
    }
    .pipeline-step.step-primary {
      background: rgba(76, 201, 194, 0.08);
      border-color: rgba(76, 201, 194, 0.22);
    }
    .pipeline-step:not(:disabled):hover {
      transform: translateY(-1px);
    }
    .pipeline-step:focus-visible {
      outline: 2px solid rgba(76, 201, 194, 0.68);
      outline-offset: 2px;
    }
    .pipeline-step:disabled {
      cursor: default;
      opacity: 0.78;
    }
    .pipeline-step .step-title {
      font-weight: 700;
      font-size: 13px;
    }
    .pipeline-step .step-state {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: rgba(236, 246, 255, 0.7);
    }
    .pipeline-step.is-neutral {
      background: rgba(255, 255, 255, 0.03);
      border-color: rgba(255, 255, 255, 0.1);
      color: rgba(241, 245, 249, 0.78);
    }
    .pipeline-step.is-running {
      background: rgba(76, 201, 194, 0.16);
      color: #dffcf9;
      border-color: rgba(76, 201, 194, 0.34);
    }
    .pipeline-step.is-running .step-state {
      color: rgba(223, 252, 249, 0.72);
    }
    .pipeline-step.is-done {
      background: rgba(148, 163, 184, 0.1);
      color: #e2e8f0;
      border-color: rgba(148, 163, 184, 0.22);
    }
    .pipeline-step.is-done .step-state {
      color: rgba(226, 232, 240, 0.78);
    }
    .pipeline-step.is-error {
      background: rgba(236, 72, 153, 0.12);
      color: #fbcfe8;
      border-color: rgba(236, 72, 153, 0.22);
    }
    .pipeline-step.is-error .step-state {
      color: rgba(251, 207, 232, 0.78);
    }
    .pipeline-step.is-blocked {
      opacity: 0.5;
    }
    .workflow-inner {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .workflow-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 11px;
      font-size: 12px;
      letter-spacing: 0.02em;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: rgba(241, 245, 249, 0.7);
      transition: background .18s ease, border-color .18s ease, color .18s ease, transform .18s ease;
    }
    .workflow-chip::before {
      content: "";
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: rgba(241, 245, 249, 0.28);
    }
    .workflow-chip.is-running {
      background: rgba(76, 201, 194, 0.12);
      border-color: rgba(76, 201, 194, 0.24);
      color: #dffcf9;
      transform: translateY(-1px);
    }
    .workflow-chip.is-running::before {
      background: rgba(76, 201, 194, 0.9);
    }
    .workflow-chip.is-done {
      background: rgba(148, 163, 184, 0.08);
      border-color: rgba(148, 163, 184, 0.18);
      color: #e2e8f0;
    }
    .workflow-chip.is-done::before {
      background: rgba(148, 163, 184, 0.78);
    }
    .workflow-chip.is-error {
      background: rgba(236, 72, 153, 0.1);
      border-color: rgba(236, 72, 153, 0.2);
      color: #fbcfe8;
    }
    .workflow-chip.is-error::before {
      background: rgba(236, 72, 153, 0.88);
    }
    .workflow-chip.is-blocked {
      opacity: 0.45;
    }
    .btn, .ghost, .quick-btn {
      border: none;
      cursor: pointer;
      border-radius: 14px;
      padding: 12px 14px;
      font-weight: 600;
      transition: transform .18s ease, opacity .18s ease, border-color .18s ease, background .18s ease;
    }
    .btn:hover, .ghost:hover, .quick-btn:hover { transform: translateY(-1px); }
    .btn {
      color: #06111d;
      background: linear-gradient(180deg, rgba(76, 201, 194, 0.96), rgba(76, 201, 194, 0.82));
      box-shadow: none;
    }
    .ghost {
      color: var(--text);
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .mini-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-top: 14px;
    }
    .mini-stat {
      padding: 12px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .mini-stat .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .12em; }
    .mini-stat .value { font-size: 22px; font-weight: 800; margin-top: 6px; }
    .status-card {
      display: grid;
      gap: 14px;
    }
    .status-title { margin: 0; font-size: 17px; line-height: 1.3; }
    .status-line {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      font-size: 14px;
      color: var(--muted);
    }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }
    .form-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }
    .field {
      display: grid;
      gap: 8px;
    }
    label {
      font-size: 13px;
      color: var(--muted);
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    input[type="text"], input[type="number"], textarea, select {
      width: 100%;
      color: var(--text);
      background: rgba(2, 6, 23, 0.38);
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 14px;
      padding: 13px 14px;
      outline: none;
      transition: border-color .18s ease, transform .18s ease;
    }
    input[type="file"] {
      width: 100%;
      color: var(--muted);
    }
    textarea { min-height: 120px; resize: vertical; }
    input:focus, textarea:focus, select:focus {
      border-color: rgba(125, 211, 252, 0.85);
      box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.12);
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .stack { display: grid; gap: 12px; }
    .log {
      max-height: 200px;
      overflow: auto;
      display: grid;
      gap: 8px;
      padding-right: 4px;
    }
    .log-item {
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.08);
      color: rgba(241, 245, 249, 0.84);
      font-size: 13px;
    }
    .results-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .result-tile {
      padding: 12px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.1);
      min-height: 90px;
    }
    .result-tile .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .12em; }
    .result-tile .value { margin-top: 8px; font-size: 18px; font-weight: 700; }
    .result-tile .small { margin-top: 6px; color: rgba(241, 245, 249, 0.72); font-size: 12.5px; line-height: 1.45; }
    .links {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .link-btn {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      text-decoration: none;
      color: var(--text);
      background: rgba(255, 255, 255, 0.03);
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      font-weight: 600;
    }
    .chat-card { display: grid; gap: 12px; position: relative; overflow: hidden; }
    .chat-card::before {
      content: "Nagabhushana Raju S";
      position: absolute;
      top: 18px;
      right: -12px;
      font-size: 28px;
      font-weight: 800;
      letter-spacing: 0.14em;
      color: rgba(241, 245, 249, 0.04);
      transform: rotate(-14deg);
      animation: driftBrand 36s ease-in-out infinite;
      pointer-events: none;
      white-space: nowrap;
    }
    .chat-card > * {
      position: relative;
      z-index: 1;
    }
    .chat-history {
      min-height: 260px;
      max-height: 400px;
      overflow: auto;
      padding: 14px;
      display: grid;
      gap: 10px;
      background: rgba(255, 255, 255, 0.025);
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.1);
      position: relative;
    }
    .bubble {
      max-width: 92%;
      padding: 12px 14px;
      border-radius: 14px;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13.5px;
      border: 1px solid rgba(255,255,255,0.1);
    }
    .bubble.user {
      justify-self: end;
      background: rgba(76, 201, 194, 0.16);
      color: var(--text);
      border-color: rgba(76, 201, 194, 0.24);
    }
    .bubble.bot {
      justify-self: start;
      background: rgba(255,255,255,0.03);
      color: var(--text);
    }
    .bubble .meta {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .1em;
      opacity: 0.75;
      margin-bottom: 6px;
    }
    .quick-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .quick-btn {
      padding: 8px 11px;
      font-size: 12.5px;
      color: var(--text);
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.1);
    }
    .explorer-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }
    .frame-card {
      min-height: 540px;
      display: grid;
      gap: 10px;
    }
    .frame-card iframe {
      width: 100%;
      min-height: 470px;
      border: none;
      border-radius: 14px;
      background: rgba(255,255,255,0.96);
    }
    .footer {
      margin: 18px 0 10px;
      text-align: center;
      color: rgba(241, 245, 249, 0.66);
      font-size: 13px;
    }
    .signature {
      display: inline-flex;
      gap: 10px;
      align-items: center;
      margin-top: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.1);
    }
    .signature::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--mint), var(--violet));
      box-shadow: 0 0 0 0 rgba(76, 201, 194, 0.35);
      animation: ping 2.4s infinite;
    }
    @media (max-width: 1120px) {
      .hero, .grid-2, .explorer-grid, .results-grid { grid-template-columns: 1fr; }
      .form-grid { grid-template-columns: 1fr; }
      .mini-grid { grid-template-columns: 1fr; }
      .frame-card { min-height: 420px; }
    }
  </style>
</head>
<body>
  <div class="brand-backdrop" aria-hidden="true">
    <span class="brand-a">Nagabhushana Raju S</span>
    <span class="brand-b">Retail Insights Studio</span>
  </div>
  <div class="blob one"></div>
  <div class="blob two"></div>
  <div class="blob three"></div>
  <div class="wrap">
    <header class="topbar">
      <div class="brand">
        <div class="logo"></div>
        <div>
          <h1>__APP_TITLE__</h1>
          <p>Non-technical dashboard for retail analysis, model training, chat, and visual exploration.</p>
        </div>
      </div>
      <div class="badge"><span class="dot"></span> Local UI ready</div>
    </header>

    <section class="hero">
      <div class="card card-inner hero-copy">
        <div class="subtitle">One click, clear steps, clean results.</div>
        <h2 class="headline">A friendly retail workspace for <span>non-tech users</span>.</h2>
        <p class="lead">Upload a dataset or use the demo. The pipeline keeps the flow simple: clean, analyze, train, and chat.</p>
        <div class="mode-strip">
          <div class="mode-row">
            <div class="field mode-field">
              <label for="workflowMode">Workflow mode</label>
              <select id="workflowMode">
                <option value="automatic">Automatic</option>
                <option value="manual">Manual</option>
              </select>
            </div>
            <div class="mode-help" id="modeHelp">Automatic runs everything in one click. Manual walks through upload, clean, and train.</div>
          </div>
        </div>
        <div class="workflow-frame">
          <div class="workflow-head">
            <div>
              <div class="pipeline-label">Data Flow Diagram</div>
              <div class="workflow-note" id="workflowNote">Click the highlighted node to continue.</div>
            </div>
            <div class="workflow-tools">
              <div class="tool-stack">
                <button id="loadDemoBtn" class="ghost workflow-tool" title="Load the built-in sample dataset instead of uploading a file.">Use Demo Data</button>
                <span class="tool-note">Loads sample retail rows and keeps the demo moving.</span>
              </div>
              <div class="tool-stack">
                <button id="refreshBtn" class="ghost workflow-tool" title="Pull the latest job state, messages, and outputs from the server.">Refresh Status</button>
                <span class="tool-note">Fetch the latest run state without restarting anything.</span>
              </div>
            </div>
          </div>
          <div class="pipeline-track" id="automaticPipeline">
            <button id="startBtn" class="pipeline-step step-primary" type="button" title="Run load, cleaning, EDA, and training in one click.">
              <span class="step-title">Run Full Flow</span><span class="step-state">Pending</span>
            </button>
            <span class="pipeline-arrow" aria-hidden="true">→</span>
            <button class="pipeline-step" type="button" data-step="load" title="Load the selected dataset or the demo dataset into the pipeline.">
              <span class="step-title">Load</span><span class="step-state">Pending</span>
            </button>
            <span class="pipeline-arrow" aria-hidden="true">→</span>
            <button class="pipeline-step" type="button" data-step="cleaning" title="Clean missing values, types, and basic outliers.">
              <span class="step-title">Clean</span><span class="step-state">Pending</span>
            </button>
            <span class="pipeline-arrow" aria-hidden="true">→</span>
            <button class="pipeline-step" type="button" data-step="eda" title="Generate exploratory analysis and summary insights.">
              <span class="step-title">EDA</span><span class="step-state">Pending</span>
            </button>
            <span class="pipeline-arrow" aria-hidden="true">→</span>
            <button class="pipeline-step" type="button" data-step="training" title="Train the model and score it against the validation split.">
              <span class="step-title">Train</span><span class="step-state">Pending</span>
            </button>
            <span class="pipeline-arrow" aria-hidden="true">→</span>
            <button class="pipeline-step" type="button" data-step="complete" title="Show that the flow finished and the outputs are ready.">
              <span class="step-title">Ready</span><span class="step-state">Pending</span>
            </button>
          </div>
          <div class="workflow-inner" id="automaticInner">
            <span class="workflow-chip" data-inner-step="ingest">Ingest</span>
            <span class="workflow-chip" data-inner-step="normalize">Normalize</span>
            <span class="workflow-chip" data-inner-step="feature">Feature Build</span>
            <span class="workflow-chip" data-inner-step="split">Split</span>
            <span class="workflow-chip" data-inner-step="fit">Fit</span>
            <span class="workflow-chip" data-inner-step="score">Score</span>
            <span class="workflow-chip" data-inner-step="export">Export</span>
          </div>
          <div class="pipeline-track hidden" id="manualPipeline">
            <button id="stageFileBtn" class="pipeline-step step-primary" type="button" title="Open the file picker and upload a dataset for manual mode.">
              <span class="step-title">Upload Data</span><span class="step-state">Pending</span>
            </button>
            <span class="pipeline-arrow" aria-hidden="true">→</span>
            <button id="cleanEdaBtn" class="pipeline-step" type="button" title="Clean the uploaded file and generate EDA results.">
              <span class="step-title">Clean & EDA</span><span class="step-state">Pending</span>
            </button>
            <span class="pipeline-arrow" aria-hidden="true">→</span>
            <button id="trainModelBtn" class="pipeline-step" type="button" title="Train the model after the cleaned data is ready.">
              <span class="step-title">Train Model</span><span class="step-state">Pending</span>
            </button>
            <span class="pipeline-arrow" aria-hidden="true">→</span>
            <button class="pipeline-step" type="button" data-step="complete" title="Mark the manual workflow as finished and ready to reuse.">
              <span class="step-title">Ready</span><span class="step-state">Pending</span>
            </button>
          </div>
          <div class="workflow-inner hidden" id="manualInner">
            <span class="workflow-chip" data-inner-step="upload">Upload</span>
            <span class="workflow-chip" data-inner-step="normalize">Normalize</span>
            <span class="workflow-chip" data-inner-step="eda">EDA</span>
            <span class="workflow-chip" data-inner-step="split">Split</span>
            <span class="workflow-chip" data-inner-step="fit">Fit</span>
            <span class="workflow-chip" data-inner-step="score">Score</span>
            <span class="workflow-chip" data-inner-step="ready">Ready</span>
          </div>
        </div>
        <div class="mini-grid">
          <div class="mini-stat"><div class="label">Run mode</div><div class="value" id="topProgress">Automatic</div></div>
          <div class="mini-stat"><div class="label">Pipeline stage</div><div class="value" id="topStage">Idle</div></div>
          <div class="mini-stat"><div class="label">Model state</div><div class="value" id="topModelState">Waiting</div></div>
        </div>
      </div>
      <div class="card card-inner status-card">
      <h3 class="status-title">Workflow status</h3>
        <div class="status-line">
          <span id="statusMessage">Ready. Choose a mode to begin.</span>
          <span id="statusText">idle</span>
        </div>
        <div class="stack">
          <div class="field">
            <label for="datasetFile">Dataset file</label>
            <input id="datasetFile" type="file" accept=",.csv,.xlsx,.xls,.json,.txt" />
            <div class="hint">If you do not upload a file, the app uses synthetic retail data so the demo always works.</div>
          </div>
          <div class="form-grid">
            <div class="field">
              <label for="demoRows">Demo rows</label>
              <input id="demoRows" type="number" min="100" step="100" value="1200" />
            </div>
            <div class="field">
              <label for="seedValue">Random seed</label>
              <input id="seedValue" type="number" value="42" />
            </div>
          </div>
        </div>
        <div class="log" id="statusLog"></div>
      </div>
    </section>

    <section class="grid-2">
      <div class="card card-inner">
        <h3 class="glass-title">Results</h3>
        <div class="results-grid" id="resultsGrid">
          <div class="result-tile"><div class="label">Status</div><div class="value" id="statusTileValue">Waiting</div><div class="small" id="statusTileSmall">Run the pipeline to populate analytics.</div></div>
          <div class="result-tile"><div class="label">Rows</div><div class="value" id="rowsTileValue">-</div><div class="small">Rows processed after cleaning.</div></div>
          <div class="result-tile"><div class="label">Best model</div><div class="value" id="modelTileValue">-</div><div class="small">Selected by holdout RMSE.</div></div>
          <div class="result-tile"><div class="label">Metrics</div><div class="value" id="metricsTileValue">-</div><div class="small">MAE / RMSE / R2.</div></div>
        </div>
        <div class="links" id="downloadLinks"></div>
      </div>
      <div class="card card-inner chat-card">
        <h3 class="glass-title">Chat with the model</h3>
        <div class="hint">Ask for a prediction or a summary in plain English. Example: Predict sales for quantity 3 price 120 category Electronics region West date 2025-04-01.</div>
        <div class="chat-history" id="chatHistory"></div>
        <textarea id="chatInput" placeholder="Type your question or prediction scenario here..."></textarea>
        <div class="hero-actions">
          <button id="sendChatBtn" class="btn">Send</button>
          <button id="clearChatBtn" class="ghost">Clear Chat</button>
        </div>
        <div class="quick-actions">
          <button class="quick-btn" data-prompt="Show model metrics">Model metrics</button>
          <button class="quick-btn" data-prompt="What are the top products?">Top products</button>
          <button class="quick-btn" data-prompt="What are the top regions?">Top regions</button>
          <button class="quick-btn" data-prompt="Show sales trend">Seasonality</button>
          <button class="quick-btn" data-prompt="Predict sales for quantity 3 price 120 category Electronics region West date 2025-04-01">Predict sales</button>
        </div>
      </div>
    </section>

    <section class="explorer-grid">
      <div class="card card-inner frame-card">
        <h3 class="glass-title">Multicolor Explorer</h3>
        <p class="muted">An interactive retail scatter plot with bright colors and hover details.</p>
        <iframe id="scatterFrame" title="Multicolor explorer" src="about:blank"></iframe>
      </div>
      <div class="card card-inner frame-card">
        <h3 class="glass-title">Interactive 3D Explorer</h3>
        <p class="muted">Change the axes and color grouping, then refresh the plot.</p>
        <div class="form-grid">
          <div class="field">
            <label for="axisX">X axis</label>
            <select id="axisX"></select>
          </div>
          <div class="field">
            <label for="axisY">Y axis</label>
            <select id="axisY"></select>
          </div>
          <div class="field">
            <label for="axisZ">Z axis</label>
            <select id="axisZ"></select>
          </div>
          <div class="field">
            <label for="colorAxis">Color by</label>
            <select id="colorAxis"></select>
          </div>
        </div>
        <div class="hero-actions" style="margin-top: 8px;">
          <button id="update3dBtn" class="ghost">Update 3D Plot</button>
        </div>
        <iframe id="plot3dFrame" title="Interactive 3D explorer" src="about:blank"></iframe>
      </div>
    </section>

    <div class="footer">
      Built for easy use by non-technical users.
      <div class="signature">__AUTHOR__</div>
    </div>
  </div>

  <script>
    const stateEndpoints = {
      status: '/api/status',
      start: '/api/start',
      mode: '/api/mode',
      manualUpload: '/api/manual/upload',
      manualAnalyze: '/api/manual/analyze',
      manualTrain: '/api/manual/train',
      chat: '/api/chat'
    };
    const elements = {
      workflowMode: document.getElementById('workflowMode'),
      modeHelp: document.getElementById('modeHelp'),
      automaticPipeline: document.getElementById('automaticPipeline'),
      manualPipeline: document.getElementById('manualPipeline'),
      automaticInner: document.getElementById('automaticInner'),
      manualInner: document.getElementById('manualInner'),
      workflowNote: document.getElementById('workflowNote'),
      startBtn: document.getElementById('startBtn'),
      loadDemoBtn: document.getElementById('loadDemoBtn'),
      stageFileBtn: document.getElementById('stageFileBtn'),
      cleanEdaBtn: document.getElementById('cleanEdaBtn'),
      trainModelBtn: document.getElementById('trainModelBtn'),
      refreshBtn: document.getElementById('refreshBtn'),
      datasetFile: document.getElementById('datasetFile'),
      demoRows: document.getElementById('demoRows'),
      seedValue: document.getElementById('seedValue'),
      statusMessage: document.getElementById('statusMessage'),
      statusText: document.getElementById('statusText'),
      statusLog: document.getElementById('statusLog'),
      topProgress: document.getElementById('topProgress'),
      topStage: document.getElementById('topStage'),
      topModelState: document.getElementById('topModelState'),
      statusTileValue: document.getElementById('statusTileValue'),
      statusTileSmall: document.getElementById('statusTileSmall'),
      rowsTileValue: document.getElementById('rowsTileValue'),
      modelTileValue: document.getElementById('modelTileValue'),
      metricsTileValue: document.getElementById('metricsTileValue'),
      downloadLinks: document.getElementById('downloadLinks'),
      chatHistory: document.getElementById('chatHistory'),
      chatInput: document.getElementById('chatInput'),
      sendChatBtn: document.getElementById('sendChatBtn'),
      clearChatBtn: document.getElementById('clearChatBtn'),
      scatterFrame: document.getElementById('scatterFrame'),
      plot3dFrame: document.getElementById('plot3dFrame'),
      axisX: document.getElementById('axisX'),
      axisY: document.getElementById('axisY'),
      axisZ: document.getElementById('axisZ'),
      colorAxis: document.getElementById('colorAxis'),
      update3dBtn: document.getElementById('update3dBtn')
    };

    const quickPrompts = Array.from(document.querySelectorAll('.quick-btn'));
    let pollTimer = null;
    let appReady = false;
    let pendingManualUpload = false;

    function clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }

    function appendStatusLine(text) {
      const item = document.createElement('div');
      item.className = 'log-item';
      item.textContent = text;
      elements.statusLog.prepend(item);
      while (elements.statusLog.children.length > 8) {
        elements.statusLog.removeChild(elements.statusLog.lastChild);
      }
    }

    function addChat(role, text, kind='bot') {
      const bubble = document.createElement('div');
      bubble.className = `bubble ${kind}`;
      bubble.innerHTML = `<div class="meta">${escapeHtml(role)}</div><div>${escapeHtml(text)}</div>`;
      elements.chatHistory.appendChild(bubble);
      elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
    }

    function setProgress(progress, state = {}) {
      if (!elements.topProgress) {
        return;
      }
      if (state.status === 'running') {
        elements.topProgress.textContent = 'Running';
        return;
      }
      if (state.status === 'done') {
        elements.topProgress.textContent = 'Done';
        return;
      }
      if (state.status === 'error') {
        elements.topProgress.textContent = 'Error';
        return;
      }
      elements.topProgress.textContent = state.workflow_mode === 'manual' ? 'Manual' : 'Automatic';
    }

    function updateDownloadLinks(links) {
      const entries = Object.entries(links || {});
      elements.downloadLinks.innerHTML = '';
      if (!entries.length) {
        return;
      }
      for (const [label, url] of entries) {
        const anchor = document.createElement('a');
        anchor.className = 'link-btn';
        anchor.href = url;
        anchor.target = '_blank';
        anchor.rel = 'noreferrer';
        anchor.textContent = label;
        elements.downloadLinks.appendChild(anchor);
      }
    }

    function formatMetricValue(value) {
      if (value === null || value === undefined || value === '') {
        return '-';
      }
      if (typeof value === 'number') {
        return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
      }
      return String(value);
    }

    function workflowHelpText(mode) {
      if (mode === 'manual') {
        return 'Manual mode: upload, clean, then train step by step.';
      }
      return 'Automatic mode: run the full pipeline in one click.';
    }

    const automaticPipelineSteps = [
      { key: 'start', label: 'Run Full Flow' },
      { key: 'load', label: 'Load' },
      { key: 'cleaning', label: 'Clean' },
      { key: 'eda', label: 'EDA' },
      { key: 'training', label: 'Train' },
      { key: 'complete', label: 'Ready' }
    ];

    const manualPipelineSteps = [
      { key: 'upload', label: 'Upload Data' },
      { key: 'cleaning', label: 'Clean & EDA' },
      { key: 'training', label: 'Train' },
      { key: 'complete', label: 'Ready' }
    ];

    const automaticInnerSteps = [
      { key: 'ingest', label: 'Ingest' },
      { key: 'normalize', label: 'Normalize' },
      { key: 'feature', label: 'Feature Build' },
      { key: 'split', label: 'Split' },
      { key: 'fit', label: 'Fit' },
      { key: 'score', label: 'Score' },
      { key: 'export', label: 'Export' }
    ];

    const manualInnerSteps = [
      { key: 'upload', label: 'Upload' },
      { key: 'normalize', label: 'Normalize' },
      { key: 'eda', label: 'EDA' },
      { key: 'split', label: 'Split' },
      { key: 'fit', label: 'Fit' },
      { key: 'score', label: 'Score' },
      { key: 'ready', label: 'Ready' }
    ];

    function normalizeStatusText(value) {
      return String(value || '').toLowerCase();
    }

    function resolveAutomaticTrackerState(state) {
      const stage = normalizeStatusText(`${state.stage || ''} ${state.workflow_step || ''}`);
      if (state.status === 'done' || stage.includes('complete') || stage.includes('done')) {
        return { doneThrough: 5, activeIndex: 5 };
      }
      if (state.status === 'error') {
        if (stage.includes('model') || stage.includes('train')) {
          return { doneThrough: 3, activeIndex: 4, error: true };
        }
        if (stage.includes('eda') || stage.includes('summary') || stage.includes('report')) {
          return { doneThrough: 2, activeIndex: 3, error: true };
        }
        if (stage.includes('clean')) {
          return { doneThrough: 1, activeIndex: 2, error: true };
        }
        if (stage.includes('load') || stage.includes('setup') || stage.includes('generate') || stage.includes('queued') || stage.includes('starting')) {
          return { doneThrough: 0, activeIndex: 1, error: true };
        }
        return { doneThrough: -1, activeIndex: 0, error: true };
      }
      if (stage.includes('model') || stage.includes('train') || stage.includes('modeling')) {
        return { doneThrough: 3, activeIndex: 4 };
      }
      if (stage.includes('eda') || stage.includes('summary') || stage.includes('report')) {
        return { doneThrough: 2, activeIndex: 3 };
      }
      if (stage.includes('clean')) {
        return { doneThrough: 1, activeIndex: 2 };
      }
      if (stage.includes('load') || stage.includes('setup') || stage.includes('generate') || stage.includes('queued') || stage.includes('starting')) {
        return { doneThrough: 0, activeIndex: 1 };
      }
      return state.status === 'running' ? { doneThrough: 0, activeIndex: 1 } : { doneThrough: -1, activeIndex: 0 };
    }

    function resolveManualTrackerState(state) {
      const stage = normalizeStatusText(`${state.stage || ''} ${state.workflow_step || ''}`);
      if (state.status === 'done' || stage.includes('trained') || stage.includes('manual_complete') || stage.includes('complete')) {
        return { doneThrough: 3, activeIndex: 3 };
      }
      if (state.status === 'error') {
        if (stage.includes('train')) {
          return { doneThrough: 1, activeIndex: 2, error: true };
        }
        if (stage.includes('analysis') || stage.includes('clean')) {
          return { doneThrough: 0, activeIndex: 1, error: true };
        }
        return { doneThrough: -1, activeIndex: 0, error: true };
      }
      if (stage.includes('training')) {
        return { doneThrough: 1, activeIndex: 2 };
      }
      if (stage.includes('analysis_ready')) {
        return { doneThrough: 1, activeIndex: null };
      }
      if (stage.includes('analysis') || stage.includes('clean') || stage.includes('manual_queue')) {
        return { doneThrough: 0, activeIndex: 1 };
      }
      if (stage.includes('uploaded')) {
        return { doneThrough: 0, activeIndex: null };
      }
      return state.status === 'running' ? { doneThrough: 0, activeIndex: 1 } : { doneThrough: -1, activeIndex: 0 };
    }

    function setWorkflowNodeTitle(node, title) {
      if (!node) {
        return;
      }
      const titleNode = node.querySelector('.step-title');
      if (titleNode) {
        titleNode.textContent = title;
      } else {
        node.textContent = title;
      }
    }

    function setWorkflowNodeState(node, label) {
      if (!node) {
        return;
      }
      const stateNode = node.querySelector('.step-state');
      if (stateNode) {
        stateNode.textContent = label;
      }
    }

    function isWorkflowButtonClickable(container, button, state, trackerState) {
      if (state.status === 'running') {
        return false;
      }
      if (!container) {
        return false;
      }
      if (container.id === 'automaticPipeline') {
        return button.id === 'startBtn';
      }
      if (container.id === 'manualPipeline') {
        if (button.id === 'stageFileBtn') {
          return true;
        }
        if (button.id === 'cleanEdaBtn') {
          return state.workflow_step === 'uploaded';
        }
        if (button.id === 'trainModelBtn') {
          return state.workflow_step === 'analysis_ready';
        }
      }
      return false;
    }

    function resolveInnerTrackerState(state, mode) {
      const stage = normalizeStatusText(`${state.stage || ''} ${state.workflow_step || ''}`);
      const innerSteps = mode === 'manual' ? manualInnerSteps : automaticInnerSteps;
      const lastIndex = innerSteps.length - 1;

      if (state.status === 'done' || stage.includes('complete') || stage.includes('done') || stage.includes('trained') || stage.includes('manual_complete')) {
        return { doneThrough: lastIndex, activeIndex: lastIndex };
      }

      if (state.status === 'error') {
        if (mode === 'manual') {
          if (stage.includes('train')) {
            return { doneThrough: 4, activeIndex: 5, error: true };
          }
          if (stage.includes('analysis') || stage.includes('clean')) {
            return { doneThrough: 1, activeIndex: 2, error: true };
          }
          return { doneThrough: -1, activeIndex: 0, error: true };
        }
        if (stage.includes('model') || stage.includes('train')) {
          return { doneThrough: 3, activeIndex: 4, error: true };
        }
        if (stage.includes('eda') || stage.includes('summary') || stage.includes('report')) {
          return { doneThrough: 2, activeIndex: 3, error: true };
        }
        if (stage.includes('clean')) {
          return { doneThrough: 1, activeIndex: 2, error: true };
        }
        return { doneThrough: 0, activeIndex: 1, error: true };
      }

      if (mode === 'manual') {
        if (stage.includes('training')) {
          return { doneThrough: 4, activeIndex: 5 };
        }
        if (stage.includes('analysis_ready')) {
          return { doneThrough: 2, activeIndex: 3 };
        }
        if (stage.includes('analysis') || stage.includes('clean') || stage.includes('manual_queue')) {
          return { doneThrough: 1, activeIndex: 2 };
        }
        if (stage.includes('uploaded')) {
          return { doneThrough: 0, activeIndex: 1 };
        }
        if (stage.includes('waiting_upload') || stage.includes('idle')) {
          return { doneThrough: -1, activeIndex: 0 };
        }
        return state.status === 'running' ? { doneThrough: 0, activeIndex: 1 } : { doneThrough: -1, activeIndex: 0 };
      }

      if (stage.includes('model') || stage.includes('train') || stage.includes('modeling')) {
        return { doneThrough: 4, activeIndex: 5 };
      }
      if (stage.includes('eda') || stage.includes('summary') || stage.includes('report')) {
        return { doneThrough: 2, activeIndex: 3 };
      }
      if (stage.includes('clean')) {
        return { doneThrough: 1, activeIndex: 2 };
      }
      if (stage.includes('load') || stage.includes('setup') || stage.includes('generate') || stage.includes('queued') || stage.includes('starting')) {
        return { doneThrough: 0, activeIndex: 1 };
      }
      return state.status === 'running' ? { doneThrough: 0, activeIndex: 1 } : { doneThrough: -1, activeIndex: 0 };
    }

    function paintPipeline(container, steps, trackerState, state) {
      if (!container) {
        return;
      }
      const buttons = Array.from(container.querySelectorAll('.pipeline-step'));
      const arrows = Array.from(container.querySelectorAll('.pipeline-arrow'));
      buttons.forEach((button, index) => {
        const stateNode = button.querySelector('.step-state');
        const step = steps[index] || { label: button.textContent.trim() };
        const clickable = isWorkflowButtonClickable(container, button, state, trackerState);
        button.disabled = !clickable;
        button.setAttribute('aria-disabled', String(!clickable));
        if (container.id === 'automaticPipeline' && button.id === 'startBtn') {
          setWorkflowNodeTitle(button, state.status === 'done' ? 'Run Again' : step.label);
        } else {
          setWorkflowNodeTitle(button, step.label);
        }
        button.classList.remove('is-neutral', 'is-running', 'is-done', 'is-error', 'is-blocked');
        let visualState = 'is-neutral';
        let label = clickable ? 'Ready' : 'Pending';
        if (trackerState.error && index === trackerState.activeIndex) {
          visualState = 'is-error';
          label = 'Stopped';
        } else if (state.status === 'done' || index <= trackerState.doneThrough) {
          visualState = 'is-done';
          label = 'Done';
        } else if (state.status === 'running' && index === trackerState.activeIndex) {
          visualState = 'is-running';
          label = 'Running';
        } else if (trackerState.activeIndex !== null && index > trackerState.activeIndex) {
          visualState = 'is-blocked';
          label = 'Blocked';
        }
        button.classList.add(visualState);
        if (stateNode) {
          stateNode.textContent = label;
        }
      });
      arrows.forEach((arrow, index) => {
        arrow.classList.remove('is-neutral', 'is-running', 'is-done', 'is-error', 'is-blocked');
        let visualState = 'is-neutral';
        if (trackerState.error && trackerState.activeIndex !== null && index === Math.max(0, trackerState.activeIndex - 1)) {
          visualState = 'is-error';
        } else if (state.status === 'done' || index < trackerState.doneThrough) {
          visualState = 'is-done';
        } else if (state.status === 'running' && trackerState.activeIndex !== null && index === Math.max(0, trackerState.activeIndex - 1)) {
          visualState = 'is-running';
        } else if (trackerState.activeIndex !== null && index >= trackerState.activeIndex) {
          visualState = 'is-blocked';
        }
        arrow.classList.add(visualState);
      });
    }

    function paintInnerWorkflow(container, steps, trackerState, state) {
      if (!container) {
        return;
      }
      const chips = Array.from(container.querySelectorAll('.workflow-chip'));
      chips.forEach((chip, index) => {
        const step = steps[index] || { label: chip.textContent.trim() };
        chip.textContent = step.label;
        chip.classList.remove('is-neutral', 'is-running', 'is-done', 'is-error', 'is-blocked');
        let visualState = 'is-neutral';
        if (trackerState.error && index === trackerState.activeIndex) {
          visualState = 'is-error';
        } else if (state.status === 'done' || index <= trackerState.doneThrough) {
          visualState = 'is-done';
        } else if (state.status === 'running' && index === trackerState.activeIndex) {
          visualState = 'is-running';
        } else if (trackerState.activeIndex !== null && index > trackerState.activeIndex) {
          visualState = 'is-blocked';
        }
        chip.classList.add(visualState);
      });
    }

    function workflowStatusNote(state) {
      const mode = state.workflow_mode === 'manual' ? 'manual' : 'automatic';
      const stage = normalizeStatusText(`${state.stage || ''} ${state.workflow_step || ''}`);
      if (state.status === 'running') {
        return `Running ${state.workflow_step || state.stage || 'step'}...`;
      }
      if (state.status === 'done') {
        return mode === 'manual'
          ? 'Manual flow complete. Upload another file to run again.'
          : 'Automatic flow complete. Click Run Full Flow to run again.';
      }
      if (mode === 'manual') {
        if (stage.includes('analysis_ready')) {
          return 'Cleaned data is ready. Click Train Model.';
        }
        if (stage.includes('uploaded')) {
          return 'File uploaded. Click Clean & EDA.';
        }
        return 'Click the highlighted node to continue.';
      }
      return 'Click Run Full Flow to start the pipeline.';
    }

    function updateWorkflowTracker(state) {
      const mode = state.workflow_mode === 'manual' ? 'manual' : 'automatic';
      const automaticTracker = resolveAutomaticTrackerState(state);
      const manualTracker = resolveManualTrackerState(state);
      const automaticInnerTracker = resolveInnerTrackerState(state, 'automatic');
      const manualInnerTracker = resolveInnerTrackerState(state, 'manual');
      elements.automaticPipeline.classList.toggle('hidden', mode !== 'automatic');
      elements.manualPipeline.classList.toggle('hidden', mode !== 'manual');
      elements.automaticInner.classList.toggle('hidden', mode !== 'automatic');
      elements.manualInner.classList.toggle('hidden', mode !== 'manual');
      paintPipeline(elements.automaticPipeline, automaticPipelineSteps, automaticTracker, state);
      paintPipeline(elements.manualPipeline, manualPipelineSteps, manualTracker, state);
      paintInnerWorkflow(elements.automaticInner, automaticInnerSteps, automaticInnerTracker, state);
      paintInnerWorkflow(elements.manualInner, manualInnerSteps, manualInnerTracker, state);
      if (elements.workflowNote) {
        elements.workflowNote.textContent = workflowStatusNote(state);
      }
    }

    function applyWorkflowMode(mode) {
      const selectedMode = mode === 'manual' ? 'manual' : 'automatic';
      elements.workflowMode.value = selectedMode;
      elements.modeHelp.textContent = workflowHelpText(selectedMode);
      elements.automaticPipeline.classList.toggle('hidden', selectedMode !== 'automatic');
      elements.manualPipeline.classList.toggle('hidden', selectedMode !== 'manual');
      elements.automaticInner.classList.toggle('hidden', selectedMode !== 'automatic');
      elements.manualInner.classList.toggle('hidden', selectedMode !== 'manual');
      if (elements.workflowNote) {
        elements.workflowNote.textContent = workflowHelpText(selectedMode);
      }
    }

    async function setWorkflowMode(mode) {
      const response = await fetch(stateEndpoints.mode, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
      });
      const state = await response.json();
      applyWorkflowMode(state.workflow_mode || mode);
      updateResults(state);
      return state;
    }

    async function uploadManualDataset() {
      const file = elements.datasetFile.files[0];
      if (!file) {
        pendingManualUpload = true;
        elements.datasetFile.click();
        return;
      }
      pendingManualUpload = false;
      const formData = new FormData();
      formData.append('dataset_file', file);
      elements.stageFileBtn.disabled = true;
      elements.cleanEdaBtn.disabled = true;
      elements.trainModelBtn.disabled = true;
      try {
        const response = await fetch(stateEndpoints.manualUpload, {
          method: 'POST',
          body: formData
        });
        const state = await response.json();
        applyWorkflowMode(state.workflow_mode || 'manual');
        updateResults(state);
      } finally {
        elements.stageFileBtn.disabled = false;
      }
    }

    async function runManualAnalysis() {
      elements.stageFileBtn.disabled = true;
      elements.cleanEdaBtn.disabled = true;
      elements.trainModelBtn.disabled = true;
      const response = await fetch(stateEndpoints.manualAnalyze, { method: 'POST' });
      const state = await response.json();
      updateResults(state);
      if (response.ok) {
        if (pollTimer) {
          clearInterval(pollTimer);
        }
        pollTimer = setInterval(fetchStatus, 900);
      }
    }

    async function runManualTraining() {
      elements.stageFileBtn.disabled = true;
      elements.cleanEdaBtn.disabled = true;
      elements.trainModelBtn.disabled = true;
      const response = await fetch(stateEndpoints.manualTrain, { method: 'POST' });
      const state = await response.json();
      updateResults(state);
      if (response.ok) {
        if (pollTimer) {
          clearInterval(pollTimer);
        }
        pollTimer = setInterval(fetchStatus, 900);
      }
    }

    function updateResults(state) {
      const analysis = state.analysis || {};
      const model = state.model || {};
      const isRunning = state.status === 'running';
      elements.statusTileValue.textContent = state.status || 'idle';
      elements.statusTileSmall.textContent = state.message || 'Ready.';
      elements.rowsTileValue.textContent = formatMetricValue(analysis.rows_after || analysis.rows_before || '-');
      elements.modelTileValue.textContent = formatMetricValue(model.best_model || model.model_name || '-');
      const metricsText = model.best_r2 !== undefined && model.best_r2 !== null
        ? `R2 ${Number(model.best_r2).toFixed(3)} | RMSE ${Number(model.best_rmse).toFixed(2)}`
        : 'Waiting';
      elements.metricsTileValue.textContent = metricsText;
      elements.topStage.textContent = state.workflow_step || state.stage || 'idle';
      elements.topModelState.textContent = state.model_ready ? 'Ready' : (state.data_ready ? 'Data Ready' : 'Waiting');
      elements.statusMessage.textContent = state.message || 'Ready.';
      elements.statusText.textContent = state.workflow_step || state.status || 'idle';
      setProgress(state.progress || 0, state);
      updateDownloadLinks(state.artifacts || {});
      elements.statusLog.innerHTML = '';
      if (state.log && state.log.length) {
        state.log.slice().reverse().forEach(entry => appendStatusLine(entry));
      }
      appReady = Boolean(state.data_ready);
      elements.sendChatBtn.disabled = !appReady || isRunning;
      elements.update3dBtn.disabled = !state.data_ready || isRunning;
      elements.workflowMode.disabled = isRunning;
      elements.datasetFile.disabled = isRunning;
      elements.scatterFrame.src = state.data_ready ? '/plot/scatter' : 'about:blank';
      refresh3dFrame();
      applyWorkflowMode(state.workflow_mode || elements.workflowMode.value || 'automatic');
      updateWorkflowTracker(state);
      if ((state.workflow_mode || 'automatic') === 'manual') {
        elements.stageFileBtn.disabled = isRunning || false;
        elements.cleanEdaBtn.disabled = isRunning || state.workflow_step !== 'uploaded';
        elements.trainModelBtn.disabled = isRunning || state.workflow_step !== 'analysis_ready';
      } else {
        elements.stageFileBtn.disabled = true;
        elements.cleanEdaBtn.disabled = true;
        elements.trainModelBtn.disabled = true;
        elements.startBtn.disabled = isRunning;
        elements.loadDemoBtn.disabled = isRunning;
      }
      setWorkflowNodeTitle(elements.startBtn, state.status === 'done' ? 'Run Again' : 'Run Full Flow');
      elements.startBtn.disabled = isRunning;
    }

    function selectedAxisParams() {
      const params = new URLSearchParams({
        x: elements.axisX.value,
        y: elements.axisY.value,
        z: elements.axisZ.value,
        color: elements.colorAxis.value
      });
      return params.toString();
    }

    function refresh3dFrame() {
      if (!elements.axisX.value || !elements.axisY.value || !elements.axisZ.value) {
        return;
      }
      elements.plot3dFrame.src = `/plot/3d?${selectedAxisParams()}`;
    }

    function populateAxes(state) {
      const columns = state.available_columns || [];
      if (!columns.length) {
        return;
      }
      const numericColumns = state.numeric_columns || columns;
      const categoricalColumns = state.categorical_columns || [];
      const axisSelections = [elements.axisX, elements.axisY, elements.axisZ];
      axisSelections.forEach(select => {
        if (!select.options.length) {
          for (const column of numericColumns) {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            select.appendChild(option);
          }
        }
      });
      if (!elements.colorAxis.options.length) {
        const noneOption = document.createElement('option');
        noneOption.value = 'None';
        noneOption.textContent = 'None';
        elements.colorAxis.appendChild(noneOption);
        for (const column of categoricalColumns) {
          const option = document.createElement('option');
          option.value = column;
          option.textContent = column;
          elements.colorAxis.appendChild(option);
        }
      }
      const defaults = state.axis_defaults || {};
      if (defaults.x && [...elements.axisX.options].some(option => option.value === defaults.x)) elements.axisX.value = defaults.x;
      if (defaults.y && [...elements.axisY.options].some(option => option.value === defaults.y)) elements.axisY.value = defaults.y;
      if (defaults.z && [...elements.axisZ.options].some(option => option.value === defaults.z)) elements.axisZ.value = defaults.z;
      if (defaults.color && [...elements.colorAxis.options].some(option => option.value === defaults.color)) elements.colorAxis.value = defaults.color;
      if (!elements.colorAxis.value) elements.colorAxis.value = 'None';
      refresh3dFrame();
    }

    async function fetchStatus() {
      const response = await fetch(stateEndpoints.status, { cache: 'no-store' });
      const state = await response.json();
      populateAxes(state);
      updateResults(state);
      return state;
    }

    async function startPipeline(useDemo = false) {
      const formData = new FormData();
      const file = elements.datasetFile.files[0];
      if (file) {
        formData.append('dataset_file', file);
      }
      formData.append('demo_rows', elements.demoRows.value || '1200');
      formData.append('seed', elements.seedValue.value || '42');
      formData.append('use_demo', useDemo ? '1' : '0');
      elements.startBtn.disabled = true;
      elements.loadDemoBtn.disabled = true;
      elements.startBtn.textContent = 'Starting...';
      const response = await fetch(stateEndpoints.start, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      updateResults(data);
      if (pollTimer) {
        clearInterval(pollTimer);
      }
      pollTimer = setInterval(fetchStatus, 900);
      await fetchStatus();
    }

    async function sendChatMessage() {
      const message = elements.chatInput.value.trim();
      if (!message) {
        return;
      }
      addChat('You', message, 'user');
      elements.chatInput.value = '';
      const response = await fetch(stateEndpoints.chat, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });
      const data = await response.json();
      addChat(data.role || '__AUTHOR__', data.response || data.error || 'No response.', 'bot');
    }

    function clearChat() {
      elements.chatHistory.innerHTML = '';
      addChat('__AUTHOR__', 'Chat cleared. Ask for a prediction, model metrics, top products, or top regions.', 'bot');
    }

    elements.startBtn.addEventListener('click', () => startPipeline(false).catch(err => addChat('__AUTHOR__', err.message, 'bot')));
    elements.loadDemoBtn.addEventListener('click', () => startPipeline(true).catch(err => addChat('__AUTHOR__', err.message, 'bot')));
    elements.stageFileBtn.addEventListener('click', () => uploadManualDataset().catch(err => addChat('__AUTHOR__', err.message, 'bot')));
    elements.cleanEdaBtn.addEventListener('click', () => runManualAnalysis().catch(err => addChat('__AUTHOR__', err.message, 'bot')));
    elements.trainModelBtn.addEventListener('click', () => runManualTraining().catch(err => addChat('__AUTHOR__', err.message, 'bot')));
    elements.workflowMode.addEventListener('change', () => setWorkflowMode(elements.workflowMode.value).catch(err => addChat('__AUTHOR__', err.message, 'bot')));
    elements.refreshBtn.addEventListener('click', () => fetchStatus().catch(() => null));
    elements.sendChatBtn.addEventListener('click', () => sendChatMessage().catch(err => addChat('__AUTHOR__', err.message, 'bot')));
    elements.clearChatBtn.addEventListener('click', clearChat);
    elements.chatInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        sendChatMessage().catch(err => addChat('__AUTHOR__', err.message, 'bot'));
      }
    });

    elements.update3dBtn.addEventListener('click', () => refresh3dFrame());
    [elements.axisX, elements.axisY, elements.axisZ, elements.colorAxis].forEach(select => select.addEventListener('change', refresh3dFrame));
    quickPrompts.forEach(button => button.addEventListener('click', () => {
      elements.chatInput.value = button.dataset.prompt || '';
      elements.chatInput.focus();
    }));

    document.getElementById('datasetFile').addEventListener('change', () => {
      if (elements.workflowMode.value === 'manual') {
        if (pendingManualUpload && elements.datasetFile.files.length) {
          pendingManualUpload = false;
          uploadManualDataset().catch(err => addChat('__AUTHOR__', err.message, 'bot'));
          return;
        }
        elements.stageFileBtn.disabled = !elements.datasetFile.files.length;
      }
      if (elements.datasetFile.files.length) {
        elements.statusMessage.textContent = `Selected ${elements.datasetFile.files[0].name}`;
      }
    });

    clearChat();
    fetchStatus().catch(() => null);
    if (!pollTimer) {
      pollTimer = setInterval(fetchStatus, 1200);
    }
  </script>
</body>
</html>
"""


def set_state(**updates: Any) -> None:
    with STATE_LOCK:
        APP_STATE.update(updates)


def append_log(message: str) -> None:
    with STATE_LOCK:
        log = list(APP_STATE.get("log", []))
        log.append(message)
        APP_STATE["log"] = log[-12:]


def _json_safe(value: Any) -> Any:
  if value is None or value is pd.NA:
    return None
  if isinstance(value, dict):
    return {str(key): _json_safe(subvalue) for key, subvalue in value.items()}
  if isinstance(value, list):
    return [_json_safe(item) for item in value]
  if isinstance(value, tuple):
    return [_json_safe(item) for item in value]
  if isinstance(value, set):
    return [_json_safe(item) for item in value]
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, pd.DataFrame):
    preview = value.head(50).copy()
    preview.columns = [str(column) for column in preview.columns]
    return {
      "_type": "dataframe",
      "columns": preview.columns.tolist(),
      "row_count": int(len(value)),
      "column_count": int(value.shape[1]),
      "rows": [_json_safe(record) for record in preview.to_dict(orient="records")],
    }
  if isinstance(value, pd.Series):
    return {
      "_type": "series",
      "name": str(value.name) if value.name is not None else None,
      "values": [_json_safe(item) for item in value.head(50).tolist()],
    }
  if isinstance(value, (pd.Timestamp, pd.Period, pd.Timedelta)):
    return str(value)
  if isinstance(value, np.generic):
    scalar = value.item()
    if isinstance(scalar, float) and not np.isfinite(scalar):
      return None
    return scalar
  if isinstance(value, np.ndarray):
    return [_json_safe(item) for item in value.tolist()]
  if isinstance(value, float) and not np.isfinite(value):
    return None
  if isinstance(value, (str, int, bool)):
    return value
  if hasattr(value, "item"):
    try:
      scalar = value.item()
      if isinstance(scalar, float) and not np.isfinite(scalar):
        return None
      return scalar
    except Exception:
      pass
  return str(value)


def snapshot_state() -> dict[str, Any]:
  with STATE_LOCK:
    snapshot = dict(APP_STATE)
  return _json_safe(snapshot)


def reset_dashboard_state(workflow_mode: str, message: str, *, clear_upload: bool = True) -> None:
  global MANUAL_DATASET_PATH, MANUAL_DATASET_NAME
  MODEL_CONTEXT.clear()
  if clear_upload:
    MANUAL_DATASET_PATH = None
    MANUAL_DATASET_NAME = None
  set_state(
    status="idle",
    progress=0,
    stage="idle",
    message=message,
    workflow_mode=workflow_mode,
    workflow_step="idle",
    manual_dataset_name=None,
    log=[],
    analysis={},
    model={},
    artifacts={},
    model_ready=False,
    data_ready=False,
    last_error=None,
  )


def build_data_context(cleaned_df: pd.DataFrame) -> dict[str, Any]:
  frame = normalize_model_frame(cleaned_df)
  product_to_category = {}
  product_to_price = {}
  product_to_quantity = {}
  if "product" in frame.columns:
    if "category" in frame.columns:
      product_to_category = frame.groupby("product")["category"].agg(lambda series: series.mode(dropna=True).iloc[0] if not series.mode(dropna=True).empty else "Other").to_dict()
    if "price" in frame.columns:
      product_to_price = frame.groupby("product")["price"].median().to_dict()
    if "quantity" in frame.columns:
      product_to_quantity = frame.groupby("product")["quantity"].median().to_dict()

  top_products = pd.DataFrame()
  if "product" in frame.columns and "sales" in frame.columns:
    top_products = frame.groupby("product", dropna=False)["sales"].sum().sort_values(ascending=False).head(10).reset_index()

  top_regions = pd.DataFrame()
  if "region" in frame.columns and "sales" in frame.columns:
    top_regions = frame.groupby("region", dropna=False)["sales"].sum().sort_values(ascending=False).head(10).reset_index()

  monthly_sales = pd.DataFrame()
  if "date" in frame.columns and "sales" in frame.columns:
    monthly_sales = (
      frame.assign(month_period=frame["date"].dt.to_period("M"))
      .groupby("month_period")["sales"]
      .sum()
      .reset_index()
    )

  return {
    "cleaned_df": frame,
    "known_products": sorted(frame["product"].dropna().astype(str).unique().tolist()) if "product" in frame.columns else [],
    "known_categories": sorted(frame["category"].dropna().astype(str).unique().tolist()) if "category" in frame.columns else [],
    "known_regions": sorted(frame["region"].dropna().astype(str).unique().tolist()) if "region" in frame.columns else [],
    "product_to_category": product_to_category,
    "product_to_price": product_to_price,
    "product_to_quantity": product_to_quantity,
    "top_products": top_products,
    "top_regions": top_regions,
    "monthly_sales": monthly_sales,
  }


def build_dataset_overview(cleaned_df: pd.DataFrame) -> str:
  if cleaned_df.empty:
    return "No cleaned dataset is available yet."

  overview_parts: list[str] = [
    f"Rows: {len(cleaned_df):,}",
    f"Columns: {cleaned_df.shape[1]:,}",
  ]
  if "sales" in cleaned_df.columns:
    overview_parts.append(f"Total sales: {format_currency(float(cleaned_df['sales'].sum()))}")
  if "date" in cleaned_df.columns and cleaned_df["date"].notna().any():
    date_series = pd.to_datetime(cleaned_df["date"], errors="coerce").dropna()
    if not date_series.empty:
      overview_parts.append(f"Date range: {date_series.min().date()} to {date_series.max().date()}")
  if "category" in cleaned_df.columns:
    overview_parts.append(f"Categories: {cleaned_df['category'].nunique(dropna=True):,}")
  if "region" in cleaned_df.columns:
    overview_parts.append(f"Regions: {cleaned_df['region'].nunique(dropna=True):,}")
  return "\n".join(overview_parts)


def build_segment_summary(cleaned_df: pd.DataFrame, column_name: str, value: str) -> str:
  if cleaned_df.empty or column_name not in cleaned_df.columns:
    return "No matching data is available yet."

  mask = cleaned_df[column_name].astype(str).str.lower() == value.lower()
  subset = cleaned_df.loc[mask].copy()
  if subset.empty:
    return f"I could not find any rows for {value}."

  lines = [f"{column_name.title()} summary for {value}"]
  lines.append(f"Rows: {len(subset):,}")
  if "sales" in subset.columns:
    lines.append(f"Total sales: {format_currency(float(subset['sales'].sum()))}")
  if "quantity" in subset.columns:
    lines.append(f"Average quantity: {float(subset['quantity'].mean()):,.2f}")
  if "price" in subset.columns:
    lines.append(f"Average price: {format_currency(float(subset['price'].mean()))}")

  if column_name != "product" and "product" in subset.columns and "sales" in subset.columns:
    top_products = subset.groupby("product", dropna=False)["sales"].sum().sort_values(ascending=False).head(5).reset_index()
    lines.append("Top products:")
    lines.append(top_products.to_string(index=False))

  if column_name != "region" and "region" in subset.columns and "sales" in subset.columns:
    top_regions = subset.groupby("region", dropna=False)["sales"].sum().sort_values(ascending=False).head(5).reset_index()
    lines.append("Top regions:")
    lines.append(top_regions.to_string(index=False))

  return "\n".join(lines)


def emit_state_progress(payload: dict[str, Any]) -> None:
    progress = float(payload.get("progress", 0))
    message = str(payload.get("message", ""))
    stage = str(payload.get("stage", ""))
    set_state(status="running", progress=progress, stage=stage, message=message)
    append_log(f"{int(round(progress)):>3}% - {message}")


def safe_filename(filename: str) -> str:
    cleaned = secure_filename(filename)
    return cleaned or f"upload_{int(time.time())}.bin"


def markdown_table(df: pd.DataFrame, max_rows: int = 8) -> str:
    if df is None or df.empty:
        return "_No data available._"
    preview = df.head(max_rows).copy()
    headers = [str(column) for column in preview.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in preview.itertuples(index=False):
        rows.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rows)


def stamp_matplotlib_signature(fig: plt.Figure) -> plt.Figure:
    fig.text(
        0.5,
        0.5,
        AUTHOR_SIGNATURE,
        ha="center",
        va="center",
        rotation=30,
        fontsize=24,
        color="gray",
        alpha=0.16,
        transform=fig.transFigure,
    )
    return fig


def normalize_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    else:
        frame["date"] = pd.Timestamp.today().normalize()

    for column in ["quantity", "price", "sales", "customer_purchase_count"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "sales" not in frame.columns and {"quantity", "price"}.issubset(frame.columns):
        frame["sales"] = frame["quantity"] * frame["price"]

    if "quantity" not in frame.columns:
        frame["quantity"] = 1.0
    if "price" not in frame.columns:
        frame["price"] = 1.0
    if "sales" not in frame.columns:
        frame["sales"] = frame["quantity"] * frame["price"]

    if "customer_purchase_count" not in frame.columns:
        if "customer_id" in frame.columns:
            frame["customer_purchase_count"] = frame.groupby("customer_id")["customer_id"].transform("size")
        else:
            frame["customer_purchase_count"] = 1

    if "category" not in frame.columns:
        if "product" in frame.columns:
            frame["category"] = frame["product"].astype(str)
        else:
            frame["category"] = "Other"
    if "region" not in frame.columns:
        frame["region"] = "Unknown"

    frame["category"] = frame["category"].astype("string").fillna("Other").str.strip()
    frame["region"] = frame["region"].astype("string").fillna("Unknown").str.strip()

    if frame["date"].isna().any():
        frame["date"] = frame["date"].fillna(pd.Timestamp.today().normalize())

    frame["month"] = frame["date"].dt.month.astype(int)
    frame["day_of_week"] = frame["date"].dt.dayofweek.astype(int)
    frame["hour"] = frame["date"].dt.hour.astype(int)
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)

    for column in ["quantity", "price", "sales", "customer_purchase_count"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            fill_value = frame[column].median()
            if pd.isna(fill_value):
                fill_value = 0
            frame[column] = frame[column].fillna(fill_value)

    frame["quantity"] = frame["quantity"].clip(lower=1)
    frame["price"] = frame["price"].clip(lower=0.01)
    frame["sales"] = frame["sales"].clip(lower=0.01)
    return frame


def train_sales_model(cleaned_df: pd.DataFrame, progress_callback: Any | None = None) -> dict[str, Any]:
    emit_state_progress({"stage": "model_prep", "progress": 88, "message": "Preparing sales model features..."}) if progress_callback is None else progress_callback({"stage": "model_prep", "progress": 88, "message": "Preparing sales model features..."})
    frame = normalize_model_frame(cleaned_df)

    feature_columns = [
        column
        for column in [
            "quantity",
            "price",
            "category",
            "region",
            "month",
            "day_of_week",
            "hour",
            "is_weekend",
            "customer_purchase_count",
        ]
        if column in frame.columns
    ]

    if "sales" not in frame.columns:
        frame["sales"] = frame["quantity"] * frame["price"]

    model_df = frame[feature_columns + ["sales"]].dropna().copy()
    if model_df.empty or len(model_df) < 20:
        raise RuntimeError("Not enough clean rows to train the sales model.")

    numeric_features = [column for column in feature_columns if column not in {"category", "region"}]
    categorical_features = [column for column in ["category", "region"] if column in feature_columns]

    X = model_df[feature_columns]
    y = model_df["sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn fallback
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encoder)]), categorical_features),
        ],
        remainder="drop",
    )

    candidates = {
        "Baseline Median": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=220, random_state=42, n_jobs=-1),
    }

    results: list[dict[str, Any]] = []
    pipelines: dict[str, Pipeline] = {}
    for index, (model_name, estimator) in enumerate(candidates.items(), start=1):
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        results.append(
            {
                "model": model_name,
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": rmse,
                "r2": float(r2_score(y_test, predictions)),
            }
        )
        pipelines[model_name] = pipeline
        current_progress = 90 + index * 2
        emit = progress_callback if progress_callback is not None else emit_state_progress
        emit({"stage": f"model_{index}", "progress": current_progress, "message": f"Evaluated {model_name}."})

    results_df = pd.DataFrame(results).sort_values(by="rmse")
    best_model_name = str(results_df.iloc[0]["model"])
    best_pipeline = pipelines[best_model_name]
    best_predictions = best_pipeline.predict(X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    model_metrics = results_df.to_dict(orient="records")
    MODEL_METRICS_PATH.write_text(json.dumps(model_metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 8))
    color_values = np.linspace(0, 1, len(y_test)) if len(y_test) else [0]
    scatter = ax.scatter(y_test, best_predictions, c=color_values, cmap="turbo", alpha=0.82, edgecolor="white", linewidth=0.5)
    limits = [min(float(y_test.min()), float(best_predictions.min())), max(float(y_test.max()), float(best_predictions.max()))]
    ax.plot(limits, limits, linestyle="--", color="#fb7185", linewidth=2)
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title(f"Predicted vs Actual Sales - {best_model_name}")
    fig.colorbar(scatter, ax=ax, label="Sample ordering")
    stamp_matplotlib_signature(fig)
    fig.tight_layout()
    fig.savefig(PREDICTION_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    product_to_category = {}
    product_to_price = {}
    product_to_quantity = {}
    if "product" in frame.columns:
        if "category" in frame.columns:
            product_to_category = frame.groupby("product")["category"].agg(lambda series: series.mode(dropna=True).iloc[0] if not series.mode(dropna=True).empty else "Other").to_dict()
        if "price" in frame.columns:
            product_to_price = frame.groupby("product")["price"].median().to_dict()
        if "quantity" in frame.columns:
            product_to_quantity = frame.groupby("product")["quantity"].median().to_dict()

    top_products = pd.DataFrame()
    if "product" in frame.columns:
        top_products = frame.groupby("product", dropna=False)["sales"].sum().sort_values(ascending=False).head(10).reset_index()
    top_regions = pd.DataFrame()
    if "region" in frame.columns:
        top_regions = frame.groupby("region", dropna=False)["sales"].sum().sort_values(ascending=False).head(10).reset_index()

    monthly_sales = pd.DataFrame()
    if "date" in frame.columns:
        monthly_sales = (
            frame.assign(month_period=frame["date"].dt.to_period("M"))
            .groupby("month_period")["sales"]
            .sum()
            .reset_index()
        )

    context = {
        "cleaned_df": frame,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "results_df": results_df,
        "best_metrics": results_df.iloc[0].to_dict(),
        "model_path": MODEL_PATH,
        "metrics_path": MODEL_METRICS_PATH,
        "prediction_plot": PREDICTION_PLOT_PATH,
        "known_products": sorted(frame["product"].dropna().astype(str).unique().tolist()) if "product" in frame.columns else [],
        "known_categories": sorted(frame["category"].dropna().astype(str).unique().tolist()) if "category" in frame.columns else [],
        "known_regions": sorted(frame["region"].dropna().astype(str).unique().tolist()) if "region" in frame.columns else [],
        "product_to_category": product_to_category,
        "product_to_price": product_to_price,
        "product_to_quantity": product_to_quantity,
        "top_products": top_products,
        "top_regions": top_regions,
        "monthly_sales": monthly_sales,
    }
    return context


def load_existing_context() -> None:
    if not CLEANED_CSV.exists() or not MODEL_PATH.exists():
        return
    try:
        cleaned_df = pd.read_csv(CLEANED_CSV)
        if "date" in cleaned_df.columns:
            cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce")
        pipeline = joblib.load(MODEL_PATH)
        metrics = []
        if MODEL_METRICS_PATH.exists():
            try:
                metrics = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
            except Exception:
                metrics = []
        context = {
            "cleaned_df": cleaned_df,
            "best_pipeline": pipeline,
            "best_model_name": metrics[0]["model"] if metrics else "Loaded model",
            "results_df": pd.DataFrame(metrics) if metrics else pd.DataFrame(),
            "best_metrics": metrics[0] if metrics else {},
            "feature_columns": [
                column
                for column in [
                    "quantity",
                    "price",
                    "category",
                    "region",
                    "month",
                    "day_of_week",
                    "hour",
                    "is_weekend",
                    "customer_purchase_count",
                ]
                if column in cleaned_df.columns or column in ["quantity", "price"]
            ],
            "numeric_features": ["quantity", "price", "month", "day_of_week", "hour", "is_weekend", "customer_purchase_count"],
            "categorical_features": ["category", "region"],
            "known_products": sorted(cleaned_df["product"].dropna().astype(str).unique().tolist()) if "product" in cleaned_df.columns else [],
            "known_categories": sorted(cleaned_df["category"].dropna().astype(str).unique().tolist()) if "category" in cleaned_df.columns else [],
            "known_regions": sorted(cleaned_df["region"].dropna().astype(str).unique().tolist()) if "region" in cleaned_df.columns else [],
            "product_to_category": cleaned_df.groupby("product")["category"].agg(lambda series: series.mode(dropna=True).iloc[0] if not series.mode(dropna=True).empty else "Other").to_dict() if {"product", "category"}.issubset(cleaned_df.columns) else {},
            "product_to_price": cleaned_df.groupby("product")["price"].median().to_dict() if {"product", "price"}.issubset(cleaned_df.columns) else {},
            "product_to_quantity": cleaned_df.groupby("product")["quantity"].median().to_dict() if {"product", "quantity"}.issubset(cleaned_df.columns) else {},
            "top_products": cleaned_df.groupby("product", dropna=False)["sales"].sum().sort_values(ascending=False).head(10).reset_index() if {"product", "sales"}.issubset(cleaned_df.columns) else pd.DataFrame(),
            "top_regions": cleaned_df.groupby("region", dropna=False)["sales"].sum().sort_values(ascending=False).head(10).reset_index() if {"region", "sales"}.issubset(cleaned_df.columns) else pd.DataFrame(),
            "monthly_sales": cleaned_df.assign(month_period=cleaned_df["date"].dt.to_period("M")).groupby("month_period")["sales"].sum().reset_index() if "date" in cleaned_df.columns and "sales" in cleaned_df.columns else pd.DataFrame(),
        }
        MODEL_CONTEXT.clear()
        MODEL_CONTEXT.update(context)
        set_state(
            status="idle",
            progress=100,
            stage="ready",
            message="Loaded existing cleaned dataset and model artifact. Ready for chat or a fresh run.",
            model_ready=True,
            data_ready=True,
            analysis={
                "rows_before": int(len(cleaned_df)),
                "rows_after": int(len(cleaned_df)),
                "columns_after": int(cleaned_df.shape[1]),
                "source_label": "Existing outputs",
            },
            model={
                "best_model": context.get("best_model_name", "Loaded model"),
                "best_mae": context.get("best_metrics", {}).get("mae"),
                "best_rmse": context.get("best_metrics", {}).get("rmse"),
                "best_r2": context.get("best_metrics", {}).get("r2"),
            },
            artifacts={
                "report": "/download/report",
                "cleaned_csv": "/download/cleaned",
                "model": "/download/model",
                "prediction_plot": "/download/prediction-plot",
            },
        )
    except Exception as exc:
        set_state(status="idle", progress=0, stage="bootstrap_failed", message=f"Existing model could not be loaded: {exc}")


def build_prediction_frame(message: str) -> tuple[pd.DataFrame, list[str], int]:
  if not MODEL_CONTEXT.get("feature_columns"):
    raise RuntimeError("Model context is not ready yet.")

  cleaned_df = MODEL_CONTEXT.get("cleaned_df", pd.DataFrame())
  frame = normalize_model_frame(cleaned_df)
  feature_defaults: dict[str, Any] = {}
  signal_count = 0

  for column in MODEL_CONTEXT["feature_columns"]:
    if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
      feature_defaults[column] = float(frame[column].median())
    elif column in frame.columns:
      mode_values = frame[column].mode(dropna=True)
      feature_defaults[column] = str(mode_values.iloc[0]) if not mode_values.empty else "Unknown"
    else:
      feature_defaults[column] = 0

  if "month" in feature_defaults and "date" in frame.columns:
    feature_defaults["month"] = int(frame["date"].dt.month.median()) if not frame["date"].dropna().empty else 1
  if "day_of_week" in feature_defaults and "date" in frame.columns:
    feature_defaults["day_of_week"] = int(frame["date"].dt.dayofweek.median()) if not frame["date"].dropna().empty else 0
  if "hour" in feature_defaults:
    feature_defaults["hour"] = 0
  if "is_weekend" in feature_defaults and "date" in frame.columns:
    feature_defaults["is_weekend"] = int(frame["date"].dt.dayofweek.median() >= 5) if not frame["date"].dropna().empty else 0
  if "customer_purchase_count" in feature_defaults:
    feature_defaults["customer_purchase_count"] = float(frame["customer_purchase_count"].median()) if "customer_purchase_count" in frame.columns else 1

  text = message.strip()
  notes: list[str] = []
  known_products = MODEL_CONTEXT.get("known_products", [])
  known_categories = MODEL_CONTEXT.get("known_categories", [])
  known_regions = MODEL_CONTEXT.get("known_regions", [])

  def match_choice(choices: list[str]) -> str | None:
    lowered = text.lower()
    matches = [choice for choice in choices if choice.lower() in lowered]
    return max(matches, key=len) if matches else None

  def extract_number(labels: list[str], as_int: bool = False):
    for label in labels:
      pattern = rf"(?:{re.escape(label).replace('\\ ', '\\s+')})\s*[:=]?\s*\$?(-?\d+(?:\.\d+)?)"
      match = re.search(pattern, text, flags=re.IGNORECASE)
      if match:
        value = float(match.group(1))
        return int(round(value)) if as_int else value
    return None

  explicit_product = match_choice(known_products)
  explicit_category = match_choice(known_categories)
  explicit_region = match_choice(known_regions)

  if explicit_product:
    notes.append(f"Detected product '{explicit_product}'.")
    signal_count += 1
    if "category" in feature_defaults:
      feature_defaults["category"] = MODEL_CONTEXT.get("product_to_category", {}).get(explicit_product, feature_defaults.get("category", "Unknown"))
    if explicit_product in MODEL_CONTEXT.get("product_to_price", {}):
      feature_defaults["price"] = float(MODEL_CONTEXT["product_to_price"][explicit_product])
      signal_count += 1
    if explicit_product in MODEL_CONTEXT.get("product_to_quantity", {}):
      feature_defaults["quantity"] = float(MODEL_CONTEXT["product_to_quantity"][explicit_product])
      signal_count += 1

  if explicit_category and "category" in feature_defaults:
    feature_defaults["category"] = explicit_category
    signal_count += 1
  if explicit_region and "region" in feature_defaults:
    feature_defaults["region"] = explicit_region
    signal_count += 1

  quantity_value = extract_number(["quantity", "qty", "units", "items"])
  if quantity_value is not None and "quantity" in feature_defaults:
    feature_defaults["quantity"] = float(quantity_value)
    signal_count += 1

  price_value = extract_number(["price", "unit price", "selling price", "sale price"])
  if price_value is not None and "price" in feature_defaults:
    feature_defaults["price"] = float(price_value)
    signal_count += 1

  customer_count_value = extract_number(["customer_purchase_count", "customer purchase count", "purchase count"], as_int=True)
  if customer_count_value is not None and "customer_purchase_count" in feature_defaults:
    feature_defaults["customer_purchase_count"] = int(customer_count_value)
    signal_count += 1

  hour_value = extract_number(["hour"], as_int=True)
  if hour_value is not None and "hour" in feature_defaults:
    feature_defaults["hour"] = int(hour_value)
    signal_count += 1

  date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
  if date_match and "date" in frame.columns:
    parsed_date = pd.to_datetime(date_match.group(1), errors="coerce")
    if pd.notna(parsed_date):
      feature_defaults["month"] = int(parsed_date.month)
      feature_defaults["day_of_week"] = int(parsed_date.dayofweek)
      feature_defaults["hour"] = int(parsed_date.hour)
      feature_defaults["is_weekend"] = int(parsed_date.dayofweek >= 5)
      notes.append(f"Used date {parsed_date.date()} to derive time features.")
      signal_count += 1
  elif "date" in text.lower():
    notes.append("No parseable YYYY-MM-DD date was found, so default time features were used.")

  prediction_frame = pd.DataFrame([feature_defaults])[MODEL_CONTEXT["feature_columns"]]
  return prediction_frame, notes, signal_count


def answer_chat(message: str) -> str:
  normalized = message.strip().lower()
  if not normalized:
    return "Type a question or a prediction request."

  cleaned_df = MODEL_CONTEXT.get("cleaned_df", pd.DataFrame())
  known_products = MODEL_CONTEXT.get("known_products", [])
  known_categories = MODEL_CONTEXT.get("known_categories", [])
  known_regions = MODEL_CONTEXT.get("known_regions", [])

  def match_choice(choices: list[str]) -> str | None:
    matches = [choice for choice in choices if choice.lower() in normalized]
    return max(matches, key=len) if matches else None

  explicit_product = match_choice(known_products)
  explicit_category = match_choice(known_categories)
  explicit_region = match_choice(known_regions)

  if any(keyword in normalized for keyword in ["help", "what can you do", "examples"]):
    return (
      "Ask me for a prediction using quantity, price, category, region, product, and date.\n"
      "Examples:\n"
      "- Predict sales for quantity 3 price 120 category Electronics region West date 2025-04-01\n"
      "- Tell me about Electronics in West region\n"
      "- Show model metrics\n"
      "- What are the top products?\n"
      "- What are the top regions?"
    )

  if any(keyword in normalized for keyword in ["summary", "overview", "dataset", "data"]):
    return build_dataset_overview(cleaned_df)

  if any(keyword in normalized for keyword in ["metric", "metrics", "best model", "r2", "rmse", "mae"]):
    results_df = MODEL_CONTEXT.get("results_df", pd.DataFrame())
    if results_df.empty:
      metrics = MODEL_CONTEXT.get("best_metrics", {})
      return "\n".join([f"{key}: {value}" for key, value in metrics.items()]) or "Model metrics are not available yet."
    display_df = results_df.copy()
    for column in ["mae", "rmse", "r2"]:
      if column in display_df.columns:
        display_df[column] = display_df[column].map(lambda value: f"{value:,.2f}")
    best = MODEL_CONTEXT.get("best_model_name", "Loaded model")
    return f"Best model: {best}\n\n{display_df.to_string(index=False)}"

  if explicit_product:
    return build_segment_summary(cleaned_df, "product", explicit_product)
  if explicit_category:
    return build_segment_summary(cleaned_df, "category", explicit_category)
  if explicit_region:
    return build_segment_summary(cleaned_df, "region", explicit_region)

  if any(keyword in normalized for keyword in ["top product", "top products", "best product"]):
    table = MODEL_CONTEXT.get("top_products", pd.DataFrame())
    if table.empty:
      return "Top products are not available yet."
    return f"Top products\n\n{table.to_string(index=False)}"

  if any(keyword in normalized for keyword in ["top region", "best region", "regions"]):
    table = MODEL_CONTEXT.get("top_regions", pd.DataFrame())
    if table.empty:
      return "Top regions are not available yet."
    return f"Top regions\n\n{table.to_string(index=False)}"

  if any(keyword in normalized for keyword in ["season", "trend", "monthly", "sales over time"]):
    table = MODEL_CONTEXT.get("monthly_sales", pd.DataFrame())
    if table.empty:
      return "Seasonality data is not available yet."
    display_df = table.copy()
    if "month_period" in display_df.columns:
      display_df["month_period"] = display_df["month_period"].astype(str)
    return f"Seasonality snapshot\n\n{display_df.head(12).to_string(index=False)}"

  wants_prediction = any(keyword in normalized for keyword in ["predict", "forecast", "estimate", "sales", "revenue"]) or any(signal in normalized for signal in ["quantity", "price", "region", "category", "product", "date"])
  if wants_prediction:
    if MODEL_CONTEXT.get("best_pipeline") is None:
      return "The data is ready, but the model is not trained yet. Run the training step first, then ask for a prediction with quantity, price, product, category, region, or date."

    prediction_frame, notes, signal_count = build_prediction_frame(message)
    if signal_count == 0:
      return (
        "I can make a prediction, but I need at least one input such as quantity, price, product, category, region, or date.\n"
        "Example: Predict sales for quantity 3 price 120 category Electronics region West date 2025-04-01"
      )

    predicted_sales = float(MODEL_CONTEXT["best_pipeline"].predict(prediction_frame)[0])
    note_lines = "\n".join(f"- {note}" for note in notes) if notes else "- Used dataset defaults for any missing inputs."
    return (
      f"Predicted sales: {format_currency(predicted_sales)}\n\n"
      f"Model: {MODEL_CONTEXT.get('best_model_name', 'Sales model')}\n\n"
      f"Inputs used:\n{prediction_frame.iloc[0].to_string()}\n\n"
      f"Notes:\n{note_lines}\n\n"
      "If you want a different scenario, change quantity, price, product, category, region, or date."
    )

  if cleaned_df.empty:
    return "Load or analyze a dataset first, then I can answer questions about sales, products, regions, and trends."

  hints: list[str] = []
  if any(word in normalized for word in ["sales", "revenue"]):
    hints.append("sales")
  if any(word in normalized for word in ["product", "category"]):
    hints.append("product/category")
  if any(word in normalized for word in ["region"]):
    hints.append("region")
  if any(word in normalized for word in ["trend", "season", "monthly", "date"]):
    hints.append("seasonality")
  if hints:
    return f"I saw {', '.join(hints)} in your message. I can answer with a summary, top lists, or a prediction if you add quantity, price, product, category, region, or date."

  return "Try asking about model metrics, top products, top regions, seasonality, dataset overview, or a sales prediction."


def build_multicolor_plot_html(cleaned_df: pd.DataFrame) -> str:
    if cleaned_df.empty:
        return "<div style='padding:18px;'>No data is available yet.</div>"
    x_axis = "quantity" if "quantity" in cleaned_df.columns else cleaned_df.select_dtypes(include=[np.number]).columns[0]
    y_axis = "sales" if "sales" in cleaned_df.columns else cleaned_df.select_dtypes(include=[np.number]).columns[-1]
    color_axis = "category" if "category" in cleaned_df.columns else ("region" if "region" in cleaned_df.columns else None)
    size_axis = "price" if "price" in cleaned_df.columns else None
    fig = px.scatter(
        cleaned_df,
        x=x_axis,
        y=y_axis,
        color=color_axis,
        size=size_axis,
        hover_data=[column for column in ["product", "region", "customer_id"] if column in cleaned_df.columns],
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_white",
        title="Multicolor Retail Explorer",
    )
    fig.update_traces(marker=dict(opacity=0.84, line=dict(width=0.4, color="white")))
    fig.add_annotation(
        text=AUTHOR_SIGNATURE,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        opacity=0.16,
        font=dict(size=18, color="gray"),
    )
    return fig.to_html(full_html=True, include_plotlyjs="inline")


def build_3d_plot_html(cleaned_df: pd.DataFrame, x_axis: str, y_axis: str, z_axis: str, color_axis: str) -> str:
    if cleaned_df.empty:
        return "<div style='padding:18px;'>No data is available yet.</div>"
    plot_df = cleaned_df.copy()
    for axis in [x_axis, y_axis, z_axis]:
        if axis in plot_df.columns:
            plot_df[axis] = pd.to_numeric(plot_df[axis], errors="coerce")
    plot_df = plot_df.dropna(subset=[axis for axis in [x_axis, y_axis, z_axis] if axis in plot_df.columns])
    if plot_df.empty:
        return "<div style='padding:18px;'>Not enough valid numeric data is available for the 3D view yet.</div>"

    color_value = None if color_axis == "None" or color_axis not in plot_df.columns else color_axis
    if color_value is not None:
        plot_df[color_value] = plot_df[color_value].astype("string").fillna("Unknown").replace({"<NA>": "Unknown", "nan": "Unknown"})

    try:
        fig = px.scatter_3d(
            plot_df,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color=color_value,
            hover_data=[column for column in ["product", "region", "category", "customer_id"] if column in plot_df.columns],
            color_discrete_sequence=px.colors.qualitative.Vivid,
            template="plotly_white",
            title=f"Interactive 3D Retail Explorer: {x_axis} vs {y_axis} vs {z_axis}",
        )
    except Exception:
        fig = px.scatter_3d(
            plot_df,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            hover_data=[column for column in ["product", "region", "category", "customer_id"] if column in plot_df.columns],
            color_discrete_sequence=px.colors.qualitative.Vivid,
            template="plotly_white",
            title=f"Interactive 3D Retail Explorer: {x_axis} vs {y_axis} vs {z_axis}",
        )
    fig.update_traces(marker=dict(size=5, opacity=0.82))
    fig.update_layout(scene=dict(bgcolor="rgb(250,250,250)"), margin=dict(l=0, r=0, t=60, b=0))
    fig.add_annotation(
        text=AUTHOR_SIGNATURE,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        opacity=0.16,
        font=dict(size=18, color="gray"),
    )
    return fig.to_html(full_html=True, include_plotlyjs="inline")


def current_cleaned_df() -> pd.DataFrame:
  if MODEL_CONTEXT.get("cleaned_df") is not None:
    return MODEL_CONTEXT["cleaned_df"]
  if not APP_STATE.get("data_ready"):
    return pd.DataFrame()
  if CLEANED_CSV.exists():
    cleaned_df = pd.read_csv(CLEANED_CSV)
    if "date" in cleaned_df.columns:
      cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce")
    return cleaned_df
  return pd.DataFrame()


def run_dashboard_job(dataset_path: Path | None, demo_rows: int, seed: int) -> None:
  global MODEL_CONTEXT
  try:
    set_state(
      status="running",
      progress=0,
      stage="starting",
      message="Starting the automatic workflow...",
      last_error=None,
      workflow_mode="automatic",
      workflow_step="running",
    )
    append_log("Job started.")
    if dataset_path is not None:
      emit_state_progress({"stage": "queued", "progress": 6, "message": f"Using uploaded file: {dataset_path.name}"})
    else:
      emit_state_progress({"stage": "demo", "progress": 6, "message": "Using synthetic demo data."})

    artifacts = run_phase1_analysis(
      input_path=dataset_path,
      output_dir=OUTPUT_DIR,
      demo_rows=demo_rows,
      seed=seed,
      progress_callback=emit_state_progress,
    )
    analysis_summary = dict(artifacts.summary)
    analysis_summary["source_label"] = artifacts.summary.get("source_label", "Unknown")
    analysis_summary["cleaned_csv"] = str(artifacts.cleaned_csv.relative_to(WORKSPACE_ROOT))
    analysis_summary["report_md"] = str(artifacts.report_md.relative_to(WORKSPACE_ROOT))
    set_state(status="running", progress=85, stage="modeling", message="Training the sales model and preparing chat features...")
    append_log("Phase 1 analysis completed. Training sales model...")

    cleaned_df = pd.read_csv(artifacts.cleaned_csv)
    if "date" in cleaned_df.columns:
      cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce")
    context = train_sales_model(cleaned_df, progress_callback=emit_state_progress)
    MODEL_CONTEXT.clear()
    MODEL_CONTEXT.update(context)

    results_df = context.get("results_df", pd.DataFrame())
    best_metrics = context.get("best_metrics", {})
    model_summary = {
      "best_model": context.get("best_model_name", "Sales model"),
      "best_mae": best_metrics.get("mae"),
      "best_rmse": best_metrics.get("rmse"),
      "best_r2": best_metrics.get("r2"),
    }
    context_artifacts = {
      "report": "/download/report",
      "cleaned_csv": "/download/cleaned",
      "model": "/download/model",
      "prediction_plot": "/download/prediction-plot",
      "multicolor_plot": "/plot/scatter",
    }

    set_state(
      status="done",
      progress=100,
      stage="complete",
      message="Analysis, model training, and visual explorer are ready.",
      workflow_mode="automatic",
      workflow_step="complete",
      model_ready=True,
      data_ready=True,
      analysis=analysis_summary,
      model=model_summary,
      artifacts=context_artifacts,
    )
    append_log("Pipeline completed successfully.")
  except Exception as exc:  # noqa: BLE001
    error_text = f"{type(exc).__name__}: {exc}"
    set_state(status="error", progress=100, stage="error", message=error_text, last_error=traceback.format_exc())
    append_log(error_text)


def run_manual_analysis_job(dataset_path: Path) -> None:
  global MODEL_CONTEXT
  try:
    set_state(
      status="running",
      progress=0,
      stage="manual_cleaning",
      message=f"Cleaning uploaded file {dataset_path.name}...",
      workflow_mode="manual",
      workflow_step="analysis_running",
      model_ready=False,
      data_ready=False,
      last_error=None,
    )
    append_log(f"Manual analysis started for {dataset_path.name}.")

    artifacts = run_phase1_analysis(
      input_path=dataset_path,
      output_dir=OUTPUT_DIR,
      demo_rows=1200,
      seed=42,
      progress_callback=emit_state_progress,
    )
    cleaned_df = pd.read_csv(artifacts.cleaned_csv)
    if "date" in cleaned_df.columns:
      cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce")

    MODEL_CONTEXT.clear()
    MODEL_CONTEXT.update(build_data_context(cleaned_df))

    analysis_summary = dict(artifacts.summary)
    analysis_summary["source_label"] = artifacts.summary.get("source_label", dataset_path.name)
    analysis_summary["cleaned_csv"] = str(artifacts.cleaned_csv.relative_to(WORKSPACE_ROOT))
    analysis_summary["report_md"] = str(artifacts.report_md.relative_to(WORKSPACE_ROOT))
    set_state(
      status="idle",
      progress=100,
      stage="analysis_ready",
      message="Cleaning and EDA are complete. Click Train Model to continue.",
      workflow_mode="manual",
      workflow_step="analysis_ready",
      model_ready=False,
      data_ready=True,
      analysis=analysis_summary,
      model={},
      artifacts={
        "report": "/download/report",
        "cleaned_csv": "/download/cleaned",
      },
    )
    append_log("Manual cleaning and EDA completed.")
  except Exception as exc:  # noqa: BLE001
    error_text = f"{type(exc).__name__}: {exc}"
    set_state(
      status="error",
      progress=100,
      stage="manual_error",
      message=error_text,
      last_error=traceback.format_exc(),
      workflow_mode="manual",
      workflow_step="error",
    )
    append_log(error_text)


def run_manual_training_job() -> None:
  global MODEL_CONTEXT
  try:
    cleaned_df = MODEL_CONTEXT.get("cleaned_df", pd.DataFrame())
    if cleaned_df.empty:
      raise RuntimeError("No cleaned dataset is available. Run the cleaning step first.")

    analysis_summary = snapshot_state().get("analysis", {})
    set_state(
      status="running",
      progress=85,
      stage="manual_training",
      message="Training the model on the uploaded dataset...",
      workflow_mode="manual",
      workflow_step="training_running",
      model_ready=False,
      data_ready=True,
    )
    append_log("Manual training started.")

    context = train_sales_model(cleaned_df, progress_callback=emit_state_progress)
    MODEL_CONTEXT.clear()
    MODEL_CONTEXT.update(context)

    best_metrics = context.get("best_metrics", {})
    model_summary = {
      "best_model": context.get("best_model_name", "Sales model"),
      "best_mae": best_metrics.get("mae"),
      "best_rmse": best_metrics.get("rmse"),
      "best_r2": best_metrics.get("r2"),
    }
    set_state(
      status="done",
      progress=100,
      stage="manual_complete",
      message="Manual workflow complete. Chat and downloads are ready.",
      workflow_mode="manual",
      workflow_step="trained",
      model_ready=True,
      data_ready=True,
      analysis=analysis_summary,
      model=model_summary,
      artifacts={
        "report": "/download/report",
        "cleaned_csv": "/download/cleaned",
        "model": "/download/model",
        "prediction_plot": "/download/prediction-plot",
        "multicolor_plot": "/plot/scatter",
      },
    )
    append_log("Manual workflow completed successfully.")
  except Exception as exc:  # noqa: BLE001
    error_text = f"{type(exc).__name__}: {exc}"
    set_state(
      status="error",
      progress=100,
      stage="manual_error",
      message=error_text,
      last_error=traceback.format_exc(),
      workflow_mode="manual",
      workflow_step="error",
    )
    append_log(error_text)


def job_is_running() -> bool:
  return CURRENT_JOB is not None and CURRENT_JOB.is_alive()


@app.route("/")
def index() -> Response:
  return Response(HTML_TEMPLATE.replace("__APP_TITLE__", APP_TITLE).replace("__AUTHOR__", AUTHOR_NAME), mimetype="text/html")


@app.route("/api/status")
def api_status() -> Response:
  state = snapshot_state()
  cleaned_df = current_cleaned_df()
  state["available_columns"] = cleaned_df.columns.tolist() if not cleaned_df.empty else []
  state["numeric_columns"] = cleaned_df.select_dtypes(include=[np.number]).columns.tolist() if not cleaned_df.empty else []
  state["categorical_columns"] = [column for column in cleaned_df.columns if column not in state.get("numeric_columns", []) and column != "date"] if not cleaned_df.empty else []
  state["has_manual_dataset"] = MANUAL_DATASET_PATH is not None and MANUAL_DATASET_PATH.exists()
  state["manual_dataset_path"] = str(MANUAL_DATASET_PATH) if MANUAL_DATASET_PATH is not None else None
  if cleaned_df.empty:
    state["axis_defaults"] = {}
  else:
    numeric_columns = state["numeric_columns"]
    categorical_columns = state["categorical_columns"]
    state["axis_defaults"] = {
      "x": "quantity" if "quantity" in numeric_columns else (numeric_columns[0] if numeric_columns else None),
      "y": "sales" if "sales" in numeric_columns else (numeric_columns[1] if len(numeric_columns) > 1 else None),
      "z": "price" if "price" in numeric_columns else (numeric_columns[2] if len(numeric_columns) > 2 else None),
      "color": "category" if "category" in categorical_columns else (categorical_columns[0] if categorical_columns else "None"),
    }
  return jsonify(snapshot_state())


@app.route("/api/start", methods=["POST"])
def api_start() -> Response:
  global CURRENT_JOB
  if job_is_running():
    return jsonify({"error": "A job is already running.", **snapshot_state()}), 409

  uploaded_file = request.files.get("dataset_file")
  demo_rows = int(request.form.get("demo_rows", 1200))
  seed = int(request.form.get("seed", 42))
  use_demo = request.form.get("use_demo", "0") == "1"
  dataset_path: Path | None = None

  reset_dashboard_state("automatic", "Automatic workflow selected. Run Full Flow to start.")

  if uploaded_file and uploaded_file.filename:
    file_path = UPLOAD_DIR / safe_filename(uploaded_file.filename)
    uploaded_file.save(file_path)
    dataset_path = file_path
  elif not use_demo and CLEANED_CSV.exists():
    dataset_path = CLEANED_CSV

  CURRENT_JOB = threading.Thread(target=run_dashboard_job, args=(dataset_path, demo_rows, seed), daemon=True)
  CURRENT_JOB.start()
  set_state(status="running", progress=2, stage="queued", message="Starting the pipeline...")
  append_log("Pipeline queued.")
  return jsonify(snapshot_state())


@app.route("/api/mode", methods=["POST"])
def api_mode() -> Response:
  payload = request.get_json(silent=True) or {}
  mode = str(payload.get("mode", "automatic")).strip().lower()
  if mode not in {"automatic", "manual"}:
    return jsonify({"error": "Mode must be automatic or manual."}), 400

  if job_is_running():
    return jsonify({"error": "Wait for the current job to finish before switching modes.", **snapshot_state()}), 409

  if mode == "automatic":
    reset_dashboard_state("automatic", "Automatic mode selected. Upload a file or use demo data, then click Run Full Flow.")
  else:
    reset_dashboard_state("manual", "Manual mode selected. Upload a dataset, then clean and train step by step.")
    set_state(workflow_mode="manual", workflow_step="waiting_upload")

  return jsonify(snapshot_state())


@app.route("/api/manual/upload", methods=["POST"])
def api_manual_upload() -> Response:
  global MANUAL_DATASET_PATH, MANUAL_DATASET_NAME
  if job_is_running():
    return jsonify({"error": "Wait for the current job to finish before uploading a new file.", **snapshot_state()}), 409

  uploaded_file = request.files.get("dataset_file")
  if not uploaded_file or not uploaded_file.filename:
    return jsonify({"error": "Please choose a dataset file first."}), 400

  reset_dashboard_state("manual", "Dataset uploaded. Click Clean & EDA next.")
  file_path = UPLOAD_DIR / safe_filename(uploaded_file.filename)
  uploaded_file.save(file_path)
  MANUAL_DATASET_PATH = file_path
  MANUAL_DATASET_NAME = file_path.name
  set_state(
    workflow_mode="manual",
    workflow_step="uploaded",
    manual_dataset_name=file_path.name,
    message=f"Uploaded {file_path.name}. Click Clean & EDA to continue.",
    status="idle",
    progress=0,
    stage="manual_uploaded",
    data_ready=False,
    model_ready=False,
    analysis={},
    model={},
    artifacts={},
  )
  append_log(f"Uploaded {file_path.name}.")
  return jsonify(snapshot_state())


@app.route("/api/manual/analyze", methods=["POST"])
def api_manual_analyze() -> Response:
  if job_is_running():
    return jsonify({"error": "A job is already running.", **snapshot_state()}), 409
  if MANUAL_DATASET_PATH is None or not MANUAL_DATASET_PATH.exists():
    return jsonify({"error": "Upload a dataset before running the analysis step."}), 400

  global CURRENT_JOB
  CURRENT_JOB = threading.Thread(target=run_manual_analysis_job, args=(MANUAL_DATASET_PATH,), daemon=True)
  CURRENT_JOB.start()
  set_state(status="running", progress=2, stage="manual_queue", message="Running Clean & EDA...", workflow_mode="manual", workflow_step="analysis_running")
  append_log("Manual analysis queued.")
  return jsonify(snapshot_state())


@app.route("/api/manual/train", methods=["POST"])
def api_manual_train() -> Response:
  if job_is_running():
    return jsonify({"error": "A job is already running.", **snapshot_state()}), 409
  if not APP_STATE.get("data_ready") or MODEL_CONTEXT.get("cleaned_df") is None:
    return jsonify({"error": "Run the cleaning step first before training the model."}), 400

  global CURRENT_JOB
  CURRENT_JOB = threading.Thread(target=run_manual_training_job, daemon=True)
  CURRENT_JOB.start()
  set_state(status="running", progress=85, stage="manual_queue", message="Starting model training...", workflow_mode="manual", workflow_step="training_running")
  append_log("Manual training queued.")
  return jsonify(snapshot_state())


@app.route("/api/chat", methods=["POST"])
def api_chat() -> Response:
  payload = request.get_json(silent=True) or {}
  message = str(payload.get("message", "")).strip()
  if not message:
    return jsonify({"error": "Please type a message."}), 400
  response = answer_chat(message)
  return jsonify({"role": AUTHOR_NAME, "response": response})


@app.route("/plot/scatter")
def plot_scatter_route() -> Response:
  cleaned_df = current_cleaned_df()
  html = build_multicolor_plot_html(cleaned_df)
  return Response(html, mimetype="text/html")


@app.route("/plot/3d")
def plot_3d_route() -> Response:
  cleaned_df = current_cleaned_df()
  if cleaned_df.empty:
    return Response("<div style='padding:18px;'>Run the analysis first to generate data for the 3D plot.</div>", mimetype="text/html")
  numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
  categorical_columns = [column for column in cleaned_df.columns if column not in numeric_columns and column != "date"]
  x_axis = request.args.get("x") or ("quantity" if "quantity" in numeric_columns else numeric_columns[0])
  y_axis = request.args.get("y") or ("sales" if "sales" in numeric_columns else (numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0]))
  z_axis = request.args.get("z") or ("price" if "price" in numeric_columns else (numeric_columns[2] if len(numeric_columns) > 2 else numeric_columns[0]))
  color_axis = request.args.get("color") or ("category" if "category" in categorical_columns else (categorical_columns[0] if categorical_columns else "None"))
  html = build_3d_plot_html(cleaned_df, x_axis, y_axis, z_axis, color_axis)
  return Response(html, mimetype="text/html")


@app.route("/download/<artifact>")
def download_artifact(artifact: str) -> Response:
  mapping = {
    "cleaned": CLEANED_CSV,
    "report": REPORT_MD,
    "model": MODEL_PATH,
    "prediction-plot": PREDICTION_PLOT_PATH,
  }
  file_path = mapping.get(artifact)
  if file_path is None or not file_path.exists():
    return jsonify({"error": "Artifact not available."}), 404
  return send_file(file_path, as_attachment=True)


def open_browser_after_delay(url: str = "http://127.0.0.1:7860") -> None:
  def _open() -> None:
    time.sleep(1.5)
    webbrowser.open(url)

  threading.Thread(target=_open, daemon=True).start()


def main() -> None:
  load_existing_context()
  open_browser_after_delay()
  app.run(host="127.0.0.1", port=7860, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
  main()
