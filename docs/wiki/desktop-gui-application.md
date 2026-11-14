---
title: Desktop GUI Application
type: architecture-guide
status: active
last_updated: 2026-06-14
owner: llm
---

# Desktop GUI Application

The PySide6 Desktop Application provides a corporate-grade, visual control panel to manage, train, and diagnose forecast models within `ts_forecasting_lab`.

It serves as the main decision-support client directly interacting with the local SQLite model registry and running pipeline processes in background threads.

---

## Architecture Design

To respect SOLID principles and prevent the **God Object** antipattern, the desktop module is split into clean layers:

```text
src/gui/
|-- main.py              # Entry point. Launches QApplication and Main Window.
|-- styles/              # Modularity for Qt Style Sheets (QSS)
|   |-- base.qss         # Palette, main container layout
|   |-- widgets.qss      # Tables, forms, buttons styling
|   |-- custom.qss       # Card frames, message boxes, empty states, logging viewer
|   `-- manager.py       # Combines all QSS sheets dynamically
|-- core/                # Separation of concern for business logic
|   |-- db_helper.py     # SQLite registry fetch methods
|   `-- process_runner.py # QProcess wrapper (non-blocking subprocess)
|-- widgets/             # Reusable graphical components
|   |-- cards.py         # Render metrics/KPIs/direction warnings
|   `-- log_viewer.py    # Autoscrolling plaintext log terminal
`-- tabs/                # High-level UI views (Tabs)
    |-- dashboard_tab.py # Shows best models and leaderboard table
    |-- training_tab.py  # Training triggers, preset selections, live execution logs
    `-- forecast_tab.py  # Visualizes forward forecasts, intervals and XAI metrics
```

---

## Core Technologies

1. **PySide6 (Qt for Python):** Used as the core framework for windows, layouts, tables, and signal-slot mechanisms.
2. **QProcess Subprocessing (Yaklaşım B):**
   - Training (`batch.py`) and forecasting (`forecast.py`) run as decoupled OS processes.
   - This isolates heavy ML resources (PyTorch, TensorFlow, LightGBM) from the GUI thread.
   - Standard output (`stdout`) and error (`stderr`) streams are dynamically parsed and forwarded to the `LogViewer` in real-time.
3. **Conda Environment Execution:**
   - The app auto-detects and triggers subprocesses using the specific interpreter path: `C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe`.
   - Fallback defaults to `sys.executable` if the environment is not found.

---

## Styling & Theme Rules

The UI incorporates a dark corporate-grade styling:
- **Base Background:** Dark Slate (`#0f172a`)
- **Card Background:** Slate Gray (`#1e293b`)
- **Accents/Controls:** Deep Corporate Blue (`#3b82f6` with `#2563eb` hover states)
- **Positive/Negative Indicators:** Green (`#22c55e`), Red (`#ef4444`), Yellow (`#eab308`)

Styles are loaded on app startup via the custom style manager:
```python
from src.gui.styles import get_combined_stylesheet
app.setStyleSheet(get_combined_stylesheet())
```

---

## UI/UX Integrity Rules

- Visible Turkish UI strings are stored as valid UTF-8 and covered by the GUI smoke test to prevent mojibake regressions (for example `U+00C3`, `U+00C4`, `U+00C5`, `U+00E2`, `U+00C2`, `U+FFFD` markers).
- Dashboard `dir_acc` and `hit_rate` values are displayed as already-percent values from `best_models`; the GUI must not multiply them by 100 again.
- Forecast empty states are shown inline in the forecast table area instead of relying only on a modal dialog.
- Training and forecast control group boxes stay compact with fixed vertical sizing so tables, cards, and logs remain visible in the first viewport.
- `QMessageBox` is styled through QSS so dialog text remains readable under the dark application theme.
- `ProcessRunner` forces subprocess Python output to UTF-8 and falls back to CP1254 decoding for legacy Windows CLI output; remaining stdout/stderr is drained on process finish so failure details are not lost.
- Forecast idle/error empty states mention the selected symbol or point users to the visible production log instead of showing a generic "select stock" message after a symbol is already selected.

---

## Commands

### Run GUI Application
To start the application in the specified conda environment:
```bash
python -m src.gui.main
```

### Run GUI Automated Tests
To run the PySide6 smoke tests using pytest:
```bash
python -m pytest tests/test_gui_smoke.py -v
```
