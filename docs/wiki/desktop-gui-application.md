---
title: Desktop GUI Application
type: architecture-guide
status: active
last_updated: 2026-06-13
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
â”œâ”€â”€ main.py            # Entry point. Launches QAplication & Main Window.
â”œâ”€â”€ styles/            # Modularity for Qt Style Sheets (QSS)
â”‚   â”œâ”€â”€ base.qss       # Palette, main container layout
â”‚   â”œâ”€â”€ widgets.qss    # Tables, forms, buttons styling
â”‚   â”œâ”€â”€ custom.qss     # Card frames, logging viewer dark style
â”‚   â””â”€â”€ manager.py     # Combines all QSS sheets dynamically
â”œâ”€â”€ core/              # Separation of concern for business logic
â”‚   â”œâ”€â”€ db_helper.py   # SQLite registry fetch methods
â”‚   â””â”€â”€ process_runner.py # QProcess wrapper (non-blocking subprocess)
â”œâ”€â”€ widgets/           # Reusable graphical components
â”‚   â”œâ”€â”€ cards.py       # Render metrics/KPIs/direction warnings
â”‚   â””â”€â”€ log_viewer.py  # Autoscrolling plaintext log terminal
â””â”€â”€ tabs/              # High-level UI views (Tabs)
    â”œâ”€â”€ dashboard_tab.py # Shows best models & leaderboard table
    â”œâ”€â”€ training_tab.py  # Training triggers, preset selections, live execution logs
    â””â”€â”€ forecast_tab.py  # Visualizes forward forecasts, intervals & XAI metrics
```

---

## Core Technologies

1. **PySide6 (Qt for Python):** Used as the core framework for windows, layouts, tables, and signal-slot mechanisms.
2. **QProcess Subprocessing (YaklaÅŸÄ±m B):**
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
