# -*- coding: utf-8 -*-
"""tests.test_gui_smoke - PySide6 masaustu GUI duman testleri (offscreen)."""

import os
import sys

import pandas as pd
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.gui.core import PeerReader  # noqa: E402
from src.gui.main import MainWindow  # noqa: E402
from src.gui.styles import get_combined_stylesheet  # noqa: E402
from src.gui.tabs import DashboardTab  # noqa: E402
from src.serving.peer_store import GlobalRunMeta, PeerStore  # noqa: E402
from src.serving.run_status import NightlyRunStatus, status_path_for_db  # noqa: E402


@pytest.fixture(scope="session")
def q_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _seed(tmp_path, with_status: bool = True) -> str:
    db = str(tmp_path / "serving_pool.db")
    store = PeerStore(db)
    rid = store.insert_run(GlobalRunMeta(model_name="EnsemblePooled", as_of_date="2026-11-15",
                                         n_symbols=3))
    store.insert_peer_scores(rid, pd.DataFrame({
        "symbol": ["TUPRS", "AKBNK", "ASELS"],
        "as_of_date": ["2026-11-15"] * 3,
        "peer_percentile": [81.0, 12.5, 55.0],
        "peer_label": ["outperform", "underperform", "inline"],
        "segment_sector": ["Industrials", "Banks", None],
        "trend_label": ["yukarı", "aşağı", "yatay"],
        "trend_prob_up": [0.56, 0.44, 0.50],
        "kolb_price_low": [97.0, 40.0, None],
        "kolb_price_high": [105.0, 44.0, None],
        "confidence_label": ["high", "low", "medium"],
        "confidence_reasons": [[], [], []], "confidence_warnings": [[], [], []],
    }))
    if with_status:
        NightlyRunStatus(status_path_for_db(db)).finish(0)
    return db


def test_styles_loading():
    qss = get_combined_stylesheet()
    assert isinstance(qss, str)
    for token in ("QMainWindow", "QPushButton", "status_card", "QMessageBox", "QTableView"):
        assert token in qss


def test_peer_reader_missing_db_does_not_create_file(tmp_path):
    db = tmp_path / "serving_pool.db"
    snap = PeerReader(str(db)).load()
    assert snap.run is None and snap.rows == [] and snap.error
    assert not db.exists()


def test_peer_reader_reads_latest_run_and_nightly_status(tmp_path):
    snap = PeerReader(_seed(tmp_path)).load()
    assert snap.run["as_of_date"] == "2026-11-15"
    assert {r["symbol"] for r in snap.rows} == {"TUPRS", "AKBNK", "ASELS"}
    assert snap.nightly["completed"] is True and snap.nightly["exit_code"] == 0
    # NaN -> None
    asels = next(r for r in snap.rows if r["symbol"] == "ASELS")
    assert asels["kolb_price_low"] is None


def test_dashboard_table_sorts_numerically_and_filters(q_app, tmp_path):
    tab = DashboardTab(PeerReader(_seed(tmp_path)))
    assert tab.model.rowCount() == 3
    headers = [tab.model.horizontalHeaderItem(i).text() for i in range(tab.model.columnCount())]
    assert headers[0] == "Hisse" and "Risk Seviyesi" in headers and "Fiyat Bandı Alt" in headers

    pct_col = headers.index("Akran Yüzdeliği")
    tab.table_view.sortByColumn(pct_col, Qt.AscendingOrder)
    assert tab.displayed_symbols() == ["AKBNK", "ASELS", "TUPRS"]  # 12.5 < 55 < 81 (metin degil sayi)
    tab.table_view.sortByColumn(pct_col, Qt.DescendingOrder)
    assert tab.displayed_symbols()[0] == "TUPRS"

    tab.txt_search.setText("akb")
    assert tab.displayed_symbols() == ["AKBNK"]
    tab.close()


def test_dashboard_missing_risk_columns_show_dash(q_app, tmp_path):
    tab = DashboardTab(PeerReader(_seed(tmp_path)))
    headers = [tab.model.horizontalHeaderItem(i).text() for i in range(tab.model.columnCount())]
    col = headers.index("Risk Seviyesi")
    assert all(tab.model.item(r, col).text() == "—" for r in range(tab.model.rowCount()))
    assert tab.card_asof.lbl_value.text() == "2026-11-15"
    assert tab.card_nightly.lbl_value.text() == "Başarılı"
    tab.close()


def test_dashboard_empty_state_shows_message(q_app, tmp_path):
    tab = DashboardTab(PeerReader(str(tmp_path / "none.db")))
    assert tab.model.rowCount() == 0 and tab.lbl_info.text()
    tab.close()


def test_main_window_single_tab_and_no_mojibake(q_app, tmp_path):
    window = MainWindow(PeerReader(_seed(tmp_path)))
    assert window.windowTitle() == "BIST AI Forecasting Lab - Karar Destek Paneli"
    assert window.tabs.count() == 1
    assert window.tabs.tabText(0) == "Gecelik Havuz Özeti"

    bad_markers = ("Ã", "Ä", "Å", "â", "Â", "�")
    texts = [window.windowTitle(), window.tabs.tabText(0)]
    texts += [window.dashboard_tab.model.horizontalHeaderItem(i).text()
              for i in range(window.dashboard_tab.model.columnCount())]
    for widget_type in (QLabel, QPushButton):
        texts += [w.text() for w in window.findChildren(widget_type)]
    assert all(not any(m in t for m in bad_markers) for t in texts)
    window.close()
