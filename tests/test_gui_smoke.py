# -*- coding: utf-8 -*-
"""
tests.test_gui_smoke - Smoke tests for PySide6 Desktop GUI components.
"""

import sys
import os
import pytest

from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QComboBox, QGroupBox

# Proje kökünü sys.path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.gui.core import DBHelper, ProcessRunner
from src.gui.styles import get_combined_stylesheet
from src.gui.tabs import DashboardTab, TrainingTab, ForecastTab
from src.gui.main import MainWindow


@pytest.fixture(scope="session")
def q_app():
    """
    Test süresince tek bir QApplication örneği sağlar.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_styles_loading():
    """
    Stil dosyalarının başarıyla okunup birleştirildiğini doğrular.
    """
    qss = get_combined_stylesheet()
    assert isinstance(qss, str)
    # Temel QSS öğelerinin içerikte bulunup bulunmadığını kontrol et
    assert "QMainWindow" in qss
    assert "QPushButton" in qss
    assert "status_card" in qss
    assert "QMessageBox" in qss


def test_db_helper_smoke():
    """
    DBHelper sınıfının çökmeden başlatılabildiğini ve sorgu yapabildiğini doğrular.
    """
    db = DBHelper()
    # Veritabanı olmasa bile fonksiyonlar boş liste dönmeli, çökmemeli
    stocks = db.get_available_stocks()
    assert isinstance(stocks, list)

    best_models = db.get_best_models_summary()
    assert isinstance(best_models, list)


def test_gui_components_instantiation(q_app):
    """
    Ana pencere ve sekmelerin çökmeden başlatılabildiğini doğrular.
    """
    db_helper = DBHelper()
    runner = ProcessRunner()

    # Bağımsız sekmelerin başlatılması
    d_tab = DashboardTab(db_helper)
    assert d_tab is not None

    t_tab = TrainingTab(db_helper, runner)
    assert t_tab is not None

    f_tab = ForecastTab(db_helper, runner)
    assert f_tab is not None

    # Ana pencerenin başlatılması
    window = MainWindow()
    assert window is not None
    assert window.windowTitle() == "BIST AI Forecasting Lab - Karar Destek Paneli"
    assert window.tabs.tabText(0) == "Özet ve Liderlik Tablosu"
    assert window.tabs.tabText(1) == "Model Eğitimi Kontrolü"
    assert window.tabs.tabText(2) == "İleri Yönlü Tahminler (XAI)"

    # Temizlik
    window.close()
    d_tab.close()
    t_tab.close()
    f_tab.close()


def test_visible_gui_text_has_no_mojibake(q_app):
    """
    Görünür GUI metinlerinde UTF-8 mojibake belirteci kalmadığını doğrular.
    """
    bad_markers = ("\u00c3", "\u00c4", "\u00c5", "\u00e2", "\u00c2", "\ufffd")
    window = MainWindow()
    texts = [
        window.windowTitle(),
        *(window.tabs.tabText(i) for i in range(window.tabs.count())),
        *(window.dashboard_tab.model.horizontalHeaderItem(i).text() for i in range(window.dashboard_tab.model.columnCount())),
        window.training_tab.cmb_stock.itemText(0),
        *(window.training_tab.cmb_preset.itemText(i) for i in range(window.training_tab.cmb_preset.count())),
        window.forecast_tab.empty_state.text(),
    ]

    for widget_type in (QLabel, QPushButton, QComboBox, QGroupBox):
        for widget in window.findChildren(widget_type):
            if hasattr(widget, "text"):
                texts.append(widget.text())
            if hasattr(widget, "placeholderText"):
                texts.append(widget.placeholderText())

    assert all(not any(marker in text for marker in bad_markers) for text in texts)
    window.close()


def test_dashboard_percent_values_are_not_double_scaled(q_app):
    """
    Dashboard yüzdeleri veritabanındaki 0-100 formatını tekrar 100 ile çarpmaz.
    """
    assert DashboardTab._format_percent(66.1) == "66.1%"
    assert DashboardTab._format_percent(None) == "—"


def test_process_runner_decodes_legacy_windows_turkish_output():
    """
    CP1254 üreten eski Windows CLI çıktısı replacement karakterine düşmez.
    """
    raw = "Model eğitim süreci başlatılıyor".encode("cp1254")
    assert ProcessRunner._decode_output(raw) == "Model eğitim süreci başlatılıyor"


def test_forecast_empty_state_mentions_selected_symbol(q_app):
    """
    Forecast sekmesi seçili sembol varken genel "hisse seçin" mesajında kalmaz.
    """
    db_helper = DBHelper()
    runner = ProcessRunner()
    f_tab = ForecastTab(db_helper, runner)
    f_tab.cmb_stock.setEditText("GARAN")
    f_tab._clear_ui()

    assert "GARAN" in f_tab.empty_state.text()
    assert "Tahmini Göster" in f_tab.empty_state.text()
    f_tab.close()
