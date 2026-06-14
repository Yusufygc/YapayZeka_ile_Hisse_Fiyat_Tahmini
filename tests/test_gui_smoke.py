# -*- coding: utf-8 -*-
"""
tests.test_gui_smoke - Smoke tests for PySide6 Desktop GUI components.
"""

import sys
import os
import pytest

from PySide6.QtWidgets import QApplication

# Proje kÃ¶kÃ¼nÃ¼ sys.path'e ekle
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
    Test sÃ¼resince tek bir QApplication Ã¶rneÄŸi saÄŸlar.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_styles_loading():
    """
    Stil dosyalarÄ±nÄ±n baÅŸarÄ±yla okunup birleÅŸtirildiÄŸini doÄŸrular.
    """
    qss = get_combined_stylesheet()
    assert isinstance(qss, str)
    # Temel QSS Ã¶ÄŸelerinin iÃ§erikte bulunup bulunmadÄ±ÄŸÄ±nÄ± kontrol et
    assert "QMainWindow" in qss
    assert "QPushButton" in qss
    assert "status_card" in qss


def test_db_helper_smoke():
    """
    DBHelper sÄ±nÄ±fÄ±nÄ±n Ã§Ã¶kmeden baÅŸlatÄ±labildiÄŸini ve sorgu yapabildiÄŸini doÄŸrular.
    """
    db = DBHelper()
    # VeritabanÄ± olmasa bile fonksiyonlar boÅŸ liste dÃ¶nmeli, Ã§Ã¶kmemeli
    stocks = db.get_available_stocks()
    assert isinstance(stocks, list)

    best_models = db.get_best_models_summary()
    assert isinstance(best_models, list)


def test_gui_components_instantiation(q_app):
    """
    Ana pencere ve sekmelerin Ã§Ã¶kmeden baÅŸlatÄ±labildiÄŸini doÄŸrular.
    """
    db_helper = DBHelper()
    runner = ProcessRunner()

    # BaÄŸÄ±msÄ±z sekmelerin baÅŸlatÄ±lmasÄ±
    d_tab = DashboardTab(db_helper)
    assert d_tab is not None

    t_tab = TrainingTab(db_helper, runner)
    assert t_tab is not None

    f_tab = ForecastTab(db_helper, runner)
    assert f_tab is not None

    # Ana pencerenin baÅŸlatÄ±lmasÄ±
    window = MainWindow()
    assert window is not None
    assert window.windowTitle() == "BIST AI Forecasting Lab - Karar Destek Paneli"

    # Temizlik
    window.close()
    d_tab.close()
    t_tab.close()
    f_tab.close()
