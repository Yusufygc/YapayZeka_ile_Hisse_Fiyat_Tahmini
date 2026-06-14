# -*- coding: utf-8 -*-
"""
src.gui.main - Application entry point.
PySide6 uygulamasÄ±nÄ± baÅŸlatÄ±r, modÃ¼ler stilleri yÃ¼kler ve sekmeleri baÄŸlar.
"""

import sys
import os

# Proje kÃ¶kÃ¼nÃ¼ sys.path'e ekleyelim
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt

from src.gui.styles import get_combined_stylesheet
from src.gui.core import DBHelper, ProcessRunner
from src.gui.tabs import DashboardTab, TrainingTab, ForecastTab


class MainWindow(QMainWindow):
    """
    Ana MasaÃ¼stÃ¼ Uygulama Penceresi.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BIST AI Forecasting Lab - Karar Destek Paneli")
        self.resize(1000, 700)

        # Temel BaÄŸÄ±mlÄ±lÄ±klarÄ±n BaÅŸlatÄ±lmasÄ± (Dependency Injection)
        self.db_helper = DBHelper()
        self.train_runner = ProcessRunner(self)
        self.forecast_runner = ProcessRunner(self)

        self.init_ui()
        self.statusBar().showMessage("Sistem hazÄ±r. VeritabanÄ± baÄŸlantÄ±sÄ± kuruldu.")

    def init_ui(self):
        # Sekmeli DÃ¼zen
        self.tabs = QTabWidget(self)
        self.tabs.setMovable(False)

        # Sekmelerin OluÅŸturulmasÄ± ve DI ile baÄŸÄ±mlÄ±lÄ±klarÄ±n geÃ§ilmesi
        self.dashboard_tab = DashboardTab(self.db_helper, self)
        self.training_tab = TrainingTab(self.db_helper, self.train_runner, self)
        self.forecast_tab = ForecastTab(self.db_helper, self.forecast_runner, self)

        # Sekmelerin Eklenmesi
        self.tabs.addTab(self.dashboard_tab, "Ã–zet ve Liderlik Tablosu")
        self.tabs.addTab(self.training_tab, "Model EÄŸitimi KontrolÃ¼")
        self.tabs.addTab(self.forecast_tab, "Ä°leri YÃ¶nlÃ¼ Tahminler (XAI)")

        # Sekme geÃ§iÅŸlerinde verileri otomatik tazele (UX Ä°yileÅŸtirmesi)
        self.tabs.currentChanged.connect(self._handle_tab_changed)

        self.setCentralWidget(self.tabs)

    def _handle_tab_changed(self, index: int):
        """
        Sekme deÄŸiÅŸtirildiÄŸinde ilgili sekmeleri gÃ¼nceller.
        """
        if index == 0:  # Dashboard Tab
            self.dashboard_tab.load_data()
            self.statusBar().showMessage("Liderlik tablosu verileri gÃ¼ncellendi.")
        elif index == 2:  # Forecast Tab
            self.forecast_tab._load_forecast_data()
            self.statusBar().showMessage("Tahmin verileri yÃ¼klendi.")


def main():
    app = QApplication(sys.argv)

    # ModÃ¼ler QSS stillerini yÃ¼kle ve uygula
    qss = get_combined_stylesheet()
    app.setStyleSheet(qss)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
