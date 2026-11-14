# -*- coding: utf-8 -*-
"""
src.gui.main - Application entry point.
PySide6 uygulamasını başlatır, modüler stilleri yükler ve sekmeleri bağlar.
"""

import sys
import os

# Proje kökünü sys.path'e ekleyelim
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
    Ana masaüstü uygulama penceresi.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BIST AI Forecasting Lab - Karar Destek Paneli")
        self.resize(1000, 700)

        # Temel bağımlılıkların başlatılması (Dependency Injection)
        self.db_helper = DBHelper()
        self.train_runner = ProcessRunner(self)
        self.forecast_runner = ProcessRunner(self)

        self.init_ui()
        self.statusBar().showMessage("Sistem hazır. Veritabanı bağlantısı kuruldu.")

    def init_ui(self):
        # Sekmeli düzen
        self.tabs = QTabWidget(self)
        self.tabs.setMovable(False)

        # Sekmelerin oluşturulması ve DI ile bağımlılıkların geçilmesi
        self.dashboard_tab = DashboardTab(self.db_helper, self)
        self.training_tab = TrainingTab(self.db_helper, self.train_runner, self)
        self.forecast_tab = ForecastTab(self.db_helper, self.forecast_runner, self)

        # Sekmelerin Eklenmesi
        self.tabs.addTab(self.dashboard_tab, "Özet ve Liderlik Tablosu")
        self.tabs.addTab(self.training_tab, "Model Eğitimi Kontrolü")
        self.tabs.addTab(self.forecast_tab, "İleri Yönlü Tahminler (XAI)")

        # Sekme geçişlerinde verileri otomatik tazele (UX iyileştirmesi)
        self.tabs.currentChanged.connect(self._handle_tab_changed)

        self.setCentralWidget(self.tabs)

    def _handle_tab_changed(self, index: int):
        """
        Sekme değiştirildiğinde ilgili sekmeleri günceller.
        """
        if index == 0:  # Dashboard Tab
            self.dashboard_tab.load_data()
            self.statusBar().showMessage("Liderlik tablosu verileri güncellendi.")
        elif index == 2:  # Forecast Tab
            self.forecast_tab._load_forecast_data()
            self.statusBar().showMessage("Tahmin verileri yüklendi.")


def main():
    app = QApplication(sys.argv)

    # Modüler QSS stillerini yükle ve uygula
    qss = get_combined_stylesheet()
    app.setStyleSheet(qss)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
