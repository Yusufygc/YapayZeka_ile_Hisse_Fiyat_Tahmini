# -*- coding: utf-8 -*-
"""
src.gui.main - Application entry point.
PySide6 uygulamasını başlatır, modüler stilleri yükler ve gecelik havuz özetini gösterir.
"""

import os
import sys

# Proje kökünü sys.path'e ekleyelim
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from src.gui.core import PeerReader
from src.gui.styles import get_combined_stylesheet
from src.gui.tabs import DashboardTab


class MainWindow(QMainWindow):
    """Ana masaüstü uygulama penceresi."""

    def __init__(self, reader: PeerReader | None = None):
        super().__init__()
        self.setWindowTitle("BIST AI Forecasting Lab - Karar Destek Paneli")
        self.resize(1100, 700)

        self.reader = reader or PeerReader()
        self.init_ui()
        self.statusBar().showMessage("Gecelik havuz verisi yüklendi.")

    def init_ui(self) -> None:
        self.tabs = QTabWidget(self)
        self.tabs.setMovable(False)
        self.dashboard_tab = DashboardTab(self.reader, self)
        self.tabs.addTab(self.dashboard_tab, "Gecelik Havuz Özeti")
        self.setCentralWidget(self.tabs)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(get_combined_stylesheet())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
