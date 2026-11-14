# -*- coding: utf-8 -*-
"""
src.gui.tabs.dashboard_tab - Summarizes trained models and metrics.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableView, QPushButton, QLineEdit, QHeaderView
)
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt, QSortFilterProxyModel
from src.gui.core import DBHelper


class DashboardTab(QWidget):
    """
    Kayıtlı hisseleri ve en iyi modellerin metriklerini listeleyen dashboard.
    """
    def __init__(self, db_helper: DBHelper, parent=None):
        super().__init__(parent)
        self.db = db_helper
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Üst panel: başlık ve yenileme / arama
        top_layout = QHBoxLayout()
        lbl_title = QLabel("Model Liderlik Tablosu (Best Models)", self)
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.txt_search = QLineEdit(self)
        self.txt_search.setPlaceholderText("Hisse ara... (örn: TUPRS)")
        self.txt_search.setFixedWidth(200)
        self.txt_search.textChanged.connect(self._handle_search)

        self.btn_refresh = QPushButton("Verileri Yenile", self)
        self.btn_refresh.setFixedWidth(120)
        self.btn_refresh.clicked.connect(self.load_data)

        top_layout.addWidget(lbl_title)
        top_layout.addStretch()
        top_layout.addWidget(self.txt_search)
        top_layout.addWidget(self.btn_refresh)

        layout.addLayout(top_layout)

        # Tablo
        self.table_view = QTableView(self)
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)

        header = self.table_view.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(80)
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.table_view.verticalHeader().setVisible(False)

        # Veri Modeli
        self.model = QStandardItemModel(self)
        self.proxy_model = QSortFilterProxyModel(self)
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setFilterKeyColumn(0)  # Hisse sütununa göre arama yap
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)

        self.table_view.setModel(self.proxy_model)

        layout.addWidget(self.table_view)

        # İlk yükleme
        self.load_data()

    def load_data(self):
        """
        SQLite veritabanından best_models verilerini çeker ve tabloya yazar.
        """
        self.model.clear()

        headers = [
            "Hisse Senedi", "En İyi Model", "Genel Başarı (Skor)",
            "Yön Tahmini (%)", "Fiyat Yakınlığı (%)", "Kazanç Güvenilirliği (Sharpe)",
            "Ortalama Sapma (RMSE)", "Hata Payı (MAE)", "Son Güncelleme"
        ]
        self.model.setHorizontalHeaderLabels(headers)

        records = self.db.get_best_models_summary()
        if not records:
            # Boş tablo durumunda kullanıcıya bilgi ver
            self.model.setRowCount(0)
            return

        for record in records:
            row_items = [
                QStandardItem(str(record["stock_symbol"])),
                QStandardItem(str(record["model_name"])),
                QStandardItem(self._format_number(record["composite_score"], 2)),
                QStandardItem(self._format_percent(record["dir_acc"])),
                QStandardItem(self._format_percent(record["hit_rate"])),
                QStandardItem(self._format_number(record["sharpe"], 2)),
                QStandardItem(self._format_number(record["rmse"], 4)),
                QStandardItem(self._format_number(record["mae"], 4)),
                QStandardItem(str(record["updated_at"] or "—"))
            ]
            row_items[1].setToolTip(str(record["model_name"]))

            # Tüm elemanları hizalama
            for item in row_items:
                item.setTextAlignment(Qt.AlignCenter)

            self.model.appendRow(row_items)

        self._apply_column_widths()

    def _handle_search(self, text: str):
        """
        Arama filtresini günceller.
        """
        self.proxy_model.setFilterFixedString(text)

    @staticmethod
    def _format_number(value, decimals: int) -> str:
        return f"{value:.{decimals}f}" if value is not None else "—"

    @staticmethod
    def _format_percent(value) -> str:
        return f"{value:.1f}%" if value is not None else "—"

    def _apply_column_widths(self):
        widths = [110, 260, 150, 140, 155, 220, 170, 140, 170]
        for idx, width in enumerate(widths):
            self.table_view.setColumnWidth(idx, width)
