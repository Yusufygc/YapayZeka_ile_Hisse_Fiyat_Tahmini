# -*- coding: utf-8 -*-
"""
src.gui.tabs.dashboard_tab - Summarizes trained models and metrics.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableView, QPushButton, QLineEdit
)
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt, QSortFilterProxyModel
from src.gui.core import DBHelper


class DashboardTab(QWidget):
    """
    KayÄ±tlÄ± hisseleri ve en iyi modellerin metriklerini listeleyen Dashboard.
    """
    def __init__(self, db_helper: DBHelper, parent=None):
        super().__init__(parent)
        self.db = db_helper
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Ãœst Panel: BaÅŸlÄ±k ve Yenile / Arama
        top_layout = QHBoxLayout()
        lbl_title = QLabel("Model Liderlik Tablosu (Best Models)", self)
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.txt_search = QLineEdit(self)
        self.txt_search.setPlaceholderText("Hisse ara... (Ã¶rn: TUPRS)")
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

        # SÃ¼tun geniÅŸliklerinin iÃ§eriÄŸe uymasÄ±
        self.table_view.horizontalHeader().setStretchLastSection(True)

        # Veri Modeli
        self.model = QStandardItemModel(self)
        self.proxy_model = QSortFilterProxyModel(self)
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setFilterKeyColumn(0)  # Hisse sÃ¼tununa gÃ¶re arama yap
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)

        self.table_view.setModel(self.proxy_model)

        layout.addWidget(self.table_view)

        # Ä°lk yÃ¼kleme
        self.load_data()

    def load_data(self):
        """
        SQLite veritabanÄ±ndan best_models verilerini Ã§eker ve tabloya yazar.
        """
        self.model.clear()

        headers = [
            "Hisse Senedi", "En Ä°yi Model", "Genel BaÅŸarÄ± (Skor)",
            "YÃ¶n Tahmini (%)", "Fiyat YakÄ±nlÄ±ÄŸÄ± (%)", "KazanÃ§ GÃ¼venilirliÄŸi (Sharpe)",
            "Ortalama Sapma (RMSE)", "Hata PayÄ± (MAE)", "Son GÃ¼ncelleme"
        ]
        self.model.setHorizontalHeaderLabels(headers)

        records = self.db.get_best_models_summary()
        if not records:
            # BoÅŸ tablo durumunda kullanÄ±cÄ±ya bilgi ver
            self.model.setRowCount(0)
            return

        for record in records:
            row_items = [
                QStandardItem(str(record["stock_symbol"])),
                QStandardItem(str(record["model_name"])),
                QStandardItem(f"{record['composite_score']:.2f}" if record['composite_score'] is not None else "â€”"),
                QStandardItem(f"{record['dir_acc']*100:.1f}%" if record['dir_acc'] is not None else "â€”"),
                QStandardItem(f"{record['hit_rate']*100:.1f}%" if record['hit_rate'] is not None else "â€”"),
                QStandardItem(f"{record['sharpe']:.2f}" if record['sharpe'] is not None else "â€”"),
                QStandardItem(f"{record['rmse']:.4f}" if record['rmse'] is not None else "â€”"),
                QStandardItem(f"{record['mae']:.4f}" if record['mae'] is not None else "â€”"),
                QStandardItem(str(record["updated_at"] or "â€”"))
            ]

            # TÃ¼m elemanlarÄ± hizalama
            for item in row_items:
                item.setTextAlignment(Qt.AlignCenter)

            self.model.appendRow(row_items)

        # SÃ¼tun geniÅŸliklerini ayarla
        self.table_view.resizeColumnsToContents()

    def _handle_search(self, text: str):
        """
        Arama filtresini gÃ¼nceller.
        """
        self.proxy_model.setFilterFixedString(text)
