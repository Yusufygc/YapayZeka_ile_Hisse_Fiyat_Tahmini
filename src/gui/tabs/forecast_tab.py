# -*- coding: utf-8 -*-
"""
src.gui.tabs.forecast_tab - Surfaces forward forecasts and performance metrics.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QGroupBox, QTableView, QMessageBox, QSplitter
)
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt
from src.gui.core import DBHelper, ProcessRunner
from src.gui.widgets import MetricCard, TrendCard, LogViewer


class ForecastTab(QWidget):
    """
    SeÃ§ilen hisse iÃ§in ileriye dÃ¶nÃ¼k tahminleri gÃ¶steren ve tetikleyen panel.
    """
    def __init__(self, db_helper: DBHelper, runner: ProcessRunner, parent=None):
        super().__init__(parent)
        self.db = db_helper
        self.runner = runner

        # Sinyal baÄŸlantÄ±larÄ±
        self.runner.output_received.connect(self._append_log)
        self.runner.finished.connect(self._forecast_finished)

        self.init_ui()

    def init_ui(self):
        # Ana Dikey DÃ¼zen
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        # Ãœst YapÄ±landÄ±rma
        top_box = QGroupBox("Tahmin Kontrolleri", self)
        top_layout = QHBoxLayout(top_box)
        top_layout.setSpacing(16)

        top_layout.addWidget(QLabel("Hisse Senedi:", self))
        self.cmb_stock = QComboBox(self)
        self.cmb_stock.setEditable(True)
        self.cmb_stock.setPlaceholderText("Hisse seÃ§in...")
        self.cmb_stock.setFixedWidth(150)
        self.cmb_stock.currentIndexChanged.connect(self._handle_stock_change)
        top_layout.addWidget(self.cmb_stock)

        self.btn_show = QPushButton("Tahmini GÃ¶ster", self)
        self.btn_show.clicked.connect(self._load_forecast_data)
        top_layout.addWidget(self.btn_show)

        self.btn_generate = QPushButton("Yeni Tahmin Ãœret (forecast.py)", self)
        self.btn_generate.clicked.connect(self._generate_new_forecast)
        top_layout.addWidget(self.btn_generate)

        top_layout.addStretch()
        main_layout.addWidget(top_box)

        # Orta KÄ±sÄ±m: Kartlar ve Tablo (Splitter ile ikiye bÃ¶lelim)
        splitter = QSplitter(Qt.Horizontal, self)

        # Sol Panel: Durum KartlarÄ±
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        self.card_last_price = MetricCard("Son KapanÄ±ÅŸ", "â€” TL", "Son gÃ¶zlemlenen veri tarihi")
        self.card_trend = TrendCard("YÃ¶n EÄŸilimi", "Belirsiz", "belirsiz", "Modelin tahmin yÃ¶nÃ¼ eÄŸilimi")
        self.card_return = MetricCard("HaftalÄ±k Beklenen Getiri", "â€” %", "BIST 5 iÅŸlem gÃ¼nÃ¼ beklentisi")
        self.card_model = MetricCard("KullanÄ±lan Model", "â€”", "Tahmin Ã¼reten aktif model")

        left_layout.addWidget(self.card_last_price)
        left_layout.addWidget(self.card_trend)
        left_layout.addWidget(self.card_return)
        left_layout.addWidget(self.card_model)
        left_layout.addStretch()

        splitter.addWidget(left_widget)

        # SaÄŸ Panel: Tahmin Tablosu
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        table_box = QGroupBox("BIST 5 GÃ¼nlÃ¼k Tahmin NoktalarÄ±", self)
        table_layout = QVBoxLayout(table_box)

        self.table_view = QTableView(self)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setStretchLastSection(True)

        self.model = QStandardItemModel(self)
        self.table_view.setModel(self.model)

        table_layout.addWidget(self.table_view)
        right_layout.addWidget(table_box)

        splitter.addWidget(right_widget)

        # GeniÅŸlik oranlarÄ±nÄ± ayarla
        splitter.setSizes([250, 500])
        main_layout.addWidget(splitter)

        # Alt Panel: Log EkranÄ± (Sadece yeni tahmin Ã¼retilirken gÃ¶rÃ¼nÃ¼r/kullanÄ±lÄ±r)
        self.log_box = QGroupBox("Tahmin Ãœretim LoglarÄ±", self)
        log_layout = QVBoxLayout(self.log_box)
        self.log_viewer = LogViewer(self)
        log_layout.addWidget(self.log_viewer)
        self.log_box.setVisible(False)  # VarsayÄ±lan olarak gizle

        main_layout.addWidget(self.log_box)

        # Hisseleri yÃ¼kle
        self._load_stocks()

    def _load_stocks(self):
        """
        KullanÄ±labilir hisseleri yÃ¼kler.
        """
        self.cmb_stock.clear()
        stocks = self.db.get_available_stocks()
        self.cmb_stock.addItems(stocks)

    def _handle_stock_change(self):
        """
        Hisse deÄŸiÅŸtiÄŸinde kartlarÄ± sÄ±fÄ±rla.
        """
        self._clear_ui()

    def _clear_ui(self):
        self.card_last_price.update_value("â€” TL", "Son gÃ¶zlemlenen veri tarihi")
        self.card_trend.update_value("Belirsiz", "Modelin tahmin yÃ¶nÃ¼ eÄŸilimi")
        self.card_trend.update_trend_style("belirsiz")
        self.card_return.update_value("â€” %", "BIST 5 iÅŸlem gÃ¼nÃ¼ beklentisi")
        self.card_model.update_value("â€”", "Tahmin Ã¼reten aktif model")
        self.model.clear()

    def _load_forecast_data(self):
        """
        SQLite'tan seÃ§ili hissenin son baÅŸarÄ±lÄ± tahmin verisini Ã§eker ve GUI'yi gÃ¼nceller.
        """
        symbol = self.cmb_stock.currentText().strip()
        if not symbol:
            return

        self._clear_ui()
        data = self.db.get_forecasts_by_symbol(symbol)

        if not data:
            QMessageBox.information(
                self, "Bilgi",
                f"'{symbol}' hissesi iÃ§in veritabanÄ±nda kayÄ±tlÄ± aktif tahmin bulunamadÄ±.\n"
                "LÃ¼tfen 'Yeni Tahmin Ãœret' butonuna tÄ±klayarak tahmin oluÅŸturun."
            )
            return

        run = data["run"]
        points = data["points"]

        # KartlarÄ± GÃ¼ncelle
        self.card_last_price.update_value(f"{run['last_close']:.2f} TL", f"Veri Tarihi: {run['last_observed_date']}")

        trend = run["trend_label"]
        self.card_trend.update_value(trend.upper())
        self.card_trend.update_trend_style(trend)

        ret = run["weekly_expected_return"]
        self.card_return.update_value(f"{ret*100:+.2f}%" if ret is not None else "â€” %")
        self.card_model.update_value(run["model_name"], f"Oturum ZamanÄ±: {run['run_at']}")

        # Tabloyu Doldur
        headers = ["Hedef Tarih", "Ä°ÅŸlem GÃ¼nÃ¼ (AdÄ±m)", "KapanÄ±ÅŸ Tahmini", "Tahmin Edilen Getiri", "Alt Bant (%80)", "Ãœst Bant (%80)"]
        self.model.setHorizontalHeaderLabels(headers)

        for pt in points:
            raw_close = pt["raw_predicted_close"] or pt["bounded_predicted_close"] or 0.0
            ret_val = pt["predicted_return"] or 0.0

            row_items = [
                QStandardItem(str(pt["target_date"])),
                QStandardItem(f"GÃ¼n {pt['horizon_index']}"),
                QStandardItem(f"{raw_close:.2f} TL"),
                QStandardItem(f"{ret_val*100:+.2f}%"),
                QStandardItem(f"{pt['lower_band']:.2f} TL" if pt['lower_band'] is not None else "â€”"),
                QStandardItem(f"{pt['upper_band']:.2f} TL" if pt['upper_band'] is not None else "â€”")
            ]

            for item in row_items:
                item.setTextAlignment(Qt.AlignCenter)

            self.model.appendRow(row_items)

        self.table_view.resizeColumnsToContents()

    def _generate_new_forecast(self):
        """
        forecast.py modÃ¼lÃ¼nÃ¼ QProcess ile arka planda tetikler.
        """
        symbol = self.cmb_stock.currentText().strip()
        if not symbol:
            QMessageBox.warning(self, "UyarÄ±", "LÃ¼tfen bir hisse senedi seÃ§in.")
            return

        self._clear_ui()
        self.log_box.setVisible(True)
        self.log_viewer.clear()

        # ButonlarÄ± kilitle (UX)
        self.btn_show.setEnabled(False)
        self.btn_generate.setEnabled(False)
        self.cmb_stock.setEnabled(False)

        # forecast.py parametreleri: --stocks TUPRS --horizon-days 5
        args = ["--stocks", symbol, "--horizon-days", "5", "--verbose"]

        self.log_viewer.append_text(f"[GUI] {symbol} iÃ§in yeni tahmin hesaplamasÄ± baÅŸlatÄ±lÄ±yor...\n")
        self.runner.start_process("src.cli.forecast", args)

    def _append_log(self, text: str):
        if self.log_box.isVisible():
            self.log_viewer.append_text(text)

    def _forecast_finished(self, exit_code: int, message: str):
        """
        Tahmin sÃ¼reci bittiÄŸinde arayÃ¼zÃ¼ Ã§Ã¶zer ve veriyi yÃ¼kler.
        """
        self.btn_show.setEnabled(True)
        self.btn_generate.setEnabled(True)
        self.cmb_stock.setEnabled(True)

        if exit_code == 0:
            self.log_viewer.append_text("[GUI] Tahmin baÅŸarÄ±yla tamamlandÄ±. Veriler yÃ¼kleniyor...\n")
            self._load_forecast_data()
        else:
            QMessageBox.warning(self, "Hata", "Tahmin Ã¼retilirken bir hata oluÅŸtu veya iÅŸlem sonlandÄ±rÄ±ldÄ±.")
