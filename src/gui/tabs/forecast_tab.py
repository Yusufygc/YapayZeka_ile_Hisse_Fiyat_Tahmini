# -*- coding: utf-8 -*-
"""
src.gui.tabs.forecast_tab - Surfaces forward forecasts and performance metrics.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QGroupBox, QTableView, QMessageBox, QSplitter,
    QSizePolicy, QHeaderView
)
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt
from src.gui.core import DBHelper, ProcessRunner
from src.gui.widgets import MetricCard, TrendCard, LogViewer


class ForecastTab(QWidget):
    """
    Seçilen hisse için ileriye dönük tahminleri gösteren ve tetikleyen panel.
    """
    def __init__(self, db_helper: DBHelper, runner: ProcessRunner, parent=None):
        super().__init__(parent)
        self.db = db_helper
        self.runner = runner

        # Sinyal bağlantıları
        self.runner.output_received.connect(self._append_log)
        self.runner.finished.connect(self._forecast_finished)

        self.init_ui()

    def init_ui(self):
        # Ana dikey düzen
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        # Üst yapılandırma
        top_box = QGroupBox("Tahmin Kontrolleri", self)
        top_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_box.setMaximumHeight(120)
        top_layout = QHBoxLayout(top_box)
        top_layout.setSpacing(16)
        top_layout.setContentsMargins(12, 16, 12, 12)

        top_layout.addWidget(QLabel("Hisse Senedi:", self))
        self.cmb_stock = QComboBox(self)
        self.cmb_stock.setEditable(True)
        self.cmb_stock.setPlaceholderText("Hisse seçin...")
        self.cmb_stock.setFixedWidth(180)
        self.cmb_stock.currentIndexChanged.connect(self._handle_stock_change)
        top_layout.addWidget(self.cmb_stock)

        self.btn_show = QPushButton("Tahmini Göster", self)
        self.btn_show.clicked.connect(self._load_forecast_data)
        top_layout.addWidget(self.btn_show)

        self.btn_generate = QPushButton("Yeni Tahmin Üret (forecast.py)", self)
        self.btn_generate.clicked.connect(self._generate_new_forecast)
        top_layout.addWidget(self.btn_generate)

        top_layout.addStretch()
        main_layout.addWidget(top_box)

        main_layout.setAlignment(top_box, Qt.AlignTop)

        # Orta kısım: kartlar ve tablo
        splitter = QSplitter(Qt.Horizontal, self)

        # Sol panel: durum kartları
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        self.card_last_price = MetricCard("Son Kapanış", "— TL", "Son gözlemlenen veri tarihi")
        self.card_trend = TrendCard("Yön Eğilimi", "Belirsiz", "belirsiz", "Modelin tahmin yönü eğilimi")
        self.card_return = MetricCard("Haftalık Beklenen Getiri", "— %", "BIST 5 işlem günü beklentisi")
        self.card_model = MetricCard("Kullanılan Model", "—", "Tahmin üreten aktif model")

        left_layout.addWidget(self.card_last_price)
        left_layout.addWidget(self.card_trend)
        left_layout.addWidget(self.card_return)
        left_layout.addWidget(self.card_model)
        left_layout.addStretch()

        splitter.addWidget(left_widget)

        # Sağ panel: tahmin tablosu
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        table_box = QGroupBox("BIST 5 Günlük Tahmin Noktaları", self)
        table_layout = QVBoxLayout(table_box)

        self.empty_state = QLabel("Tahmin görüntülemek için bir hisse seçin.", self)
        self.empty_state.setObjectName("empty_state")
        self.empty_state.setAlignment(Qt.AlignCenter)
        self.empty_state.setWordWrap(True)

        self.table_view = QTableView(self)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setStretchLastSection(False)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table_view.verticalHeader().setVisible(False)

        self.model = QStandardItemModel(self)
        self.table_view.setModel(self.model)

        table_layout.addWidget(self.empty_state)
        table_layout.addWidget(self.table_view)
        right_layout.addWidget(table_box)

        splitter.addWidget(right_widget)

        # Genişlik oranlarını ayarla
        splitter.setSizes([280, 720])
        main_layout.addWidget(splitter)

        # Alt panel: log ekranı (sadece yeni tahmin üretilirken görünür/kullanılır)
        self.log_box = QGroupBox("Tahmin Üretim Logları", self)
        log_layout = QVBoxLayout(self.log_box)
        self.log_viewer = LogViewer(self)
        log_layout.addWidget(self.log_viewer)
        self.log_box.setVisible(False)  # Varsayılan olarak gizle

        main_layout.addWidget(self.log_box)

        # Hisseleri yükle
        self._load_stocks()
        self._clear_ui()

    def _load_stocks(self):
        """
        Kullanılabilir hisseleri yükler.
        """
        self.cmb_stock.clear()
        stocks = self.db.get_available_stocks()
        self.cmb_stock.addItems(stocks)

    def _handle_stock_change(self):
        """
        Hisse değiştiğinde kartları sıfırla.
        """
        self._clear_ui()

    def _clear_ui(self):
        self.card_last_price.update_value("— TL", "Son gözlemlenen veri tarihi")
        self.card_trend.update_value("Belirsiz", "Modelin tahmin yönü eğilimi")
        self.card_trend.update_trend_style("belirsiz")
        self.card_return.update_value("— %", "BIST 5 işlem günü beklentisi")
        self.card_model.update_value("—", "Tahmin üreten aktif model")
        self.model.clear()
        self.empty_state.setText(self._empty_state_prompt())
        self.empty_state.setVisible(True)
        self.table_view.setVisible(False)

    def _load_forecast_data(self):
        """
        SQLite'tan seçili hissenin son başarılı tahmin verisini çeker ve GUI'yi günceller.
        """
        symbol = self.cmb_stock.currentText().strip()
        if not symbol:
            return

        self._clear_ui()
        data = self.db.get_forecasts_by_symbol(symbol)

        if not data:
            self.empty_state.setText(
                f"{symbol} için kayıtlı aktif tahmin bulunamadı.\n"
                "Yeni Tahmin Üret butonuyla tahmin oluşturabilirsiniz."
            )
            self.empty_state.setVisible(True)
            self.table_view.setVisible(False)
            return

        run = data["run"]
        points = data["points"]

        # Kartları güncelle
        self.card_last_price.update_value(f"{run['last_close']:.2f} TL", f"Veri Tarihi: {run['last_observed_date']}")

        trend = run["trend_label"]
        self.card_trend.update_value(trend.upper())
        self.card_trend.update_trend_style(trend)

        ret = run["weekly_expected_return"]
        self.card_return.update_value(f"{ret*100:+.2f}%" if ret is not None else "— %")
        self.card_model.update_value(run["model_name"], f"Oturum Zamanı: {run['run_at']}")

        # Tabloyu doldur
        headers = ["Hedef Tarih", "İşlem Günü (Adım)", "Kapanış Tahmini", "Tahmin Edilen Getiri", "Alt Bant (%80)", "Üst Bant (%80)"]
        self.model.setHorizontalHeaderLabels(headers)

        for pt in points:
            raw_close = pt["raw_predicted_close"] or pt["bounded_predicted_close"] or 0.0
            ret_val = pt["predicted_return"] or 0.0

            row_items = [
                QStandardItem(str(pt["target_date"])),
                QStandardItem(f"Gün {pt['horizon_index']}"),
                QStandardItem(f"{raw_close:.2f} TL"),
                QStandardItem(f"{ret_val*100:+.2f}%"),
                QStandardItem(f"{pt['lower_band']:.2f} TL" if pt['lower_band'] is not None else "—"),
                QStandardItem(f"{pt['upper_band']:.2f} TL" if pt['upper_band'] is not None else "—")
            ]

            for item in row_items:
                item.setTextAlignment(Qt.AlignCenter)

            self.model.appendRow(row_items)

        self.empty_state.setVisible(False)
        self.table_view.setVisible(True)
        self._apply_column_widths()

    def _generate_new_forecast(self):
        """
        forecast.py modülünü QProcess ile arka planda tetikler.
        """
        symbol = self.cmb_stock.currentText().strip()
        if not symbol:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir hisse senedi seçin.")
            return

        self._clear_ui()
        self.log_box.setVisible(True)
        self.log_viewer.clear()

        # Butonları kilitle (UX)
        self.btn_show.setEnabled(False)
        self.btn_generate.setEnabled(False)
        self.cmb_stock.setEnabled(False)

        # forecast.py parametreleri: --stocks TUPRS --horizon-days 5
        args = ["--stocks", symbol, "--horizon-days", "5", "--verbose"]

        self.log_viewer.append_text(f"[GUI] {symbol} için yeni tahmin hesaplaması başlatılıyor...\n")
        self.runner.start_process("src.cli.forecast", args)

    def _append_log(self, text: str):
        if self.log_box.isVisible():
            self.log_viewer.append_text(text)

    def _forecast_finished(self, exit_code: int, message: str):
        """
        Tahmin süreci bittiğinde arayüzü çözer ve veriyi yükler.
        """
        self.btn_show.setEnabled(True)
        self.btn_generate.setEnabled(True)
        self.cmb_stock.setEnabled(True)

        if exit_code == 0:
            self.log_viewer.append_text("[GUI] Tahmin başarıyla tamamlandı. Veriler yükleniyor...\n")
            self._load_forecast_data()
        else:
            self.empty_state.setText(
                "Tahmin üretimi tamamlanamadı.\n"
                "Ayrıntılar için alttaki üretim loglarını kontrol edin."
            )
            self.empty_state.setVisible(True)
            self.table_view.setVisible(False)

    def _apply_column_widths(self):
        widths = [150, 150, 170, 180, 150, 150]
        for idx, width in enumerate(widths):
            self.table_view.setColumnWidth(idx, width)

    def _empty_state_prompt(self) -> str:
        symbol = self.cmb_stock.currentText().strip() if hasattr(self, "cmb_stock") else ""
        if symbol:
            return f"{symbol} için tahmin görüntülemek üzere 'Tahmini Göster' düğmesine tıklayın."
        return "Tahmin görüntülemek için bir hisse seçin."
