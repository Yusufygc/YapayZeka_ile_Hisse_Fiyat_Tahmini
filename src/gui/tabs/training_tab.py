# -*- coding: utf-8 -*-
"""
src.gui.tabs.training_tab - Triggers and monitors batch training runs.
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSpinBox, QPushButton, QGroupBox, QMessageBox
)
from src.gui.core import DBHelper, ProcessRunner
from src.gui.widgets import LogViewer


class TrainingTab(QWidget):
    """
    Model eÄŸitimi yapÄ±landÄ±rma ve baÅŸlatma paneli.
    """
    def __init__(self, db_helper: DBHelper, runner: ProcessRunner, parent=None):
        super().__init__(parent)
        self.db = db_helper
        self.runner = runner

        # Sinyal baÄŸlantÄ±larÄ±
        self.runner.output_received.connect(self._append_log)
        self.runner.finished.connect(self._training_finished)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # YapÄ±landÄ±rma Paneli (GroupBox)
        config_box = QGroupBox("EÄŸitim YapÄ±landÄ±rmasÄ±", self)
        config_layout = QHBoxLayout(config_box)
        config_layout.setSpacing(20)

        # 1. Hisse SeÃ§imi
        vbox_stock = QVBoxLayout()
        vbox_stock.addWidget(QLabel("Hisse Senedi:", self))
        self.cmb_stock = QComboBox(self)
        self.cmb_stock.setEditable(True)  # Kolay arama desteÄŸi
        self.cmb_stock.setPlaceholderText("Hisse seÃ§in veya arayÄ±n...")
        self.cmb_stock.setFixedWidth(150)
        vbox_stock.addWidget(self.cmb_stock)
        config_layout.addLayout(vbox_stock)

        # 2. Model SeÃ§im Presetleri
        vbox_preset = QVBoxLayout()
        vbox_preset.addWidget(QLabel("Model Paketi (Preset):", self))
        self.cmb_preset = QComboBox(self)
        self.cmb_preset.addItems([
            "HÄ±zlÄ± (XGBoost + LSTM)",
            "AÄŸaÃ§ TabanlÄ± Modeller",
            "YalnÄ±zca Derin Ã–ÄŸrenme",
            "Modern Baselines",
            "YalnÄ±zca Prophet",
            "TÃ¼m Modeller"
        ])
        self.cmb_preset.setFixedWidth(200)
        vbox_preset.addWidget(self.cmb_preset)
        config_layout.addLayout(vbox_preset)

        # 3. Paralel Worker SayÄ±sÄ±
        vbox_workers = QVBoxLayout()
        vbox_workers.addWidget(QLabel("EÅŸzamanlÄ± Ä°ÅŸÃ§iler (Workers):", self))
        self.sb_workers = QSpinBox(self)
        self.sb_workers.setRange(1, 16)
        self.sb_workers.setValue(2)
        self.sb_workers.setFixedWidth(80)
        vbox_workers.addWidget(self.sb_workers)
        config_layout.addLayout(vbox_workers)

        # 4. Eylemler (Butonlar)
        vbox_actions = QVBoxLayout()
        vbox_actions.addStretch()

        btn_action_layout = QHBoxLayout()
        self.btn_start = QPushButton("EÄŸitimi BaÅŸlat", self)
        self.btn_start.clicked.connect(self._start_training)

        self.btn_stop = QPushButton("Durdur", self)
        self.btn_stop.setObjectName("btn_secondary")  # Red styling
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)

        btn_action_layout.addWidget(self.btn_start)
        btn_action_layout.addWidget(self.btn_stop)
        vbox_actions.addLayout(btn_action_layout)
        config_layout.addLayout(vbox_actions)

        config_layout.addStretch()
        layout.addWidget(config_box)

        # Log Ä°zleyici Paneli (GroupBox)
        log_box = QGroupBox("CanlÄ± Ä°ÅŸlem LoglarÄ±", self)
        log_layout = QVBoxLayout(log_box)
        self.log_viewer = LogViewer(self)
        log_layout.addWidget(self.log_viewer)
        layout.addWidget(log_box)

        # Hisseleri yÃ¼kle
        self._load_stocks()

    def _load_stocks(self):
        """
        KullanÄ±labilir hisseleri yÃ¼kler ve combobox'a ekler.
        """
        self.cmb_stock.clear()
        self.cmb_stock.addItem("TÃœM HÄ°SSELER (Batch)")

        stocks = self.db.get_available_stocks()
        self.cmb_stock.addItems(stocks)

    def _start_training(self):
        """
        Model eÄŸitimi iÃ§in batch CLI aracÄ±nÄ± arka planda tetikler.
        """
        selected_stock = self.cmb_stock.currentText().strip()
        if not selected_stock:
            QMessageBox.warning(self, "UyarÄ±", "LÃ¼tfen geÃ§erli bir hisse senedi seÃ§in.")
            return

        # Preset eÅŸleÅŸtirme
        preset_idx = self.cmb_preset.currentIndex()
        # "HÄ±zlÄ± (XGBoost + LSTM)", "AÄŸaÃ§ TabanlÄ±", "Derin Ã–ÄŸrenme", "Baselines", "Prophet", "TÃ¼m"
        # batch.py parametreleri model seÃ§imi iÃ§in:
        # --disabled-models veya default candidate_models.
        # Bizim batch.py'da model preset argÃ¼manÄ± yoksa, modelleri direkt --disabled-models ile filtreleyebiliriz
        # veya interactive.py presets mantÄ±ÄŸÄ±na gÃ¶re argÃ¼man geÃ§ebiliriz.
        # batch.py'nin parametrelerini okuduÄŸumuzda: --stocks veya --universe parametreleri var.
        # AyrÄ±ca model kÄ±sÄ±tlamalarÄ± iÃ§in batch.py argÃ¼manlarÄ±nÄ± kontrol ettik, --disabled-models veya specific models seÃ§imi destekliyor.
        # KolaylÄ±k aÃ§Ä±sÄ±ndan ve batch.py yetenekleri Ã§erÃ§evesinde args listesi:
        args = []

        if selected_stock == "TÃœM HÄ°SSELER (Batch)":
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            universe_file = os.path.join(project_root, "data", "bist_universe.csv")
            if os.path.exists(universe_file):
                args.extend(["--universe", universe_file])
            else:
                QMessageBox.critical(self, "Hata", f"BIST universe dosyasÄ± bulunamadÄ±: {universe_file}")
                return
        else:
            args.extend(["--stocks", selected_stock])

        args.extend(["--mode", "walk_forward"])
        args.extend(["--workers", str(self.sb_workers.value())])

        # Model preset kÄ±sÄ±tlamalarÄ±nÄ± batch.py formatÄ±na uyarlama
        # batch.py iÃ§inde:
        # parser.add_argument("--disabled-models", default="")
        # parser.add_argument("--require-available", action="store_true")
        # batch.py specific model listesi yerine --disabled-models ile Ã§alÄ±ÅŸabiliyor.
        # interactive presets mantÄ±ÄŸÄ±na gÃ¶re disabled edilecek modelleri belirleyelim:
        all_models = ["XGBoost", "Random Forest", "LightGBM Return", "LSTM", "LSTM Lite", "AttentionLSTM v2", "Prophet", "Ridge Return", "ElasticNet Return", "DLinear", "NLinear"]
        disabled = []

        if preset_idx == 0:  # HÄ±zlÄ± (XGBoost + LSTM)
            disabled = [m for m in all_models if m not in ["XGBoost", "LSTM"]]
        elif preset_idx == 1:  # AÄŸaÃ§ TabanlÄ±
            disabled = [m for m in all_models if m not in ["XGBoost", "Random Forest", "LightGBM Return"]]
        elif preset_idx == 2:  # Derin Ã–ÄŸrenme
            disabled = [m for m in all_models if m not in ["LSTM", "LSTM Lite", "AttentionLSTM v2"]]
        elif preset_idx == 3:  # Modern Baselines
            disabled = [m for m in all_models if m not in ["Ridge Return", "ElasticNet Return", "LightGBM Return", "DLinear", "NLinear"]]
        elif preset_idx == 4:  # Prophet
            disabled = [m for m in all_models if m != "Prophet"]

        if disabled:
            args.extend(["--disabled-models", ",".join(disabled)])

        # GUI Durumunu GÃ¼ncelle (UX Koruma)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.cmb_stock.setEnabled(False)
        self.cmb_preset.setEnabled(False)
        self.sb_workers.setEnabled(False)

        self.log_viewer.clear()
        self.log_viewer.append_text("[GUI] Model eÄŸitim sÃ¼reci baÅŸlatÄ±lÄ±yor...\n")

        # Ã‡alÄ±ÅŸtÄ±r
        self.runner.start_process("src.cli.batch", args)

    def _stop_training(self):
        """
        Ã‡alÄ±ÅŸmakta olan eÄŸitimi zorla sonlandÄ±rÄ±r.
        """
        self.runner.kill_process()

    def _append_log(self, text: str):
        """
        Ä°ÅŸlem loglarÄ±nÄ± canlÄ± ekrana yazar.
        """
        self.log_viewer.append_text(text)

    def _training_finished(self, exit_code: int, message: str):
        """
        EÄŸitim bittiÄŸinde arayÃ¼z kontrollerini tekrar aktif eder.
        """
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.cmb_stock.setEnabled(True)
        self.cmb_preset.setEnabled(True)
        self.sb_workers.setEnabled(True)

        # KullanÄ±cÄ±ya iÅŸlem bitti mesajÄ± gÃ¶ster (UX)
        if exit_code == 0:
            QMessageBox.information(self, "Bilgi", "Model eÄŸitimi baÅŸarÄ±yla tamamlandÄ±!")
        else:
            QMessageBox.warning(self, "Hata / Ä°ptal", "EÄŸitim sÃ¼reci tamamlanamadÄ± veya durduruldu.")
