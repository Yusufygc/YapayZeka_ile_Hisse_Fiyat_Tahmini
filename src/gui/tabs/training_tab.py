# -*- coding: utf-8 -*-
"""
src.gui.tabs.training_tab - Triggers and monitors batch training runs.
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSpinBox, QPushButton, QGroupBox, QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt
from src.gui.core import DBHelper, ProcessRunner
from src.gui.widgets import LogViewer


class TrainingTab(QWidget):
    """
    Model eğitimi yapılandırma ve başlatma paneli.
    """
    def __init__(self, db_helper: DBHelper, runner: ProcessRunner, parent=None):
        super().__init__(parent)
        self.db = db_helper
        self.runner = runner

        # Sinyal bağlantıları
        self.runner.output_received.connect(self._append_log)
        self.runner.finished.connect(self._training_finished)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Yapılandırma paneli (GroupBox)
        config_box = QGroupBox("Eğitim Yapılandırması", self)
        config_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        config_box.setMaximumHeight(150)
        config_layout = QHBoxLayout(config_box)
        config_layout.setSpacing(20)
        config_layout.setContentsMargins(12, 16, 12, 12)

        # 1. Hisse seçimi
        vbox_stock = QVBoxLayout()
        vbox_stock.addWidget(QLabel("Hisse Senedi:", self))
        self.cmb_stock = QComboBox(self)
        self.cmb_stock.setEditable(True)  # Kolay arama desteği
        self.cmb_stock.setPlaceholderText("Hisse seçin veya arayın...")
        self.cmb_stock.setFixedWidth(150)
        vbox_stock.addWidget(self.cmb_stock)
        config_layout.addLayout(vbox_stock)

        # 2. Model seçim presetleri
        vbox_preset = QVBoxLayout()
        vbox_preset.addWidget(QLabel("Model Paketi (Preset):", self))
        self.cmb_preset = QComboBox(self)
        self.cmb_preset.addItems([
            "Hızlı (XGBoost)",
            "Ağaç Tabanlı Modeller",
            "Lineer Baselines",
            "Modern Baselines",
            "Tüm Modeller"
        ])
        self.cmb_preset.setFixedWidth(200)
        vbox_preset.addWidget(self.cmb_preset)
        config_layout.addLayout(vbox_preset)

        # 3. Paralel worker sayısı
        vbox_workers = QVBoxLayout()
        vbox_workers.addWidget(QLabel("Eşzamanlı İşçiler (Workers):", self))
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
        self.btn_start = QPushButton("Eğitimi Başlat", self)
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

        layout.setAlignment(config_box, Qt.AlignTop)

        # Log izleyici paneli (GroupBox)
        log_box = QGroupBox("Canlı İşlem Logları", self)
        log_layout = QVBoxLayout(log_box)
        self.log_viewer = LogViewer(self)
        log_layout.addWidget(self.log_viewer)
        layout.addWidget(log_box)

        # Hisseleri yükle
        self._load_stocks()

    def _load_stocks(self):
        """
        Kullanılabilir hisseleri yükler ve combobox'a ekler.
        """
        self.cmb_stock.clear()
        self.cmb_stock.addItem("TÜM HİSSELER (Batch)")

        stocks = self.db.get_available_stocks()
        self.cmb_stock.addItems(stocks)

    def _start_training(self):
        """
        Model eğitimi için batch CLI aracını arka planda tetikler.
        """
        selected_stock = self.cmb_stock.currentText().strip()
        if not selected_stock:
            QMessageBox.warning(self, "Uyarı", "Lütfen geçerli bir hisse senedi seçin.")
            return

        # Preset eşleştirme
        preset_idx = self.cmb_preset.currentIndex()
        # "Hızlı (XGBoost)", "Ağaç Tabanlı", "Lineer Baselines", "Modern Baselines", "Tüm"
        # batch.py parametreleri model seçimi için:
        # --disabled-models veya default candidate_models.
        # Bizim batch.py'da model preset argümanı yoksa, modelleri direkt --disabled-models ile filtreleyebiliriz
        # veya interactive.py presets mantığına göre argüman geçebiliriz.
        # batch.py'nin parametrelerini okuduğumuzda: --stocks veya --universe parametreleri var.
        # Ayrıca model kısıtlamaları için batch.py argümanlarını kontrol ettik, --disabled-models veya specific models seçimi destekliyor.
        # Kolaylık açısından ve batch.py yetenekleri çerçevesinde args listesi:
        args = []

        if selected_stock == "TÜM HİSSELER (Batch)":
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            universe_file = os.path.join(project_root, "data", "bist_universe.csv")
            if os.path.exists(universe_file):
                args.extend(["--universe", universe_file])
            else:
                QMessageBox.critical(self, "Hata", f"BIST universe dosyası bulunamadı: {universe_file}")
                return
        else:
            args.extend(["--stocks", selected_stock])

        args.extend(["--mode", "walk_forward"])
        args.extend(["--workers", str(self.sb_workers.value())])

        # Model preset kısıtlamalarını batch.py formatına uyarlama
        # batch.py içinde:
        # parser.add_argument("--disabled-models", default="")
        # parser.add_argument("--require-available", action="store_true")
        # batch.py specific model listesi yerine --disabled-models ile çalışabiliyor.
        # interactive presets mantığına göre disabled edilecek modelleri belirleyelim:
        all_models = ["XGBoost", "LightGBM Return", "Ridge Return"]
        disabled = []

        if preset_idx == 0:  # Hızlı (XGBoost)
            disabled = [m for m in all_models if m not in ["XGBoost"]]
        elif preset_idx == 1:  # Ağaç Tabanlı
            disabled = [m for m in all_models if m not in ["XGBoost", "LightGBM Return"]]
        elif preset_idx == 2:  # Lineer Baselines
            disabled = [m for m in all_models if m not in ["Ridge Return"]]
        elif preset_idx == 3:  # Modern Baselines
            disabled = [m for m in all_models if m not in ["Ridge Return", "LightGBM Return"]]

        if disabled:
            args.extend(["--disabled-models", ",".join(disabled)])

        # GUI durumunu güncelle (UX koruma)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.cmb_stock.setEnabled(False)
        self.cmb_preset.setEnabled(False)
        self.sb_workers.setEnabled(False)

        self.log_viewer.clear()
        self.log_viewer.append_text("[GUI] Model eğitim süreci başlatılıyor...\n")

        # Çalıştır
        self.runner.start_process("src.cli.batch", args)

    def _stop_training(self):
        """
        Çalışmakta olan eğitimi zorla sonlandırır.
        """
        self.runner.kill_process()

    def _append_log(self, text: str):
        """
        İşlem loglarını canlı ekrana yazar.
        """
        self.log_viewer.append_text(text)

    def _training_finished(self, exit_code: int, message: str):
        """
        Eğitim bittiğinde arayüz kontrollerini tekrar aktif eder.
        """
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.cmb_stock.setEnabled(True)
        self.cmb_preset.setEnabled(True)
        self.sb_workers.setEnabled(True)

        # Kullanıcıya işlem bitti mesajı göster (UX)
        if exit_code == 0:
            QMessageBox.information(self, "Bilgi", "Model eğitimi başarıyla tamamlandı!")
        else:
            QMessageBox.warning(self, "Hata / İptal", "Eğitim süreci tamamlanamadı veya durduruldu.")
