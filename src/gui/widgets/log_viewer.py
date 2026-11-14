# -*- coding: utf-8 -*-
"""
src.gui.widgets.log_viewer - Customized log viewer widget with autoscroll.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QHBoxLayout, QPushButton
from PySide6.QtGui import QTextCursor


class LogViewer(QWidget):
    """
    Subprocess çıktısını anlık gösteren ve otomatik aşağı kaydıran log bileşeni.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Log ekranı
        self.text_edit = QPlainTextEdit(self)
        self.text_edit.setObjectName("log_viewer")
        self.text_edit.setReadOnly(True)
        self.text_edit.setMaximumBlockCount(5000)  # Hafıza dostu limit

        # Kontrol butonları
        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("Logu Temizle", self)
        self.btn_clear.setMaximumWidth(120)
        self.btn_clear.clicked.connect(self.clear)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_clear)

        layout.addWidget(self.text_edit)
        layout.addLayout(btn_layout)

    def append_text(self, text: str):
        """
        Log ekranına yeni metin ekler ve otomatik aşağı kaydırır.
        """
        # ANSI escape kodlarını temizleme
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_text = ansi_escape.sub('', text)
        clean_text = clean_text.replace("\r\n", "\n").replace("\r", "\n").replace("\b", "")

        # Scroll pozisyonunu kontrol et
        vertical_bar = self.text_edit.verticalScrollBar()
        was_at_bottom = vertical_bar.value() == vertical_bar.maximum()

        self.text_edit.insertPlainText(clean_text)

        if was_at_bottom:
            # En alta kaydır
            self.text_edit.moveCursor(QTextCursor.End)
            vertical_bar.setValue(vertical_bar.maximum())

    def clear(self):
        """
        Ekranı temizler.
        """
        self.text_edit.clear()
