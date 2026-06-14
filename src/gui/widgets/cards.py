# -*- coding: utf-8 -*-
"""
src.gui.widgets.cards - Corporate metric and information cards.
"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class MetricCard(QFrame):
    """
    Kurumsal stilize edilmiÅŸ metrik gÃ¶sterim kartÄ±.
    """
    def __init__(self, title: str, value: str, desc: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("status_card")
        self.init_ui(title, value, desc)

    def init_ui(self, title: str, value: str, desc: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        self.lbl_title = QLabel(title, self)
        self.lbl_title.setObjectName("card_title")
        self.lbl_title.setAlignment(Qt.AlignLeft)

        self.lbl_value = QLabel(value, self)
        self.lbl_value.setObjectName("card_value")
        self.lbl_value.setAlignment(Qt.AlignLeft)

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_value)

        if desc:
            self.lbl_desc = QLabel(desc, self)
            self.lbl_desc.setObjectName("card_desc")
            self.lbl_desc.setAlignment(Qt.AlignLeft)
            self.lbl_desc.setWordWrap(True)
            layout.addWidget(self.lbl_desc)

    def update_value(self, value: str, desc: str = None):
        """
        Kart deÄŸerini dinamik gÃ¼nceller.
        """
        self.lbl_value.setText(value)
        if desc and hasattr(self, 'lbl_desc'):
            self.lbl_desc.setText(desc)


class TrendCard(MetricCard):
    """
    YÃ¶n eÄŸilimine gÃ¶re (YukarÄ±, AÅŸaÄŸÄ±, Yatay) renklenen Ã¶zel kart.
    """
    def __init__(self, title: str, value: str, trend: str, desc: str = "", parent=None):
        super().__init__(title, value, desc, parent)
        self.update_trend_style(trend)

    def update_trend_style(self, trend: str):
        """
        Trend yÃ¶nÃ¼ne gÃ¶re metin stilini ayarlar.
        """
        trend = trend.lower()
        if "yukarÄ±" in trend or "up" in trend:
            self.lbl_value.setStyleSheet("color: #22c55e;") # Emerald Green
        elif "aÅŸaÄŸÄ±" in trend or "down" in trend:
            self.lbl_value.setStyleSheet("color: #ef4444;") # Red
        elif "yatay" in trend or "sideways" in trend:
            self.lbl_value.setStyleSheet("color: #eab308;") # Yellow
        else:
            self.lbl_value.setStyleSheet("color: #94a3b8;") # Gray
