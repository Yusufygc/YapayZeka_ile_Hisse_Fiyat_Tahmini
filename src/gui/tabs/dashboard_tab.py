# -*- coding: utf-8 -*-
"""
src.gui.tabs.dashboard_tab - Gecelik havuz (PeerStore) son kosu ozeti.

Siralanabilir tablo: sembol, sektor, akran etiketi/yuzdeligi, trend, risk seviyesi
ve fiyat bandi. Veri `data/serving_pool.db` son kosusundan okunur (salt-okunur).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from src.gui.core import PeerReader, PeerSnapshot
from src.gui.widgets import MetricCard

_DASH = "—"

# (baslik, alan, bicimleyici)
_Fmt = Callable[[Any], str]


def _num(decimals: int) -> _Fmt:
    return lambda v: f"{float(v):.{decimals}f}"


def _text(v: Any) -> str:
    return str(v)


COLUMNS: List[Tuple[str, str, _Fmt]] = [
    ("Hisse", "symbol", _text),
    ("Sektör", "segment_sector", _text),
    ("Akran Etiketi", "peer_label", _text),
    ("Akran Yüzdeliği", "peer_percentile", _num(1)),
    ("Trend", "trend_label", _text),
    ("Akranı Geçme Olasılığı", "trend_prob_outperform", _num(3)),
    ("Risk Seviyesi", "risk_level", _text),
    ("Fiyat Bandı Alt", "kolb_price_low", _num(2)),
    ("Fiyat Bandı Üst", "kolb_price_high", _num(2)),
]


class DashboardTab(QWidget):
    """Son gecelik kosunun sembol bazli ozet tablosu."""

    def __init__(self, reader: PeerReader, parent=None):
        super().__init__(parent)
        self.reader = reader
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        cards = QHBoxLayout()
        self.card_asof = MetricCard("Veri Tarihi (as_of)", _DASH, " ")
        self.card_run = MetricCard("Havuz Koşusu", _DASH, " ")
        self.card_nightly = MetricCard("Gece İşi Durumu", _DASH, " ")
        for card in (self.card_asof, self.card_run, self.card_nightly):
            cards.addWidget(card)
        layout.addLayout(cards)

        top = QHBoxLayout()
        title = QLabel("Gecelik Havuz — Son Koşu (Kol-B + Risk)", self)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.txt_search = QLineEdit(self)
        self.txt_search.setPlaceholderText("Hisse ara... (örn: TUPRS)")
        self.txt_search.setFixedWidth(200)
        self.txt_search.textChanged.connect(self._handle_search)
        self.btn_refresh = QPushButton("Verileri Yenile", self)
        self.btn_refresh.setFixedWidth(120)
        self.btn_refresh.clicked.connect(self.load_data)
        top.addWidget(title)
        top.addStretch()
        top.addWidget(self.txt_search)
        top.addWidget(self.btn_refresh)
        layout.addLayout(top)

        self.lbl_info = QLabel("", self)
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        self.table_view = QTableView(self)
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        header = self.table_view.horizontalHeader()
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(80)
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.table_view.verticalHeader().setVisible(False)

        self.model = QStandardItemModel(self)
        self.proxy_model = QSortFilterProxyModel(self)
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setSortRole(Qt.UserRole)
        self.proxy_model.setFilterKeyColumn(0)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.table_view.setModel(self.proxy_model)
        layout.addWidget(self.table_view)

        self.load_data()

    # ------------------------------------------------------------------ data
    def load_data(self) -> None:
        snap = self.reader.load()
        self._fill_cards(snap)
        self._fill_table(snap.rows)
        self.lbl_info.setText(snap.error or "")

    def _fill_cards(self, snap: PeerSnapshot) -> None:
        run = snap.run or {}
        self.card_asof.update_value(str(run.get("as_of_date") or _DASH)[:10])
        if run:
            self.card_run.update_value(
                f"#{run.get('run_id')}",
                f"{run.get('model_name') or ''} · {run.get('n_symbols') or 0} sembol",
            )
        else:
            self.card_run.update_value(_DASH, "")
        self.card_nightly.update_value(*self._nightly_text(snap))

    @staticmethod
    def _nightly_text(snap: PeerSnapshot) -> Tuple[str, str]:
        if snap.nightly_error:
            return "Okunamadı", snap.nightly_error
        st = snap.nightly
        if not st:
            return _DASH, "nightly_status.json yok"
        if st.get("completed") and st.get("exit_code") == 0:
            value = "Başarılı"
        elif st.get("completed"):
            value = f"Çıkış kodu {st.get('exit_code')}"
        elif st.get("phase") == "crashed":
            value = "Çöktü"
        else:
            value = "Yarım / çalışıyor"
        return value, f"faz: {st.get('phase')} · bitiş: {st.get('finished_at') or _DASH}"

    def _fill_table(self, rows: List[Dict[str, Any]]) -> None:
        self.model.clear()
        self.model.setHorizontalHeaderLabels([c[0] for c in COLUMNS])
        for rec in rows:
            items = [self._make_item(rec.get(field), fmt) for _, field, fmt in COLUMNS]
            self.model.appendRow(items)
        self.table_view.resizeColumnsToContents()

    @staticmethod
    def _make_item(value: Any, fmt: _Fmt) -> QStandardItem:
        if value is None:
            item = QStandardItem(_DASH)
            item.setData(float("-inf"), Qt.UserRole)
        else:
            item = QStandardItem(fmt(value))
            item.setData(float(value) if isinstance(value, (int, float)) else str(value), Qt.UserRole)
        item.setTextAlignment(Qt.AlignCenter)
        item.setEditable(False)
        return item

    def _handle_search(self, text: str) -> None:
        self.proxy_model.setFilterFixedString(text)

    def displayed_symbols(self) -> List[str]:
        """Filtre/siralama sonrasi gorunen sembol sirasi (test icin)."""
        return [
            self.proxy_model.index(r, 0).data() for r in range(self.proxy_model.rowCount())
        ]
