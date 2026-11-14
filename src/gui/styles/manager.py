# -*- coding: utf-8 -*-
"""
src.gui.styles.manager - Theme and QSS loader.
Modüler QSS dosyalarını okur, birleştirir ve uygulamaya sunar.
"""

import os

STYLE_DIR = os.path.dirname(os.path.abspath(__file__))
QSS_FILES = ["base.qss", "widgets.qss", "custom.qss"]


def get_combined_stylesheet() -> str:
    """
    Tüm modüler QSS dosyalarını sırasıyla okur ve tek bir stylesheet string'i döndürür.
    """
    combined = []
    for filename in QSS_FILES:
        filepath = os.path.join(STYLE_DIR, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    combined.append(f.read())
            except Exception as e:
                print(f"[GUI Style Manager] Hata: {filename} okunamadı: {e}")
        else:
            print(f"[GUI Style Manager] Uyarı: Stil dosyası bulunamadı: {filepath}")

    return "\n\n".join(combined)
