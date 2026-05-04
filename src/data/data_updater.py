# -*- coding: utf-8 -*-
"""
data_updater.py — Akıllı Veri Güncelleme Servisi
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hedef CSV dosyasını okuyarak son iş gününden bu yana geçen süreyi kontrol eder.
Eğer veri geride kalmışsa, kullanıcı onayını takiben yfinance üzerinden
eksik günleri indirir, sütun ve tarih mantığını projeye uyarlayarak
orijinal CSV'yi günceller.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime


class DataUpdater:
    """Veri seti güncelliğini denetleyen ve tamamlayan sınıf."""

    @staticmethod
    def check_and_update(csv_path: str, stock_symbol: str) -> None:
        """
        Kullanıcıya veri setinin güncelliğini bildirir ve geride kalmışsa yfinance
        üzerinden eksik günleri tamamlamak isteyip istemediğini sorar.
        """
        try:
            df_raw = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"  [UYARI] {csv_path} dosyası bulunamadı. Lütfen kontrol edin.")
            return

        # Sütun adı TR veya EN olabilir
        date_col = "Tarih" if "Tarih" in df_raw.columns else "Date"
        last_date_str = str(df_raw[date_col].iloc[-1])
        
        # Ham verideki son tarihi parse et
        try:
            last_date = pd.to_datetime(last_date_str, format="mixed", dayfirst=True)
        except ValueError:
            last_date = pd.to_datetime(last_date_str, dayfirst=True)
        today = pd.to_datetime(datetime.today().date())

        diff_days = (today - last_date).days

        if diff_days > 1:
            print(f"\n  [INFO] Veri setinizdeki son tarih : {last_date.strftime('%d/%m/%Y')}")
            print(f"  [INFO] Günümüz tarihi           : {today.strftime('%d/%m/%Y')}")
            ans = input(f"  [SORU] Veri setiniz günümüzden {diff_days} gün kadar geride. \n  Eksik veriler yfinance ({stock_symbol}.IS) üzerinden tamamlansın mı? (e/h): ")
            
            if ans.lower() == 'e':
                print("  [INFO] yfinance'a bağlanılıyor, yeni veriler indiriliyor...")
                ticker = f"{stock_symbol}.IS"
                
                # Son tarihten bugüne (bugün dahil olması için +1 gün) kadar veri çekilecek
                start_str = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                end_str = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                
                new_data = yf.download(ticker, start=start_str, end=end_str, progress=False)
                
                if not new_data.empty:
                    # MultiIndex sorunu varsa temizle
                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_data.columns = new_data.columns.droplevel(1)
                    
                    new_data.reset_index(inplace=True)
                    
                    # TR sütun eşleştirmesi (genel senaryo)
                    inv_map = {
                        "Date": "Tarih",
                        "Open": "Açılış",
                        "High": "Yüksek",
                        "Low": "Düşük",
                        "Close": "Kapanış",
                        "Adj Close": "Düzeltilmiş_Kapanış",
                        "Volume": "Hacim"
                    }
                    new_data.rename(columns=inv_map, inplace=True)
                    
                    # Eğer okunan orijinal CSV "EN" formatındaysa (mesela 'Date' yerine) düzelt:
                    if date_col == "Date":
                        new_data.rename(columns={"Tarih": "Date", "Düzeltilmiş_Kapanış": "Adj_Close"}, inplace=True)
                        # Ek olarak diğerleri de EN ise onları da geri çevirmek gerekir
                        # (Ancak orijinal CSV yapısına göre dinamik bir map sağlanabilir)
                    
                    # Orijinal dosyada hangi kolonlar varsa sadece onları koru
                    common_cols = [c for c in df_raw.columns if c in new_data.columns]
                    new_data = new_data[common_cols]
                    
                    # Orijinal dosyanın tarih formatına tam uyumlu kaydet ki loader patlamasın
                    if "-" in last_date_str:
                        if len(last_date_str.split("-")[0]) == 4:
                            fmt = "%Y-%m-%d"
                        else:
                            fmt = "%d-%m-%Y"
                    elif "/" in last_date_str:
                        if len(last_date_str.split("/")[0]) == 4:
                            fmt = "%Y/%m/%d"
                        else:
                            fmt = "%d/%m/%Y"
                    else:
                        fmt = "%d.%m.%Y"
                    
                    new_data[date_col] = new_data[date_col].dt.strftime(fmt)
                    
                    # CSV'ye append modunda kaydet (header yazma)
                    new_data.to_csv(csv_path, mode='a', header=False, index=False)
                    print(f"  [OK] {len(new_data)} iş günü değerindeki yeni veri başarıyla {os.path.basename(csv_path)} dosyasına yazıldı.")
                else:
                    print("  [INFO] Girdiğiniz aralıkta finansal veri bulunamadı (Tatiller vb. olabilir).")
            else:
                print("  [INFO] Veri güncellemesi atlandı. Mevcut eski veriler kullanılıyor.")
        else:
            print("  [INFO] CSV veri setiniz halihazırda son güne ait. Güncellemeye gerek yok.")
