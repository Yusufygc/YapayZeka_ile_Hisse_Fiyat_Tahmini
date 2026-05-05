# -*- coding: utf-8 -*-
"""
report_writer.py - Persist XAI outputs as CSV, Markdown and PNG.
"""

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.reporting_utils import route_output_path, with_output_extension, write_csv_and_aligned_view


class XAIReportWriter:
    def __init__(
        self,
        output_dir: str,
        *,
        write_tables: bool = False,
        write_markdown: bool = True,
    ):
        self.output_dir = output_dir
        self.write_tables = write_tables
        self.write_markdown = write_markdown
        os.makedirs(self.output_dir, exist_ok=True)

    def write(self, payload: Dict[str, pd.DataFrame | str], suffix: str = "latest") -> None:
        top_reasons = payload.get("top_reasons")
        daily_reasons = payload.get("daily_reasons")
        signal_reasons = payload.get("signal_reasons")
        trade_explanations = payload.get("trade_explanations")
        summary_md = str(payload.get("summary_md", ""))

        if isinstance(top_reasons, pd.DataFrame) and not top_reasons.empty:
            if self.write_tables:
                write_csv_and_aligned_view(
                    top_reasons,
                    os.path.join(self.output_dir, f"xai_top_reasons_{suffix}.csv"),
                )
            group_importance = self._group_importance(top_reasons)
            if not group_importance.empty:
                if self.write_tables:
                    write_csv_and_aligned_view(
                        group_importance,
                        os.path.join(self.output_dir, f"xai_group_importance_{suffix}.csv"),
                    )
                self._plot_group_importance(
                    group_importance,
                    os.path.join(self.output_dir, f"xai_group_importance_{suffix}.png"),
                )
            self._plot_importance(top_reasons, os.path.join(self.output_dir, f"xai_feature_importance_{suffix}.png"))

        if isinstance(daily_reasons, pd.DataFrame) and not daily_reasons.empty:
            if self.write_tables:
                write_csv_and_aligned_view(
                    daily_reasons,
                    os.path.join(self.output_dir, f"xai_daily_reasons_{suffix}.csv"),
                )

        if isinstance(signal_reasons, pd.DataFrame) and not signal_reasons.empty:
            if self.write_tables:
                write_csv_and_aligned_view(
                    signal_reasons,
                    os.path.join(self.output_dir, f"xai_signal_reasons_{suffix}.csv"),
                )
            self._plot_signal_timeline(signal_reasons, os.path.join(self.output_dir, f"xai_signal_timeline_{suffix}.png"))
            self._plot_threshold_diagnostics(signal_reasons, os.path.join(self.output_dir, f"xai_threshold_diagnostics_{suffix}.png"))

        if isinstance(trade_explanations, pd.DataFrame) and not trade_explanations.empty:
            if self.write_tables:
                write_csv_and_aligned_view(
                    trade_explanations,
                    os.path.join(self.output_dir, f"xai_trade_explanations_{suffix}.csv"),
                )

        # ── [A5] TFT attention heatmap ────────────────────────────────────────
        tft_attention_data = payload.get("tft_attention_data")
        if isinstance(tft_attention_data, dict):
            for model_tag, attn_arr in tft_attention_data.items():
                safe_tag = str(model_tag).replace(" ", "_")
                self._plot_tft_attention_heatmap(
                    attn_arr,
                    model_tag,
                    os.path.join(
                        self.output_dir,
                        f"xai_tft_attention_{safe_tag}_{suffix}.png",
                    ),
                )

        if self.write_markdown:
            summary_path = with_output_extension(os.path.join(self.output_dir, f"xai_summary_{suffix}.md"), ".md")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as handle:
                handle.write(summary_md.strip())
                handle.write("\n")

        print(f"[OK] XAI raporlari kaydedildi -> {self.output_dir}")

    @staticmethod
    def _group_importance(top_reasons: pd.DataFrame) -> pd.DataFrame:
        required = {"Model", "Feature_Group", "Importance"}
        if top_reasons.empty or not required.issubset(top_reasons.columns):
            return pd.DataFrame()
        grouped = (
            top_reasons.groupby(["Model", "Feature_Group"], as_index=False)["Importance"]
            .sum()
            .sort_values(["Model", "Importance"], ascending=[True, False])
        )
        totals = grouped.groupby("Model")["Importance"].transform("sum").replace(0, 1.0)
        grouped["Group_Importance_Share"] = grouped["Importance"] / totals
        return grouped

    def _plot_group_importance(self, group_importance: pd.DataFrame, save_path: str) -> None:
        save_path = route_output_path(save_path)
        if group_importance.empty:
            return
        plot_df = (
            group_importance.groupby("Feature_Group", as_index=False)["Importance"]
            .mean()
            .sort_values("Importance", ascending=False)
        )
        plt.figure(figsize=(9, 4.8))
        plt.bar(plot_df["Feature_Group"], plot_df["Importance"], color="#0f766e")
        plt.xlabel("Feature group")
        plt.ylabel("Ortalama etki skoru")
        plt.title("Feature group bazinda XAI onemi")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close()

    def _plot_importance(self, top_reasons: pd.DataFrame, save_path: str) -> None:
        save_path = route_output_path(save_path)
        plot_df = (
            top_reasons.groupby(["Feature", "Readable_Feature"], as_index=False)["Importance"]
            .mean()
            .sort_values("Importance", ascending=False)
            .head(12)
        )
        if plot_df.empty:
            return

        labels = plot_df["Readable_Feature"].str.slice(0, 56)
        fig_height = max(4.5, len(plot_df) * 0.45)
        plt.figure(figsize=(11, fig_height))
        plt.barh(labels, plot_df["Importance"], color="#2563eb")
        plt.gca().invert_yaxis()
        plt.xlabel("Ortalama etki skoru")
        plt.title("Model tahmininde en etkili sinyaller")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close()

    def _plot_signal_timeline(self, signal_reasons: pd.DataFrame, save_path: str) -> None:
        save_path = route_output_path(save_path)
        plot_df = signal_reasons.copy()
        if plot_df.empty or "Decision" not in plot_df.columns:
            return

        decision_map = {"NO_TRADE": 0, "EXIT": 1, "HOLD": 2, "BUY": 3}
        plot_df["Decision_Code"] = plot_df["Decision"].map(decision_map).fillna(0)

        plt.figure(figsize=(14, 6))
        for model_name, group in plot_df.groupby("Model"):
            plt.plot(range(len(group)), group["Decision_Code"], marker="o", linewidth=1.2, label=model_name)
        plt.yticks([0, 1, 2, 3], ["NO_TRADE", "EXIT", "HOLD", "BUY"])
        plt.xlabel("Karar sirasi")
        plt.ylabel("Sinyal karari")
        plt.title("Walk-forward sinyal karar zaman cizgisi")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close()

    def _plot_threshold_diagnostics(self, signal_reasons: pd.DataFrame, save_path: str) -> None:
        save_path = route_output_path(save_path)
        needed = {"Expected_Return", "Entry_Threshold", "Exit_Threshold"}
        if signal_reasons.empty or not needed.issubset(signal_reasons.columns):
            return

        plt.figure(figsize=(14, 6))
        first_model = str(signal_reasons["Model"].iloc[0]) if "Model" in signal_reasons.columns else "Model"
        group = signal_reasons[signal_reasons["Model"] == first_model].copy() if "Model" in signal_reasons.columns else signal_reasons.copy()
        x = range(len(group))
        plt.plot(x, group["Expected_Return"], label="Expected Return", linewidth=1.4)
        plt.plot(x, group["Entry_Threshold"], label="Entry Threshold", linewidth=1.2)
        plt.plot(x, group["Exit_Threshold"], label="Exit Threshold", linewidth=1.2)
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        plt.xlabel("Karar sirasi")
        plt.ylabel("Getiri / esik")
        plt.title(f"{first_model} sinyal esigi diagnostigi")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close()

    def _plot_tft_attention_heatmap(
        self,
        attn_arr,
        model_tag: str,
        save_path: str,
    ) -> None:
        """
        [A5] TFT capraz dikkat isi haritasini PNG olarak kaydeder.

        attn_arr : (N, H, T) dikkat agirliklari numpy dizisi.
        N ornekler uzerinde ortalaması alınarak (H, T) matris imshow ile gorsellestirilir.

        matplotlib.image.imsave kullanilir — Figure/Axes/backend olusturmaz,
        dogrudan PIL uzerinden PNG yazar. Bu yaklasim Windows'ta
        FigureCanvasAgg + tight_layout kombinasyonunun tetikledigi
        "maximum recursion depth exceeded" hatasini onler.
        """
        save_path = route_output_path(save_path)
        try:
            import matplotlib.cm as _cm
            import matplotlib.image as _mplimg

            arr = np.asarray(attn_arr, dtype=float)
            if arr.ndim != 3 or arr.size == 0:
                return
            mean_map = arr.mean(axis=0)   # (H, T)

            # [0, 1] normalizasyonu — Blues colormap icin
            lo, hi = float(mean_map.min()), float(mean_map.max())
            norm_map = (mean_map - lo) / (hi - lo + 1e-8)

            # Blues colormap → RGBA float32 (H, T, 4)
            rgba = _cm.Blues(norm_map)

            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            # mplimg.imsave PIL'e devreder — renderer/backend gerekmez
            _mplimg.imsave(save_path, rgba)
            print("[OK] TFT dikkat isi haritasi kaydedildi -> " + save_path)
        except Exception as exc:
            print("[WARN] TFT dikkat isi haritasi olusturulamadi: " + str(exc))
