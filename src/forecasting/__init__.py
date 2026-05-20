"""Forward forecast services for production BIST predictions."""

__all__ = ["BistMarketRules", "ForecastRunner"]


def __getattr__(name):
    if name == "BistMarketRules":
        from src.forecasting.bist_rules import BistMarketRules

        return BistMarketRules
    if name == "ForecastRunner":
        from src.forecasting.runner import ForecastRunner

        return ForecastRunner
    raise AttributeError(name)
