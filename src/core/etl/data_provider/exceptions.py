class InstrumentNotFoundError(Exception):
    """Raised when the provider rejects a ticker/instrument identifier."""

    def __init__(self, symbol: str, message: str = "Instrument not found"):
        self.symbol = symbol
        super().__init__(f"{symbol}: {message}")
