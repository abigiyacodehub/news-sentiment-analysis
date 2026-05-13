from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
NEWS_ZIP_CANDIDATES = [
    DATA_DIR / "newsData.zip",
    PROJECT_ROOT.parent / "newsData.zip",
    Path("C:/Users/Administrator/Documents/Kifiya AI Mastery Training/newsData.zip"),
]
STOCK_PRICE_DIR = DATA_DIR / "stock_prices"
