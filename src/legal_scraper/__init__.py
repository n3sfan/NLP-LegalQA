import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legal_scraper.client import LegalDocumentClient
from legal_scraper.parser import LegalDocumentParser
from legal_scraper.scraper import LegalDocumentScraper

__all__ = ["LegalDocumentClient", "LegalDocumentParser", "LegalDocumentScraper"]
