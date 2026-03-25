# tests/test_neo4j_importer.py
from legal_scraper.neo4j_importer import build_article_uid


def test_build_article_uid_is_document_scoped():
    uid = build_article_uid("56/2024/QH15", "1")
    assert uid == "56/2024/QH15::article::1"
