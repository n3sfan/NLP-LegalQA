# tests/test_neo4j_importer.py
from legal_scraper.neo4j_importer import (
    build_article_uid,
    build_clause_uid,
    build_point_uid,
)


def test_build_article_uid_is_document_scoped():
    uid = build_article_uid("56/2024/QH15", "1")
    assert uid == "56/2024/QH15::article::1"


def test_build_clause_uid_uses_article_context():
    assert build_clause_uid("56/2024/QH15", "1", "3") == "56/2024/QH15::article::1::clause::3"


def test_build_point_uid_uses_full_parent_path():
    assert (
        build_point_uid("56/2024/QH15", "1", "3", "a")
        == "56/2024/QH15::article::1::clause::3::point::a"
    )
