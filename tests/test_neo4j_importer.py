# tests/test_neo4j_importer.py
from legal_scraper.neo4j_importer import (
    build_article_uid,
    build_clause_uid,
    build_point_uid,
    get_constraint_statements,
    load_parsed_payload,
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


def test_constraint_statements_include_document_unique_key():
    statements = get_constraint_statements()
    assert any("Document" in s and "doc_identity" in s for s in statements)


def test_constraint_statements_cover_all_labels():
    labels = [
        "DocumentGroup", "DocumentType", "EffectStatus",
        "Organization", "Signer", "Field",
        "Part", "Chapter", "Section", "Article", "Clause", "Point",
    ]
    for label in labels:
        assert any(label in s for s in get_constraint_statements()), f"Missing constraint for {label}"


def test_load_parsed_payload_extracts_document_identity(tmp_path):
    path = tmp_path / "test.json"
    path.write_text('{"doc_stem":"test","nodes":{"document":{"doc_identity":"56/2024/QH15"}},"relationships":[]}', encoding="utf-8")
    payload = load_parsed_payload(path)
    assert payload["nodes"]["document"]["doc_identity"] == "56/2024/QH15"

