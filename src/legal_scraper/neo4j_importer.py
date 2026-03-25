"""Neo4j importer for parsed Vietnamese legal documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


# ── UID builders ──────────────────────────────────────────────────────────────


def build_part_uid(doc_identity: str, number: str) -> str:
    return f"{doc_identity}::part::{number}"


def build_chapter_uid(doc_identity: str, number: str) -> str:
    return f"{doc_identity}::chapter::{number}"


def build_section_uid(doc_identity: str, number: str) -> str:
    return f"{doc_identity}::section::{number}"


def build_article_uid(doc_identity: str, number: str) -> str:
    return f"{doc_identity}::article::{number}"


def build_clause_uid(doc_identity: str, parent_article: str, number: str) -> str:
    return f"{doc_identity}::article::{parent_article}::clause::{number}"


def build_point_uid(
    doc_identity: str, parent_article: str, parent_clause: str, letter: str
) -> str:
    return (
        f"{doc_identity}::article::{parent_article}"
        f"::clause::{parent_clause}::point::{letter}"
    )


# ── Payload helper ────────────────────────────────────────────────────────────


def load_parsed_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Constraint DDL ───────────────────────────────────────────────────────────


def get_constraint_statements() -> list[str]:
    return [
        (
            "CREATE CONSTRAINT document_doc_identity IF NOT EXISTS "
            "FOR (n:Document) REQUIRE n.doc_identity IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT document_group_id IF NOT EXISTS "
            "FOR (n:DocumentGroup) REQUIRE n.id IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT document_type_id IF NOT EXISTS "
            "FOR (n:DocumentType) REQUIRE n.id IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT effect_status_id IF NOT EXISTS "
            "FOR (n:EffectStatus) REQUIRE n.id IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT organization_id IF NOT EXISTS "
            "FOR (n:Organization) REQUIRE n.id IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT signer_id IF NOT EXISTS "
            "FOR (n:Signer) REQUIRE n.id IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT field_id IF NOT EXISTS "
            "FOR (n:Field) REQUIRE n.id IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT part_uid IF NOT EXISTS "
            "FOR (n:Part) REQUIRE n.uid IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT chapter_uid IF NOT EXISTS "
            "FOR (n:Chapter) REQUIRE n.uid IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT section_uid IF NOT EXISTS "
            "FOR (n:Section) REQUIRE n.uid IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT article_uid IF NOT EXISTS "
            "FOR (n:Article) REQUIRE n.uid IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT clause_uid IF NOT EXISTS "
            "FOR (n:Clause) REQUIRE n.uid IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT point_uid IF NOT EXISTS "
            "FOR (n:Point) REQUIRE n.uid IS UNIQUE"
        ),
    ]


# ── Importer ──────────────────────────────────────────────────────────────────


class Neo4jImporter:
    """MERGE-based, idempotent importer for parsed legal JSON into Neo4j."""

    def __init__(
        self, uri: str, user: str, password: str, database: str = "neo4j"
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    # ── Setup ────────────────────────────────────────────────────────────────

    def ensure_constraints(self) -> None:
        """Create uniqueness constraints if they don't already exist."""
        with self.driver.session(database=self.database) as session:
            for stmt in get_constraint_statements():
                session.run(stmt)

    # ── Core import ─────────────────────────────────────────────────────────

    def import_parsed_file(self, path: Path) -> dict[str, int]:
        """
        Import a single parsed JSON file into Neo4j inside one transaction.

        Returns a dict with node/relationship merge counts.
        """
        payload = load_parsed_payload(path)
        nodes: dict[str, Any] = payload.get("nodes", {})
        relationships: list[dict[str, Any]] = payload.get("relationships", [])
        preamble: str = payload.get("preamble", "") or ""
        footer: str = payload.get("footer", "") or ""

        doc: dict[str, Any] | None = nodes.get("document")
        if doc is None:
            return {"skipped": 1, "nodes": 0, "rels": 0}

        doc_identity: str = doc["doc_identity"]
        counters: dict[str, int] = {"nodes": 0, "rels": 0}

        with self.driver.session(database=self.database) as session:
            # All writes in one transaction
            result = session.execute_write(self._tx_import, nodes, relationships, doc_identity, preamble, footer)
            counters.update(result)

        return counters

    def _tx_import(
        self,
        tx,
        nodes: dict[str, Any],
        relationships: list[dict[str, Any]],
        doc_identity: str,
        preamble: str,
        footer: str,
    ) -> dict[str, int]:
        counters: dict[str, int] = {"nodes": 0, "rels": 0}

        # ── Document ──────────────────────────────────────────────────────────
        doc = nodes.get("document", {})
        tx.run(
            """
            MERGE (d:Document {doc_identity: $doc_identity})
            SET d.doc_guid   = $doc_guid,
                d.doc_name    = $doc_name,
                d.issue_date  = $issue_date,
                d.effect_date = $effect_date,
                d.expire_date = $expire_date,
                d.gazette_number = $gazette_number,
                d.gazette_date   = $gazette_date,
                d.preamble    = $preamble,
                d.footer      = $footer
            """,
            doc_identity=doc_identity,
            doc_guid=doc.get("doc_guid"),
            doc_name=doc.get("doc_name"),
            issue_date=doc.get("issue_date"),
            effect_date=doc.get("effect_date"),
            expire_date=doc.get("expire_date"),
            gazette_number=doc.get("gazette_number"),
            gazette_date=doc.get("gazette_date"),
            preamble=preamble,
            footer=footer,
        )
        counters["nodes"] += 1

        # ── Metadata entities + relationships ─────────────────────────────────
        counters["nodes"] += self._upsert_metadata(tx, nodes, doc_identity)

        # ── Content hierarchy nodes + relationships ────────────────────────────
        counters["nodes"] += self._upsert_content_nodes(tx, nodes, doc_identity)
        counters["rels"] += self._upsert_content_relationships(tx, relationships, doc_identity)

        return counters

    def _upsert_metadata(self, tx, nodes: dict[str, Any], doc_identity: str) -> int:
        count = 0

        def upsert_node(label: str, key: str, props: dict) -> None:
            tx.run(
                f"MERGE (n:{label} {{{key}: ${key}}}) SET n += $props",
                key=props[key],
                props=props,
            )

        # DocumentGroup
        for g in nodes.get("document_group", []):
            upsert_node("DocumentGroup", "id", g)
            tx.run(
                """
                MATCH (d:Document {doc_identity: $doc_identity})
                MATCH (g:DocumentGroup {id: $id})
                MERGE (d)-[:BELONGS_TO_GROUP]->(g)
                """,
                doc_identity=doc_identity,
                id=g["id"],
            )
            count += 1

        # DocumentType
        for t in nodes.get("document_type", []):
            upsert_node("DocumentType", "id", t)
            tx.run(
                """
                MATCH (d:Document {doc_identity: $doc_identity})
                MATCH (t:DocumentType {id: $id})
                MERGE (d)-[:HAS_TYPE]->(t)
                """,
                doc_identity=doc_identity,
                id=t["id"],
            )
            count += 1

        # EffectStatus
        for s in nodes.get("effect_status", []):
            upsert_node("EffectStatus", "id", s)
            tx.run(
                """
                MATCH (d:Document {doc_identity: $doc_identity})
                MATCH (s:EffectStatus {id: $id})
                MERGE (d)-[:HAS_STATUS]->(s)
                """,
                doc_identity=doc_identity,
                id=s["id"],
            )
            count += 1

        # Organization
        for o in nodes.get("organizations", []):
            upsert_node("Organization", "id", o)
            tx.run(
                """
                MATCH (d:Document {doc_identity: $doc_identity})
                MATCH (o:Organization {id: $id})
                MERGE (d)-[:ISSUED_BY]->(o)
                """,
                doc_identity=doc_identity,
                id=o["id"],
            )
            count += 1

        # Signer
        for s in nodes.get("signers", []):
            upsert_node("Signer", "id", s)
            tx.run(
                """
                MATCH (d:Document {doc_identity: $doc_identity})
                MATCH (s:Signer {id: $id})
                MERGE (d)-[:SIGNED_BY]->(s)
                """,
                doc_identity=doc_identity,
                id=s["id"],
            )
            count += 1

        # Field
        for f in nodes.get("fields", []):
            upsert_node("Field", "id", f)
            tx.run(
                """
                MATCH (d:Document {doc_identity: $doc_identity})
                MATCH (f:Field {id: $id})
                MERGE (d)-[:IN_FIELD]->(f)
                """,
                doc_identity=doc_identity,
                id=f["id"],
            )
            count += 1

        # Related documents
        for rel in nodes.get("related_documents", []):
            target_identity = rel.get("doc_identity") or rel.get("doc_guid", "")
            if target_identity:
                # Upsert stub with at least doc_identity
                tx.run(
                    """
                    MERGE (t:Document {doc_identity: $target_identity})
                    SET t.doc_name = coalesce(t.doc_name, $doc_name),
                        t.doc_guid  = coalesce(t.doc_guid, $doc_guid)
                    """,
                    target_identity=target_identity,
                    doc_name=rel.get("doc_name"),
                    doc_guid=rel.get("doc_guid"),
                )
                tx.run(
                    """
                    MATCH (d:Document {doc_identity: $doc_identity})
                    MATCH (t:Document {doc_identity: $target_identity})
                    MERGE (d)-[:RELATED_TO]->(t)
                    """,
                    doc_identity=doc_identity,
                    target_identity=target_identity,
                )
                count += 1

        return count

    def _upsert_content_nodes(
        self, tx, nodes: dict[str, Any], doc_identity: str
    ) -> int:
        count = 0

        # Parts
        for p in nodes.get("parts", []):
            uid = build_part_uid(doc_identity, p["number"])
            tx.run(
                """
                MERGE (n:Part {uid: $uid})
                SET n.number = $number, n.title = $title,
                    n.content = $content, n.doc_identity = $doc_identity
                """,
                uid=uid,
                number=p["number"],
                title=p.get("title"),
                content=p.get("content", ""),
                doc_identity=doc_identity,
            )
            count += 1

        # Chapters
        for ch in nodes.get("chapters", []):
            uid = build_chapter_uid(doc_identity, ch["number"])
            tx.run(
                """
                MERGE (n:Chapter {uid: $uid})
                SET n.number = $number, n.title = $title,
                    n.content = $content,
                    n.doc_identity = $doc_identity,
                    n.parent_part = $parent_part
                """,
                uid=uid,
                number=ch["number"],
                title=ch.get("title"),
                content=ch.get("content", ""),
                doc_identity=doc_identity,
                parent_part=ch.get("parent_part"),
            )
            count += 1

        # Sections
        for sec in nodes.get("sections", []):
            uid = build_section_uid(doc_identity, sec["number"])
            tx.run(
                """
                MERGE (n:Section {uid: $uid})
                SET n.number = $number, n.title = $title,
                    n.content = $content,
                    n.doc_identity = $doc_identity,
                    n.parent_chapter = $parent_chapter
                """,
                uid=uid,
                number=sec["number"],
                title=sec.get("title"),
                content=sec.get("content", ""),
                doc_identity=doc_identity,
                parent_chapter=sec.get("parent_chapter"),
            )
            count += 1

        # Articles
        for a in nodes.get("articles", []):
            uid = build_article_uid(doc_identity, a["number"])
            tx.run(
                """
                MERGE (n:Article {uid: $uid})
                SET n.number = $number, n.title = $title,
                    n.content = $content,
                    n.doc_identity = $doc_identity,
                    n.parent_chapter = $parent_chapter,
                    n.parent_section = $parent_section,
                    n.order = $order
                """,
                uid=uid,
                number=a["number"],
                title=a.get("title"),
                content=a.get("content", ""),
                doc_identity=doc_identity,
                parent_chapter=a.get("parent_chapter"),
                parent_section=a.get("parent_section"),
                order=a.get("order"),
            )
            count += 1

        # Clauses
        for cl in nodes.get("clauses", []):
            uid = build_clause_uid(doc_identity, cl["parent_article"], cl["number"])
            tx.run(
                """
                MERGE (n:Clause {uid: $uid})
                SET n.number = $number, n.content = $content,
                    n.doc_identity = $doc_identity,
                    n.parent_article = $parent_article,
                    n.order = $order
                """,
                uid=uid,
                number=cl["number"],
                content=cl.get("content", ""),
                doc_identity=doc_identity,
                parent_article=cl["parent_article"],
                order=cl.get("order"),
            )
            count += 1

        # Points
        for pt in nodes.get("points", []):
            uid = build_point_uid(
                doc_identity,
                pt["parent_article"],
                pt["parent_clause"],
                pt["letter"],
            )
            tx.run(
                """
                MERGE (n:Point {uid: $uid})
                SET n.letter = $letter, n.content = $content,
                    n.doc_identity = $doc_identity,
                    n.parent_article = $parent_article,
                    n.parent_clause = $parent_clause,
                    n.order = $order
                """,
                uid=uid,
                letter=pt["letter"],
                content=pt.get("content", ""),
                doc_identity=doc_identity,
                parent_article=pt.get("parent_article"),
                parent_clause=pt.get("parent_clause"),
                order=pt.get("order"),
            )
            count += 1

        return count

    def _upsert_content_relationships(
        self, tx, relationships: list[dict[str, Any]], doc_identity: str
    ) -> int:
        count = 0
        for rel in relationships:
            rel_type = rel["type"]
            from_label = rel["from_label"]
            from_id = rel["from_id"]
            to_label = rel["to_label"]
            to_id = rel["to_id"]

            # Skip inter-document relationships here (handled in _upsert_metadata)
            if from_label == "Document" and rel_type in (
                "BELONGS_TO_GROUP",
                "HAS_TYPE",
                "HAS_STATUS",
                "ISSUED_BY",
                "SIGNED_BY",
                "IN_FIELD",
            ):
                continue

            tx.run(
                f"""
                MATCH (from:{from_label} {{{_id_prop(from_label)}: $from_id}})
                MATCH (to:{to_label} {{{_id_prop(to_label)}: $to_id}})
                MERGE (from)-[r:{rel_type}]->(to)
                """,
                from_id=from_id,
                to_id=to_id,
            )
            count += 1

        return count

    # ── Directory import ─────────────────────────────────────────────────────

    def import_parsed_directory(
        self,
        input_dir: Path,
        pattern: str = "*.json",
        fail_fast: bool = False,
    ) -> dict[str, Any]:
        """
        Import all parsed JSON files from *input_dir*.

        Returns a summary dict:
          total, succeeded, failed, node_count, rel_count, errors
        """
        files = sorted(input_dir.glob(pattern))
        succeeded = 0
        failed = 0
        node_count = 0
        rel_count = 0
        errors: list[dict[str, str]] = []

        for path in files:
            try:
                counters = self.import_parsed_file(path)
                node_count += counters.get("nodes", 0)
                rel_count += counters.get("rels", 0)
                succeeded += 1
            except Exception as exc:  # pragma: no cover
                errors.append({"file": path.name, "error": str(exc)})
                failed += 1
                if fail_fast:
                    break

        return {
            "total": len(files),
            "succeeded": succeeded,
            "failed": failed,
            "node_count": node_count,
            "rel_count": rel_count,
            "errors": errors,
        }


# ── ID property helpers ──────────────────────────────────────────────────────

def _id_prop(label: str) -> str:
    """Return the primary key property name for a node label."""
    return {
        "Document": "doc_identity",
        "DocumentGroup": "id",
        "DocumentType": "id",
        "EffectStatus": "id",
        "Organization": "id",
        "Signer": "id",
        "Field": "id",
        "Part": "uid",
        "Chapter": "uid",
        "Section": "uid",
        "Article": "uid",
        "Clause": "uid",
        "Point": "uid",
    }.get(label, "id")
