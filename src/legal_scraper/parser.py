"""Parsers for extracting structured entities from legal documents."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from legal_scraper.models import (
    Article,
    Chapter,
    Clause,
    Document,
    DocumentGroup,
    DocumentType,
    EffectStatus,
    Field,
    Organization,
    Part,
    Point,
    RelatedDocument,
    Relationship,
    Section,
    Signer,
)


# ── Metadata Parser ────────────────────────────────────────────


class MetadataParser:
    """Parse JSON metadata files into structured entities and relationships."""

    def parse(self, json_path: Path) -> dict:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        doc_identity = data["docIdentity"]

        document = Document(
            doc_guid=data["docGUId"],
            doc_identity=doc_identity,
            doc_name=data["docName"],
            issue_date=data.get("issueDate"),
            effect_date=data.get("effectDate"),
            expire_date=data.get("expireDate"),
            gazette_number=data.get("gazetteNumber"),
            gazette_date=data.get("gazetteDate"),
        )

        entities: dict[str, list] = {
            "document_group": [],
            "document_type": [],
            "effect_status": [],
            "organizations": [],
            "signers": [],
            "fields": [],
            "related_documents": [],
        }
        relationships: list[Relationship] = []

        # Document Group
        if dg := data.get("docGroup"):
            group = DocumentGroup(id=dg["docGroupId"], name=dg["docGroupName"])
            entities["document_group"].append(group)
            relationships.append(Relationship(
                type="BELONGS_TO_GROUP",
                from_label="Document", from_id=doc_identity,
                to_label="DocumentGroup", to_id=str(dg["docGroupId"]),
            ))

        # Document Type
        if dt := data.get("docType"):
            doc_type = DocumentType(id=dt["docTypeId"], name=dt["docTypeName"])
            entities["document_type"].append(doc_type)
            relationships.append(Relationship(
                type="HAS_TYPE",
                from_label="Document", from_id=doc_identity,
                to_label="DocumentType", to_id=str(dt["docTypeId"]),
            ))

        # Effect Status
        if es := data.get("effectStatus"):
            status = EffectStatus(id=es["effectStatusId"], name=es["effectStatusName"])
            entities["effect_status"].append(status)
            relationships.append(Relationship(
                type="HAS_STATUS",
                from_label="Document", from_id=doc_identity,
                to_label="EffectStatus", to_id=str(es["effectStatusId"]),
            ))

        # Organizations (organs)
        for org_data in data.get("organs") or []:
            org = Organization(id=org_data["organId"], name=org_data["organName"])
            entities["organizations"].append(org)
            relationships.append(Relationship(
                type="ISSUED_BY",
                from_label="Document", from_id=doc_identity,
                to_label="Organization", to_id=str(org_data["organId"]),
            ))

        # Signers
        for sig_data in data.get("signers") or []:
            signer = Signer(id=sig_data["signerId"], name=sig_data["signerName"])
            entities["signers"].append(signer)
            relationships.append(Relationship(
                type="SIGNED_BY",
                from_label="Document", from_id=doc_identity,
                to_label="Signer", to_id=str(sig_data["signerId"]),
            ))

        # Fields
        for f_data in data.get("fields") or []:
            fld = Field(id=f_data["fieldId"], name=f_data["fieldName"])
            entities["fields"].append(fld)
            relationships.append(Relationship(
                type="IN_FIELD",
                from_label="Document", from_id=doc_identity,
                to_label="Field", to_id=str(f_data["fieldId"]),
            ))

        # Related documents
        for rel_data in data.get("docListOther") or []:
            rel_doc = RelatedDocument(
                doc_guid=rel_data["docGUId"],
                doc_identity=rel_data.get("docIdentity", ""),
                doc_name=rel_data.get("docName", ""),
                field_name=rel_data.get("fieldName"),
            )
            entities["related_documents"].append(rel_doc)
            relationships.append(Relationship(
                type="RELATED_TO",
                from_label="Document", from_id=doc_identity,
                to_label="Document", to_id=rel_data.get("docIdentity", rel_data["docGUId"]),
            ))

        return {
            "document": asdict(document),
            "entities": {k: [asdict(e) for e in v] for k, v in entities.items()},
            "relationships": [asdict(r) for r in relationships],
        }


# ── Content Parser ──────────────────────────────────────────────


# Vietnamese legal docs use this letter sequence for Điểm (Point)
_VIET_POINT_LETTERS = set("abcdeđghiklmnopqrstuvxy")
_VIET_POINT_RE_CLASS = "".join(sorted(_VIET_POINT_LETTERS - {"đ"})) + "đ"

# Regex patterns for structural elements
_RE_PART = re.compile(r"^(?:PHẦN|Phần)\s+thứ\s+(\S+?)\.?\s+(.*)", re.IGNORECASE)
_RE_CHAPTER = re.compile(r"^Chương\s+([IVXLCDM]+|\d+)\.?\s*(.*)", re.IGNORECASE)
_RE_SECTION = re.compile(r"^Mục\s+(\d+)\.?\s*(.*)")
_RE_ARTICLE = re.compile(r"^Điều\s+(\d+[a-z]?)\.?\s*(.*)")
_RE_CLAUSE = re.compile(r"^(\d+)\.\s+(.*)")
_RE_POINT = re.compile(rf"^([{_VIET_POINT_RE_CLASS}])\)\s+(.*)")

# Footer / appendix boundary — first match marks end of legal content
_RE_FOOTER_START = re.compile(
    r"^(?:"
    r"Luật này được Quốc hội.*thông qua"
    r"|CHỦ TỊCH QUỐC HỘI"
    r"|TM\.\s*CHÍNH PHỦ"
    r"|KT\.\s*THỦ TƯỚNG"
    r"|Nơi nhận:"
    r"|\(\*\)\s*Nguồn dữ liệu"
    r"|Phụ lục"
    r")",
    re.MULTILINE | re.IGNORECASE,
)


class ContentParser:
    """Parse plain-text legal document content into a hierarchical structure."""

    def parse(self, txt_path: Path, doc_identity: str) -> dict:
        text = txt_path.read_text(encoding="utf-8")
        return self.parse_text(text, doc_identity)

    def _split_document(self, text: str) -> tuple[str, str]:
        """Split text into (main_content, footer)."""
        match = _RE_FOOTER_START.search(text)
        if match:
            return text[:match.start()], text[match.start():]
        return text, ""

    def parse_text(self, text: str, doc_identity: str) -> dict:
        # Split footer/appendix before parsing
        main_text, footer = self._split_document(text)

        parts: list[Part] = []
        chapters: list[Chapter] = []
        sections: list[Section] = []
        articles: list[Article] = []
        clauses: list[Clause] = []
        points: list[Point] = []
        preamble_lines: list[str] = []

        cur_part: Part | None = None
        cur_chapter: Chapter | None = None
        cur_section: Section | None = None
        cur_article: Article | None = None
        cur_clause: Clause | None = None
        cur_point: Point | None = None
        in_quote: bool = False

        def append_content(ln: str) -> None:
            if cur_point:
                cur_point.content += "\n" + ln
            elif cur_clause:
                cur_clause.content += "\n" + ln
            elif cur_article:
                cur_article.content += "\n" + ln
            elif cur_section:
                cur_section.content += "\n" + ln
            elif cur_chapter:
                cur_chapter.content += "\n" + ln
            elif cur_part:
                cur_part.content += "\n" + ln
            else:
                preamble_lines.append(ln)

        for raw_line in main_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # ── Quote block tracking ──
            # Lines starting with " or " open a quoted amendment block.
            # Everything inside is treated as content, not parsed as hierarchy.
            opens_quote = line[0] in ('"', '\u201c')
            # Strip trailing punctuation, then check for closing quote
            closes_quote = line.rstrip('.;:, ') [-1:] in ('"', '\u201d')

            if opens_quote and not in_quote:
                in_quote = True

            if in_quote:
                append_content(line)
                if closes_quote:
                    in_quote = False
                continue

            # ── Hierarchy matching (only when NOT inside a quoted block) ──

            # ── Part ──
            if m := _RE_PART.match(line):
                cur_part = Part(
                    number=m.group(1), title=m.group(2).strip(),
                    doc_identity=doc_identity, order=len(parts),
                )
                parts.append(cur_part)
                cur_chapter = cur_section = cur_article = cur_clause = cur_point = None
                continue

            # ── Chapter ──
            if m := _RE_CHAPTER.match(line):
                cur_chapter = Chapter(
                    number=m.group(1).upper(), title=m.group(2).strip(),
                    doc_identity=doc_identity,
                    parent_part=cur_part.number if cur_part else None,
                    order=len(chapters),
                )
                chapters.append(cur_chapter)
                cur_section = cur_article = cur_clause = cur_point = None
                continue

            # ── Section ──
            if m := _RE_SECTION.match(line):
                cur_section = Section(
                    number=m.group(1), title=m.group(2).strip(),
                    doc_identity=doc_identity,
                    parent_chapter=cur_chapter.number if cur_chapter else None,
                    order=len(sections),
                )
                sections.append(cur_section)
                cur_article = cur_clause = cur_point = None
                continue

            # ── Article ──
            if m := _RE_ARTICLE.match(line):
                cur_article = Article(
                    number=m.group(1), title=m.group(2).strip(),
                    doc_identity=doc_identity,
                    parent_chapter=cur_chapter.number if cur_chapter else None,
                    parent_section=cur_section.number if cur_section else None,
                    order=len(articles),
                )
                articles.append(cur_article)
                cur_clause = cur_point = None
                continue

            # ── Clause (only valid inside an Article) ──
            if cur_article and (m := _RE_CLAUSE.match(line)):
                cur_clause = Clause(
                    number=m.group(1), content=m.group(2).strip(),
                    doc_identity=doc_identity,
                    parent_article=cur_article.number,
                    order=len(clauses),
                )
                clauses.append(cur_clause)
                cur_point = None
                continue

            # ── Point (only valid inside a Clause) ──
            if cur_clause and (m := _RE_POINT.match(line)):
                cur_point = Point(
                    letter=m.group(1), content=m.group(2).strip(),
                    doc_identity=doc_identity,
                    parent_article=cur_article.number if cur_article else "",
                    parent_clause=cur_clause.number,
                    order=len(points),
                )
                points.append(cur_point)
                continue

            # ── Content continuation ──
            append_content(line)

        # Build relationships
        relationships: list[dict] = []

        for part in parts:
            relationships.append(asdict(Relationship(
                type="HAS_PART",
                from_label="Document", from_id=doc_identity,
                to_label="Part", to_id=part.number,
            )))

        for ch in chapters:
            if ch.parent_part:
                relationships.append(asdict(Relationship(
                    type="HAS_CHAPTER",
                    from_label="Part", from_id=ch.parent_part,
                    to_label="Chapter", to_id=ch.number,
                )))
            else:
                relationships.append(asdict(Relationship(
                    type="HAS_CHAPTER",
                    from_label="Document", from_id=doc_identity,
                    to_label="Chapter", to_id=ch.number,
                )))

        for sec in sections:
            if sec.parent_chapter:
                relationships.append(asdict(Relationship(
                    type="HAS_SECTION",
                    from_label="Chapter", from_id=sec.parent_chapter,
                    to_label="Section", to_id=sec.number,
                )))
            else:
                relationships.append(asdict(Relationship(
                    type="HAS_SECTION",
                    from_label="Document", from_id=doc_identity,
                    to_label="Section", to_id=sec.number,
                )))

        for art in articles:
            if art.parent_section:
                relationships.append(asdict(Relationship(
                    type="HAS_ARTICLE",
                    from_label="Section", from_id=art.parent_section,
                    to_label="Article", to_id=art.number,
                )))
            elif art.parent_chapter:
                relationships.append(asdict(Relationship(
                    type="HAS_ARTICLE",
                    from_label="Chapter", from_id=art.parent_chapter,
                    to_label="Article", to_id=art.number,
                )))
            else:
                relationships.append(asdict(Relationship(
                    type="HAS_ARTICLE",
                    from_label="Document", from_id=doc_identity,
                    to_label="Article", to_id=art.number,
                )))

        for cl in clauses:
            relationships.append(asdict(Relationship(
                type="HAS_CLAUSE",
                from_label="Article", from_id=cl.parent_article,
                to_label="Clause", to_id=cl.number,
            )))

        for pt in points:
            relationships.append(asdict(Relationship(
                type="HAS_POINT",
                from_label="Clause", from_id=f"{pt.parent_article}.{pt.parent_clause}",
                to_label="Point", to_id=pt.letter,
            )))

        return {
            "preamble": "\n".join(preamble_lines),
            "footer": footer.strip(),
            "nodes": {
                "parts": [asdict(p) for p in parts],
                "chapters": [asdict(c) for c in chapters],
                "sections": [asdict(s) for s in sections],
                "articles": [asdict(a) for a in articles],
                "clauses": [asdict(c) for c in clauses],
                "points": [asdict(p) for p in points],
            },
            "relationships": relationships,
        }


# ── Main Orchestrator ───────────────────────────────────────────


class LegalDocumentParser:
    """Orchestrates metadata + content parsing for legal documents."""

    def __init__(self) -> None:
        self.metadata_parser = MetadataParser()
        self.content_parser = ContentParser()

    def parse_document(self, stem: str, data_dir: Path) -> dict:
        """Parse a single document by its filename stem (e.g. '59-2020-QH14')."""
        json_path = data_dir / f"{stem}.json"
        txt_path = data_dir / f"{stem}.txt"

        result: dict = {"doc_stem": stem}

        if json_path.exists():
            meta = self.metadata_parser.parse(json_path)
            result["nodes"] = {"document": meta["document"], **meta["entities"]}
            result["relationships"] = meta["relationships"]
        else:
            result["nodes"] = {}
            result["relationships"] = []

        if txt_path.exists():
            doc_identity = result["nodes"].get("document", {}).get("doc_identity", stem.replace("-", "/"))
            content = self.content_parser.parse(txt_path, doc_identity)
            result["nodes"].update(content["nodes"])
            result["relationships"].extend(content["relationships"])
            result["preamble"] = content.get("preamble", "")
            result["footer"] = content.get("footer", "")

        return result

    def parse_directory(self, data_dir: Path, output_dir: Path) -> list[Path]:
        """Parse all paired .json/.txt documents in a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect unique stems (filenames without extension)
        stems = {p.stem for p in data_dir.glob("*.json")} | {p.stem for p in data_dir.glob("*.txt")}
        # Exclude files in subdirectories
        stems = {s for s in stems if (data_dir / f"{s}.json").exists() or (data_dir / f"{s}.txt").exists()}

        saved: list[Path] = []
        for stem in sorted(stems):
            result = self.parse_document(stem, data_dir)
            out_path = output_dir / f"{stem}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            saved.append(out_path)

        return saved
