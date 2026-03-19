"""Data models for Neo4j graph nodes."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Metadata nodes ──────────────────────────────────────────────


@dataclass
class Document:
    doc_guid: str
    doc_identity: str
    doc_name: str
    issue_date: Optional[str] = None
    effect_date: Optional[str] = None
    expire_date: Optional[str] = None
    gazette_number: Optional[str] = None
    gazette_date: Optional[str] = None


@dataclass
class DocumentGroup:
    id: int
    name: str


@dataclass
class DocumentType:
    id: int
    name: str


@dataclass
class EffectStatus:
    id: int
    name: str


@dataclass
class Organization:
    id: int
    name: str


@dataclass
class Signer:
    id: int
    name: str


@dataclass
class Field:
    id: int
    name: str


@dataclass
class RelatedDocument:
    doc_guid: str
    doc_identity: str
    doc_name: str
    field_name: Optional[str] = None


# ── Content nodes ───────────────────────────────────────────────


@dataclass
class Part:
    number: str
    title: str
    doc_identity: str
    order: int
    content: str = ""


@dataclass
class Chapter:
    number: str
    title: str
    doc_identity: str
    parent_part: Optional[str]
    order: int
    content: str = ""


@dataclass
class Section:
    number: str
    title: str
    doc_identity: str
    parent_chapter: Optional[str]
    order: int
    content: str = ""


@dataclass
class Article:
    number: str
    title: str
    doc_identity: str
    parent_chapter: Optional[str]
    parent_section: Optional[str]
    order: int
    content: str = ""


@dataclass
class Clause:
    number: str
    doc_identity: str
    parent_article: str
    order: int
    content: str = ""


@dataclass
class Point:
    letter: str
    doc_identity: str
    parent_article: str
    parent_clause: str
    order: int
    content: str = ""


@dataclass
class Appendix:
    number: str
    title: str
    doc_identity: str
    order: int
    content: str = ""


# ── Relationship ────────────────────────────────────────────────


@dataclass
class Relationship:
    type: str
    from_label: str
    from_id: str
    to_label: str
    to_id: str


# ── Amend Relationship ─────────────────────────────────────────


@dataclass
class Amend:
    """Represents an amendment relationship."""

    # The amending document 
    amending_doc_identity: str  # e.g., "67/2020/QH14"
    amending_article: str  # e.g., "Điều 2"
    amending_clause: str | None = None  

    # Type of amendment
    amend_type: str = "sửa đổi" 

    # The target document being amended
    target_doc_identity: str | None = None  # e.g., "41/2004/QH11"
    target_article: str | None = None  # e.g., "Điều 163"
    target_clause: str | None = None  # e.g., "khoản 1"
    target_point: str | None = None  # e.g., "điểm đ"

    # Optional
    original_text: str | None = None

    # New text being added/replaced
    new_text: str | None = None


@dataclass
class AmendRelationship:
    """Neo4j relationship for amend."""

    type: str = "AMENDS"

    from_label: str = "Document"
    from_id: str = ""  # amending_doc_identity

    to_label: str = "Document"
    to_id: str = ""  # target_doc_identity

    # Additional properties
    amending_article: str = ""
    amending_clause: str | None = None
    amend_type: str = "sửa đổi"
    target_article: str = ""
    target_clause: str | None = None
    target_point: str | None = None
