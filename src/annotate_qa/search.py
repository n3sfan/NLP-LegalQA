"""Builds an in-memory search index from parsed legal documents."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ArticleEntry:
    doc_identity: str
    doc_name: str       # e.g. "Luật Trật tự, an toàn giao thông đường bộ năm 2024"
    article_num: int
    title: str
    uid: str            # e.g. "56/2024/QH15::article::10"

    def matches(self, query: str) -> bool:
        q = query.lower().strip()
        # Strip Vietnamese "Điều N" prefix if present, keep remainder
        if q.startswith("điều "):
            q_stripped = q[len("điều "):]
            if q_stripped.isdigit() and int(q_stripped) == self.article_num:
                return True
        if str(self.article_num) == q:
            return True
        if q in self.doc_name.lower():
            return True
        if q in self.doc_identity.lower():
            return True
        if q in self.title.lower():
            return True
        return False

    def matches_content(self, query: str) -> bool:
        """Return True if query appears in article title or doc name (for clause/point ranking)."""
        q = query.lower().strip()
        return (
            q in self.doc_name.lower()
            or q in self.doc_identity.lower()
            or q in self.title.lower()
        )


@dataclass
class ClauseEntry:
    doc_identity: str
    article_num: int
    clause_num: int
    content: str
    uid: str   # e.g. "56/2024/QH15::article::1::clause::1"


@dataclass
class PointEntry:
    doc_identity: str
    article_num: int
    clause_num: int
    point_letter: str
    content: str
    uid: str   # e.g. "56/2024/QH15::article::1::clause::1::point::a"


class SearchIndex:
    def __init__(self) -> None:
        self.entries: list[ArticleEntry] = []
        # Keyed by (doc_identity, article_num) → list of entries
        self._clauses: dict[tuple[str, int], list[ClauseEntry]] = {}
        self._points:  dict[tuple[str, int], list[PointEntry]] = {}

    def build(self, parsed_dir: Path) -> None:
        """Load all parsed JSON files and index every article, clause, and point."""
        self.entries.clear()
        self._clauses.clear()
        self._points.clear()

        for json_path in parsed_dir.glob("*.json"):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            self._index_document(data)

    def _index_document(self, data: dict) -> None:
        nodes = data.get("nodes", {})
        doc_identity = nodes.get("document", {}).get("doc_identity", "")
        doc_name = nodes.get("document", {}).get("doc_name", "")

        # ---- Articles ----
        for article in nodes.get("articles", []):
            num_str = article.get("number", "0")
            try:
                num = int(num_str)
            except (ValueError, TypeError):
                num = 0
            title = article.get("title", "")
            uid = f"{doc_identity}::article::{num}"
            self.entries.append(
                ArticleEntry(
                    doc_identity=doc_identity,
                    doc_name=doc_name,
                    article_num=num,
                    title=title,
                    uid=uid,
                )
            )

        # ---- Clauses ----
        for clause in nodes.get("clauses", []):
            parent_str = str(clause.get("parent_article", ""))
            try:
                art_num = int(parent_str)
            except (ValueError, TypeError):
                art_num = 0
            clause_str = str(clause.get("number", ""))
            try:
                cl_num = int(clause_str)
            except (ValueError, TypeError):
                cl_num = 0
            content = clause.get("content", "")
            uid = f"{doc_identity}::article::{art_num}::clause::{cl_num}"
            key = (doc_identity, art_num)
            if key not in self._clauses:
                self._clauses[key] = []
            self._clauses[key].append(
                ClauseEntry(
                    doc_identity=doc_identity,
                    article_num=art_num,
                    clause_num=cl_num,
                    content=content,
                    uid=uid,
                )
            )

        # ---- Points ----
        for point in nodes.get("points", []):
            parent_art_str = str(point.get("parent_article", ""))
            try:
                art_num = int(parent_art_str)
            except (ValueError, TypeError):
                art_num = 0
            parent_cl_str = str(point.get("parent_clause", ""))
            try:
                cl_num = int(parent_cl_str)
            except (ValueError, TypeError):
                cl_num = 0
            letter = point.get("letter", "")
            content = point.get("content", "")
            uid = f"{doc_identity}::article::{art_num}::clause::{cl_num}::point::{letter}"
            key = (doc_identity, art_num)
            if key not in self._points:
                self._points[key] = []
            self._points[key].append(
                PointEntry(
                    doc_identity=doc_identity,
                    article_num=art_num,
                    clause_num=cl_num,
                    point_letter=letter,
                    content=content,
                    uid=uid,
                )
            )

    def search(self, query: str, limit: int = 20) -> list[ArticleEntry]:
        if not query.strip():
            return []
        q = query.lower().strip()
        # Normalise "điều N" → "N" for sort comparison
        q_art = q
        if q_art.startswith("điều "):
            q_art = q_art[len("điều "):]
        results = [e for e in self.entries if e.matches(q)]
        def sort_key(e: ArticleEntry) -> tuple[int, int, int, str]:
            art_match   = 0 if str(e.article_num) == q_art else 1
            name_match  = 0 if q in e.doc_name.lower() else 1
            ident_match = 0 if q in e.doc_identity.lower() else 1
            return (art_match, name_match, ident_match, e.doc_identity)
        results.sort(key=sort_key)
        return results[:limit]

    def search_all(
        self, query: str, limit: int = 20
    ) -> tuple[list[ArticleEntry], list[ClauseEntry], list[PointEntry]]:
        """Search articles, clauses, and points; group by type, ranked by relevance."""
        if not query.strip():
            return [], [], []
        q = query.lower().strip()

        # Articles
        q_art = q
        if q_art.startswith("điều "):
            q_art = q_art[len("điều "):]
        articles = [e for e in self.entries if e.matches(q)]
        def art_sort_key(e: ArticleEntry) -> tuple[int, int, int, str]:
            art_match   = 0 if str(e.article_num) == q_art else 1
            name_match  = 0 if q in e.doc_name.lower() else 1
            ident_match = 0 if q in e.doc_identity.lower() else 1
            return (art_match, name_match, ident_match, e.doc_identity)
        articles.sort(key=art_sort_key)
        articles = articles[:limit]

        # Clauses: match on content, plus rank by doc/title relevance
        doc_ids = {e.doc_identity for e in self.entries}
        all_clauses: list[ClauseEntry] = []
        for key, cls in self._clauses.items():
            doc_identity, _ = key
            doc_matches = any(
                e.doc_identity == doc_identity and e.matches_content(q)
                for e in self.entries
            )
            for c in cls:
                if q in c.content.lower():
                    all_clauses.append(c)
            # Clause sort key: doc/title match first, then clause number
        def clause_sort_key(c: ClauseEntry) -> tuple[int, int, int, str, int]:
            doc_matches = any(
                e.doc_identity == c.doc_identity and e.matches_content(q)
                for e in self.entries
            )
            exact_num = 0 if str(c.clause_num) == q_art else 1
            return (0 if doc_matches else 1, exact_num, 0, c.doc_identity, c.clause_num)
        all_clauses.sort(key=clause_sort_key)
        clauses = all_clauses[:limit]

        # Points: match on content, rank by doc/title relevance
        all_points: list[PointEntry] = []
        for pts in self._points.values():
            for p in pts:
                if q in p.content.lower():
                    all_points.append(p)
        def point_sort_key(p: PointEntry) -> tuple[int, int, str, int, str]:
            doc_matches = any(
                e.doc_identity == p.doc_identity and e.matches_content(q)
                for e in self.entries
            )
            return (0 if doc_matches else 1, 0, p.doc_identity, p.clause_num, p.point_letter)
        all_points.sort(key=point_sort_key)
        points = all_points[:limit]

        return articles, clauses, points

    def get_article_children(
        self, doc_identity: str, article_num: int
    ) -> tuple[list[ClauseEntry], list[PointEntry]]:
        """Return clauses and points for a given article."""
        key = (doc_identity, article_num)
        clauses = sorted(self._clauses.get(key, []), key=lambda c: c.clause_num)
        points  = sorted(self._points.get(key, []),  key=lambda p: (p.clause_num, p.point_letter))
        return clauses, points
