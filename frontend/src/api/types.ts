// ─────────────────────────────────────────────────────────────────────────────
// Single source of truth for all TypeScript types mirroring Python dataclasses
// ─────────────────────────────────────────────────────────────────────────────

// Node label union (matches embedder.py SearchResult)
export type NodeLabel = 'Document' | 'Part' | 'Chapter' | 'Section' | 'Article' | 'Clause' | 'Point';

// ── Pipeline types (from embedder.py) ──────────────────────────────────────────

export interface SearchResult {
  uid: string;
  label: NodeLabel;
  score: number; // cosine similarity 0–1
}

export interface RerankResult {
  uid: string;
  label: NodeLabel;
  score: number;        // vector similarity
  text: string;         // full text content
  rerank_score: number; // cross-encoder score (higher = better)
}

export interface SubQuery {
  query: string;
  index?: number;
}

export interface DecomposeResult {
  sub_queries: SubQuery[];
  reasoning: string;
  success: boolean;
}

// ── Article hierarchy (from models.py) ────────────────────────────────────────

export interface Point {
  uid: string;
  letter: string;
  content: string;
  order: number;
}

export interface Clause {
  uid: string;
  number: string;
  content: string;
  order: number;
  points: Point[];
}

export interface Article {
  uid: string;
  number: string;
  title: string;
  content: string;
  order: number;
  clauses: Clause[];
}

export interface Document {
  doc_identity: string;
  doc_name: string;
  issue_date?: string;
  effect_date?: string;
  expire_date?: string;
  gazette_number?: string;
  gazette_date?: string;
}

// ── API response types (from annotate_qa/server.py) ───────────────────────────

export interface SearchApiArticle {
  type: 'article';
  doc_identity: string;
  doc_name: string;
  article_num: number;
  title: string;
  uid: string;
}

export interface SearchApiClause {
  type: 'clause';
  doc_identity: string;
  article_num: number;
  clause_num: number;
  content: string;
  uid: string;
}

export interface SearchApiPoint {
  type: 'point';
  doc_identity: string;
  article_num: number;
  clause_num: number;
  point_letter: string;
  content: string;
  uid: string;
}

export interface SearchApiResponse {
  articles: SearchApiArticle[];
  clauses: SearchApiClause[];
  points: SearchApiPoint[];
}

export interface ArticleApiResponse {
  doc_identity: string;
  article_num: number;
  article_uid: string;
  clauses: SearchApiClause[];
  points: SearchApiPoint[];
}

// ── Pipeline state machine ─────────────────────────────────────────────────────

export type PipelinePhase =
  | 'idle'
  | 'decomposing'
  | 'vector_searching'
  | 'reranking'
  | 'done'
  | 'error';

export interface PipelineState {
  phase: PipelinePhase;
  query: string;
  decompose: DecomposeResult | null;
  vectorResults: SearchResult[];
  rerankResults: RerankResult[];
  error: string | null;
}