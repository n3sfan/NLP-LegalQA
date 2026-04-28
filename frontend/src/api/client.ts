/**
 * Real HTTP client for annotate_qa FastAPI server.
 *
 * Pipeline endpoints (decompose, vector_search, rerank) are NOT yet
 * implemented in FastAPI — they are stubbed with comments below.
 * Once added, uncomment the relevant functions and swap imports in hooks.
 *
 * Existing FastAPI endpoints (currently wired):
 *   GET /api/search?q=          → SearchApiResponse
 *   GET /api/article?doc_identity=&article_num=  → ArticleApiResponse
 */

import type {
  SearchApiResponse,
  ArticleApiResponse,
} from './types';

const BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ── Existing FastAPI endpoints ──────────────────────────────────────────────────

export async function searchKeyword(q: string): Promise<SearchApiResponse> {
  return apiFetch<SearchApiResponse>(`/api/search?q=${encodeURIComponent(q)}`);
}

export async function getArticle(doc_identity: string, article_num: number): Promise<ArticleApiResponse> {
  const params = new URLSearchParams({ doc_identity: String(doc_identity), article_num: String(article_num) });
  return apiFetch<ArticleApiResponse>(`/api/article?${params}`);
}

// ── Pipeline endpoints (add to annotate_qa/server.py) ──────────────────────────
// Uncomment each block as you implement the corresponding FastAPI route.
//
// ── POST /api/pipeline/decompose ────────────────────────────────────────────
// Body: { query: string }
// Returns: DecomposeResult
// ─────────────────────────────────────────────────────────────────────────────
/*
export async function decomposeQuery(query: string): Promise<DecomposeResult> {
  return apiFetch<DecomposeResult>('/api/pipeline/decompose', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
}
*/

// ── POST /api/pipeline/search ─────────────────────────────────────────────────
// Body: { sub_queries: string[], k?: number }
// Returns: SearchResult[]
// ─────────────────────────────────────────────────────────────────────────────
/*
export async function vectorSearch(sub_queries: string[], k = 5): Promise<SearchResult[]> {
  return apiFetch<SearchResult[]>('/api/pipeline/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sub_queries, k }),
  });
}
*/

// ── POST /api/pipeline/rerank ───────────────────────────────────────────────
// Body: { sub_queries: string[], results: SearchResult[] }
// Returns: RerankResult[]
// ─────────────────────────────────────────────────────────────────────────────
/*
export async function rerankResults(
  sub_queries: string[],
  results: SearchResult[],
): Promise<RerankResult[]> {
  return apiFetch<RerankResult[]>('/api/pipeline/rerank', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sub_queries, results }),
  });
}
*/

// ── GET /api/pipeline/hierarchy?uids=uid1,uid2,… ─────────────────────────────
// Returns: Record<string, string>  (uid → hierarchy string)
// ─────────────────────────────────────────────────────────────────────────────
/*
export async function fetchHierarchy(uids: string[]): Promise<Record<string, string>> {
  return apiFetch<Record<string, string>>(`/api/pipeline/hierarchy?uids=${uids.join(',')}`);
}
*/
