/**
 * Search pipeline state machine hook.
 * Orchestrates: query decomposition → vector search → cross-encoder reranking.
 * Currently uses mock API — swap for real ./client.ts functions when backend is ready.
 */

import { useReducer } from 'react';
import type {
  PipelineState,
  DecomposeResult,
  SearchResult,
  RerankResult,
} from '@/api/types';
import { mockDecomposeQuery, mockVectorSearch, mockRerank } from '@/api/mock';

type Action =
  | { type: 'START'; query: string }
  | { type: 'DECOMPOSE_DONE'; result: DecomposeResult }
  | { type: 'VECTOR_DONE'; results: SearchResult[] }
  | { type: 'RERANK_DONE'; results: RerankResult[] }
  | { type: 'ERROR'; message: string }
  | { type: 'RESET' };

export const INITIAL_STATE: PipelineState = {
  phase: 'idle',
  query: '',
  decompose: null,
  vectorResults: [],
  rerankResults: [],
  error: null,
};

function reducer(state: PipelineState, action: Action): PipelineState {
  switch (action.type) {
    case 'START':
      return { ...INITIAL_STATE, query: action.query, phase: 'decomposing' };
    case 'DECOMPOSE_DONE':
      return { ...state, phase: 'vector_searching', decompose: action.result };
    case 'VECTOR_DONE':
      return { ...state, phase: 'reranking', vectorResults: action.results };
    case 'RERANK_DONE':
      return { ...state, phase: 'done', rerankResults: action.results };
    case 'ERROR':
      return { ...state, phase: 'error', error: action.message };
    case 'RESET':
      return INITIAL_STATE;
    default:
      return state;
  }
}

export function useSearchPipeline() {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);

  async function run(query: string) {
    if (!query.trim()) return;
    dispatch({ type: 'START', query });

    try {
      // ── Step 1: Decompose ────────────────────────────────────────────────────
      const decompose = await mockDecomposeQuery(query);
      dispatch({ type: 'DECOMPOSE_DONE', result: decompose });

      if (!decompose.success) {
        dispatch({ type: 'ERROR', message: 'Không thể phân tích truy vấn.' });
        return;
      }

      // ── Step 2: Vector search ────────────────────────────────────────────────
      const subQueries = decompose.sub_queries.map((s) => s.query);
      const vectorResults = await mockVectorSearch(subQueries);
      dispatch({ type: 'VECTOR_DONE', results: vectorResults });

      if (vectorResults.length === 0) {
        dispatch({ type: 'ERROR', message: 'Không tìm thấy kết quả phù hợp.' });
        return;
      }

      // ── Step 3: Rerank ──────────────────────────────────────────────────────
      const rerankResults = await mockRerank(subQueries, vectorResults);
      dispatch({ type: 'RERANK_DONE', results: rerankResults });
    } catch (err) {
      dispatch({ type: 'ERROR', message: String(err) });
    }
  }

  function reset() {
    dispatch({ type: 'RESET' });
  }

  return { state, run, reset };
}
