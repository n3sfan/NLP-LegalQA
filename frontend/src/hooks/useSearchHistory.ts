/**
 * localStorage-persisted search history.
 */

import { useReducer, useEffect } from 'react';

export interface HistoryEntry {
  id: string;
  query: string;
  timestamp: number;
  resultCount: number;
}

type Action =
  | { type: 'LOAD'; entries: HistoryEntry[] }
  | { type: 'ADD'; entry: HistoryEntry }
  | { type: 'REMOVE'; id: string }
  | { type: 'CLEAR' };

function historyReducer(state: HistoryEntry[], action: Action): HistoryEntry[] {
  switch (action.type) {
    case 'LOAD':
      return action.entries;
    case 'ADD':
      // Deduplicate same query
      return [action.entry, ...state.filter((e) => e.query !== action.entry.query)].slice(0, 50);
    case 'REMOVE':
      return state.filter((e) => e.id !== action.id);
    case 'CLEAR':
      return [];
    default:
      return state;
  }
}

const STORAGE_KEY = 'legalqa_history';

function loadFromStorage(): HistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]') as HistoryEntry[];
  } catch {
    return [];
  }
}

function saveToStorage(entries: HistoryEntry[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
}

export function useSearchHistory() {
  const [entries, dispatch] = useReducer(historyReducer, [], loadFromStorage);

  useEffect(() => {
    saveToStorage(entries);
  }, [entries]);

  function addEntry(query: string, resultCount: number) {
    const entry: HistoryEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      query,
      timestamp: Date.now(),
      resultCount,
    };
    dispatch({ type: 'ADD', entry });
  }

  function removeEntry(id: string) {
    dispatch({ type: 'REMOVE', id });
  }

  function clearHistory() {
    dispatch({ type: 'CLEAR' });
  }

  return { entries, addEntry, removeEntry, clearHistory };
}
