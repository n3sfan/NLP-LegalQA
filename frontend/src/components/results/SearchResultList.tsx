import { CitationCard } from './CitationCard';
import { ResultSkeleton } from './ResultSkeleton';
import { EmptyState } from '@/components/search/EmptyState';
import type { RerankResult } from '@/api/types';

interface SearchResultListProps {
  results: RerankResult[];
  loading: boolean;
  query: string;
  onSelect?: (uid: string) => void;
}

export function SearchResultList({ results, loading, query, onSelect }: SearchResultListProps) {
  if (loading) return <ResultSkeleton count={4} />;

  if (!results.length) {
    return <EmptyState query={query} />;
  }

  // Sort by rerank_score descending
  const sorted = [...results].sort((a, b) => b.rerank_score - a.rerank_score);

  return (
    <div className="h-[calc(100vh-18rem)] overflow-y-auto pr-4">
      <div className="space-y-3">
        {sorted.map((result, i) => (
          <CitationCard
            key={result.uid}
            result={result}
            rank={i + 1}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  );
}