import { useCallback, useEffect } from 'react';
import { toast } from 'sonner';
import { useSearchPipeline } from '@/hooks/useSearchPipeline';
import { useSearchHistory } from '@/hooks/useSearchHistory';
import { ChatInput } from '@/components/search/ChatInput';
import { PhaseProgress } from '@/components/search/PhaseProgress';
import { QueryDecomposer } from '@/components/search/QueryDecomposer';
import { SearchResultList } from '@/components/results/SearchResultList';
import { EmptyState } from '@/components/search/EmptyState';
import { Separator } from '@/components/ui/separator';

export function HomePage() {
  const { state, run, reset } = useSearchPipeline();
  const { addEntry } = useSearchHistory();

  const isLoading =
    state.phase === 'decomposing' ||
    state.phase === 'vector_searching' ||
    state.phase === 'reranking';

  const handleSubmit = useCallback(
    (query: string) => {
      reset();
      run(query);
    },
    [run, reset]
  );

  // Save to history when done
  useEffect(() => {
    if (state.phase === 'done' && state.query) {
      addEntry(state.query, state.rerankResults.length);
    }
  }, [state.phase, state.query, state.rerankResults.length, addEntry]);

  // Error toast
  useEffect(() => {
    if (state.phase === 'error' && state.error) {
      toast.error(state.error);
    }
  }, [state.phase, state.error]);

  const showResults =
    state.phase === 'done' || state.phase === 'reranking';

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-4 px-4 py-6">
      {/* Hero / intro */}
      <div className="text-center">
        <h1 className="text-3xl font-bold tracking-tight">Pháp Luật QA</h1>
        <p className="mt-1 text-muted-foreground">
          Tra cứu pháp luật Việt Nam bằng câu hỏi tự nhiên
        </p>
      </div>

      {/* Input */}
      <ChatInput
        onSubmit={handleSubmit}
        loading={isLoading}
        placeholder="Ví dụ: không đội mũ bảo hiểm phạt bao nhiêu?"
      />

      {/* Phase progress */}
      {isLoading && <PhaseProgress phase={state.phase} />}

      {/* Decomposition */}
      <QueryDecomposer
        decomposition={state.decompose}
        loading={state.phase === 'decomposing'}
      />

      <Separator />

      {/* Results or empty state */}
      {showResults ? (
        <SearchResultList
          results={state.rerankResults}
          loading={state.phase === 'reranking'}
          query={state.query}
        />
      ) : state.phase === 'idle' ? (
        <EmptyState />
      ) : null}
    </div>
  );
}
