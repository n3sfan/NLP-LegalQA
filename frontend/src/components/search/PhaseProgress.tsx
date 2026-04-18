import { cn } from '@/lib/utils';
import type { PipelinePhase } from '@/api/types';

const PHASES: { phase: PipelinePhase; label: string }[] = [
  { phase: 'decomposing',      label: 'Đang phân tích truy vấn' },
  { phase: 'vector_searching', label: 'Đang tìm kiếm vector' },
  { phase: 'reranking',        label: 'Đang xếp hạng lại' },
];

export function PhaseProgress({ phase }: { phase: PipelinePhase }) {
  if (phase === 'idle' || phase === 'done' || phase === 'error') return null;

  const activeIndex = PHASES.findIndex((p) => p.phase === phase);

  return (
    <div className="flex items-center gap-2 text-sm text-muted-foreground">
      {PHASES.map(({ phase: p, label }, i) => {
        const isActive  = i === activeIndex;
        const isDone    = i < activeIndex;
        const isPending = i > activeIndex;

        return (
          <div key={p} className="flex items-center gap-1.5">
            {/* Step indicator */}
            <span
              className={cn(
                'flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-semibold',
                isActive  && 'bg-primary text-primary-foreground animate-phase-pulse',
                isDone    && 'bg-green-500 text-white',
                isPending && 'bg-muted text-muted-foreground'
              )}
            >
              {isDone ? '✓' : i + 1}
            </span>
            <span className={cn(isActive && 'text-foreground font-medium')}>{label}</span>
            {i < PHASES.length - 1 && (
              <span className="text-muted-foreground/40 mx-0.5">›</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
