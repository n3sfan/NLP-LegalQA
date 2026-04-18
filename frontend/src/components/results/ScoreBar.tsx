import { cn } from '@/lib/utils';
import { Progress } from '@/components/ui/progress';

interface ScoreBarProps {
  score: number; // 0–1
  label?: string;
  className?: string;
}

export function ScoreBar({ score, label, className }: ScoreBarProps) {
  const pct = Math.round(score * 100);
  const color =
    score >= 0.85 ? 'bg-green-500' :
    score >= 0.70 ? 'bg-yellow-500' :
    'bg-red-500';

  return (
    <div className={cn('space-y-1', className)}>
      {label && (
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{label}</span>
          <span className="font-mono font-medium text-foreground">{pct}%</span>
        </div>
      )}
      <Progress
        value={pct}
        className="h-1.5"
        style={
          { '--progress-foreground': 'currentColor' } as React.CSSProperties
        }
      />
      <div
        className={cn('h-1.5 -mt-3 rounded-full transition-all duration-700', color)}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
