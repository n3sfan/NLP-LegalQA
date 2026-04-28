import { Copy, ChevronRight } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { HierarchyBreadcrumb } from './HierarchyBreadcrumb';
import { ScoreBar } from './ScoreBar';
import { useCopyToClipboard } from '@/hooks/useCopyToClipboard';
import type { RerankResult } from '@/api/types';
import { parseUID } from '@/lib/uid';

interface CitationCardProps {
  result: RerankResult;
  rank: number;
  onSelect?: (uid: string) => void;
}

const BADGE_VARIANT_MAP: Record<string, 'article' | 'clause' | 'point' | 'chapter' | 'section' | 'part' | 'document'> = {
  Article:  'article',
  Clause:   'clause',
  Point:    'point',
  Chapter:  'chapter',
  Section:  'section',
  Part:     'part',
  Document: 'document',
};

export function CitationCard({ result, rank, onSelect }: CitationCardProps) {
  const { copy } = useCopyToClipboard();
  const parsed = parseUID(result.uid);

  return (
    <Card
      className="group cursor-pointer transition-shadow hover:shadow-md"
      onClick={() => onSelect?.(result.uid)}
    >
      <CardContent className="space-y-2 p-4">
        {/* Header row: rank + label badge + copy */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-muted text-[10px] font-bold text-muted-foreground">
              {rank}
            </span>
            <Badge variant={BADGE_VARIANT_MAP[result.label] ?? 'default'}>
              {result.label}
            </Badge>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
            onClick={(e) => {
              e.stopPropagation();
              copy(result.text, 'Đã sao chép điều khoản');
            }}
          >
            <Copy className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* Hierarchy breadcrumb */}
        <HierarchyBreadcrumb
          uid={result.uid}
          docIdentity={parsed.docIdentity}
        />

        {/* Content */}
        <Separator className="my-2" />
        <p className="text-sm leading-relaxed">{result.text}</p>

        {/* Score bars */}
        <div className="flex flex-col gap-2 pt-1">
          <ScoreBar score={result.rerank_score} label="Điểm xếp hạng" />
          <ScoreBar score={result.score} label="Điểm vector" />
        </div>

        {/* Navigate hint */}
        <div className="flex items-center gap-1 text-xs text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity pt-1">
          <span>Xem chi tiết</span>
          <ChevronRight className="h-3 w-3" />
        </div>
      </CardContent>
    </Card>
  );
}