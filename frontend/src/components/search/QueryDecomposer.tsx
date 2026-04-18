import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Skeleton } from '@/components/ui/skeleton';
import type { DecomposeResult } from '@/api/types';
import { Sparkles } from 'lucide-react';

interface QueryDecomposerProps {
  decomposition: DecomposeResult | null;
  loading: boolean;
}

export function QueryDecomposer({ decomposition, loading }: QueryDecomposerProps) {
  if (loading) {
    return (
      <Card className="mb-4">
        <CardHeader className="pb-2">
          <Skeleton className="h-5 w-48" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-4 w-full mb-2" />
          <Skeleton className="h-4 w-3/4 mb-3" />
          <div className="flex gap-2">
            <Skeleton className="h-6 w-32" />
            <Skeleton className="h-6 w-40" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!decomposition || !decomposition.sub_queries.length) return null;

  return (
    <Card className="mb-4">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <Sparkles className="h-4 w-4 text-primary" />
          Phân tích truy vấn
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Reasoning */}
        <Accordion type="single" collapsible className="w-full mb-3">
          <AccordionItem value="reasoning" className="border-b-0 px-0">
            <AccordionTrigger className="py-1 text-xs text-muted-foreground hover:no-underline">
              Xem suy luận phân tích
            </AccordionTrigger>
            <AccordionContent className="text-sm leading-relaxed text-foreground/80">
              {decomposition.reasoning}
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        {/* Sub-queries */}
        <div className="flex flex-wrap gap-2">
          {decomposition.sub_queries.map((sq, i) => (
            <Badge
              key={i}
              variant="outline"
              className="text-xs font-normal"
            >
              <span className="mr-1.5 text-[10px] font-bold text-muted-foreground">#{i + 1}</span>
              {sq.query}
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
