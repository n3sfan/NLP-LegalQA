import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, BookOpen } from 'lucide-react';
import { useArticle } from '@/hooks/useArticle';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';

export function ArticlePage() {
  const { doc_identity, article_num } = useParams<{
    doc_identity: string;
    article_num: string;
  }>();

  const num = article_num ? parseInt(article_num, 10) : 0;

  const { data, loading, error } = useArticle({
    doc_identity: doc_identity ?? '',
    article_num: num,
  });

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-4 px-4 py-6">
      {/* Back nav */}
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" asChild>
          <Link to="/">
            <ArrowLeft className="mr-1 h-4 w-4" />
            Quay lại
          </Link>
        </Button>
      </div>

      {/* Header */}
      <div className="flex items-start gap-3">
        <BookOpen className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
        <div>
          <h1 className="text-xl font-bold">
            {data ? (
              <>Điều {num} — {doc_identity}</>
            ) : (
              <Skeleton className="h-6 w-64" />
            )}
          </h1>
          <p className="mt-0.5 text-sm text-muted-foreground">{doc_identity}</p>
        </div>
      </div>

      {loading && (
        <div className="space-y-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-20 w-full rounded-lg" />
          ))}
        </div>
      )}

      {error && (
        <Card className="border-destructive">
          <CardContent className="p-4 text-sm text-destructive">
            {error}
          </CardContent>
        </Card>
      )}

      {data && (
        <div className="space-y-4">
          {/* Clauses */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Các khoản</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {data.clauses.map((clause) => (
                <div key={clause.uid}>
                  <div className="flex items-start gap-2">
                    <Badge variant="clause" className="mt-0.5 shrink-0">
                      Khoản {clause.clause_num}
                    </Badge>
                    <p className="text-sm leading-relaxed">{clause.content}</p>
                  </div>
                  {/* Points for this clause */}
                  {data.points
                    .filter((p) => p.clause_num === clause.clause_num)
                    .map((point) => (
                      <div key={point.uid} className="ml-8 mt-2 flex items-start gap-2">
                        <Badge variant="point" className="mt-0.5 shrink-0 text-xs">
                          Điểm {point.point_letter}
                        </Badge>
                        <p className="text-sm leading-relaxed text-foreground/80">
                          {point.content}
                        </p>
                      </div>
                    ))}
                  <Separator className="mt-3" />
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
