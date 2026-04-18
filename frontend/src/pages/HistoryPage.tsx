import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { History, Trash2, RotateCcw, Search } from 'lucide-react';
import { useSearchHistory } from '@/hooks/useSearchHistory';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

function formatDate(timestamp: number) {
  return new Intl.DateTimeFormat('vi-VN', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(timestamp));
}

export function HistoryPage() {
  const { entries, removeEntry, clearHistory } = useSearchHistory();
  const navigate = useNavigate();

  const handleRerun = useCallback(
    (query: string) => {
      // Navigate to home with query param — HomePage could pick it up via URLSearchParams
      navigate(`/?q=${encodeURIComponent(query)}`);
    },
    [navigate]
  );

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-4 px-4 py-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <History className="h-5 w-5 text-muted-foreground" />
          <h1 className="text-xl font-semibold">Lịch sử tra cứu</h1>
        </div>
        {entries.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground"
            onClick={clearHistory}
          >
            <Trash2 className="mr-1.5 h-4 w-4" />
            Xóa tất cả
          </Button>
        )}
      </div>

      {entries.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <Search className="mb-4 h-12 w-12 text-muted-foreground/40" />
          <h2 className="mb-2 text-lg font-medium">Chưa có lịch sử</h2>
          <p className="text-sm text-muted-foreground">
            Các câu hỏi bạn đã tra cứu sẽ xuất hiện ở đây.
          </p>
          <Button variant="outline" className="mt-4" onClick={() => navigate('/')}>
            Bắt đầu tra cứu
          </Button>
        </div>
      ) : (
        <div className="space-y-2">
          {entries.map((entry) => (
            <Card key={entry.id}>
              <CardContent className="flex items-center justify-between gap-3 p-4">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{entry.query}</p>
                  <div className="mt-1 flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">
                      {formatDate(entry.timestamp)}
                    </span>
                    <Badge variant="secondary" className="text-xs">
                      {entry.resultCount} kết quả
                    </Badge>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => handleRerun(entry.query)}
                    title="Chạy lại"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-destructive"
                    onClick={() => removeEntry(entry.id)}
                    title="Xóa"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
