import { Scale } from 'lucide-react';

interface EmptyStateProps {
  query?: string;
}

export function EmptyState({ query }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <Scale className="mb-4 h-12 w-12 text-muted-foreground/40" />
      <h2 className="mb-2 text-lg font-medium text-foreground">
        {query ? 'Không tìm thấy kết quả' : 'Hỏi về pháp luật Việt Nam'}
      </h2>
      <p className="max-w-sm text-sm text-muted-foreground">
        {query
          ? `Không tìm thấy kết quả nào cho "${query}". Hãy thử diễn đạt câu hỏi theo cách khác.`
          : 'Nhập câu hỏi về pháp luật Việt Nam để bắt đầu tra cứu. Hệ thống sẽ phân tích, tìm kiếm và trả về các điều khoản pháp luật liên quan.'}
      </p>
    </div>
  );
}
