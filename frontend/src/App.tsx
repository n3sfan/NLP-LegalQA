import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Toaster } from '@/components/ui/sonner';
import { Layout } from '@/components/layout/Layout';
import { HomePage } from '@/pages/HomePage';
import { HistoryPage } from '@/pages/HistoryPage';
import { ArticlePage } from '@/pages/ArticlePage';

export default function App() {
  return (
    <BrowserRouter>
      <TooltipProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route
              path="/article/:doc_identity/:article_num"
              element={<ArticlePage />}
            />
          </Routes>
        </Layout>
        <Toaster />
      </TooltipProvider>
    </BrowserRouter>
  );
}
