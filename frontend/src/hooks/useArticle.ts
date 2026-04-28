/**
 * Fetch a single article by doc_identity + article_num.
 * Uses mock by default — swap mockGetArticle → getArticle from ./client
 */

import { useState, useEffect } from 'react';
import type { ArticleApiResponse } from '@/api/types';
import { mockGetArticle } from '@/api/mock';

interface UseArticleOptions {
  doc_identity: string;
  article_num: number;
}

export function useArticle({ doc_identity, article_num }: UseArticleOptions) {
  const [data, setData] = useState<ArticleApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!doc_identity || !article_num) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    mockGetArticle(doc_identity, article_num)
      .then((result) => {
        if (!cancelled) {
          setData(result);
          setLoading(false);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, [doc_identity, article_num]);

  return { data, loading, error };
}
