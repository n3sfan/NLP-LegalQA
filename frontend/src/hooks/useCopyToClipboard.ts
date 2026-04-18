/**
 * Copy text to clipboard with success/error feedback via Sonner toast.
 */

import { useState, useCallback } from 'react';
import { toast } from 'sonner';

export function useCopyToClipboard() {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async (text: string, label = 'Đã sao chép') => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      toast.success(label);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error('Không thể sao chép');
    }
  }, []);

  return { copy, copied };
}
