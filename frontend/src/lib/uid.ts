/**
 * UID parser for Vietnamese legal document hierarchy UIDs.
 * Format: {doc_identity}::{level}::{...}
 * Example: "56/2024/QH15::article::10::clause::1::point::a"
 */

export type HierarchyLevel = 'document' | 'part' | 'chapter' | 'section' | 'article' | 'clause' | 'point';

export interface ParsedUID {
  docIdentity: string;
  level: HierarchyLevel;
  part?: string;
  chapter?: string;
  section?: string;
  article?: string;
  clause?: string;
  point?: string;
}

export function parseUID(uid: string): ParsedUID {
  const parts = uid.split('::');
  const docIdentity = parts[0];
  const level = (parts[1] ?? 'document') as HierarchyLevel;

  const result: ParsedUID = { docIdentity, level };

  switch (level) {
    case 'part':
      result.part = parts[2];
      break;
    case 'chapter':
      if (parts[2] === 'part') {
        result.part = parts[3];
        result.chapter = parts[5];
      } else {
        result.chapter = parts[2];
      }
      break;
    case 'section':
      if (parts[2] === 'part')    result.part     = parts[3];
      if (parts[4] === 'chapter')  result.chapter  = parts[5];
      result.section = parts[parts.length - 1];
      break;
    case 'article':
      result.article = parts[2];
      break;
    case 'clause':
      result.article = parts[2];
      result.clause  = parts[4];
      break;
    case 'point':
      result.article = parts[2];
      result.clause  = parts[4];
      result.point  = parts[6];
      break;
  }

  return result;
}

export const LEVEL_LABELS: Record<HierarchyLevel, string> = {
  document: 'Văn bản',
  part:     'Phần',
  chapter:  'Chương',
  section:  'Mục',
  article:  'Điều',
  clause:   'Khoản',
  point:    'Điểm',
};

export function levelLabel(level: HierarchyLevel): string {
  return LEVEL_LABELS[level] ?? level;
}

/** "Điều 10 › Khoản 1 › Điểm a" chain from UID */
export interface BreadcrumbItem {
  label: string;
  value: string;
}

export function uidToBreadcrumb(uid: string): BreadcrumbItem[] {
  const parsed = parseUID(uid);
  const crumbs: BreadcrumbItem[] = [];

  if (parsed.article) crumbs.push({ label: 'Điều',  value: parsed.article });
  if (parsed.clause)  crumbs.push({ label: 'Khoản', value: parsed.clause });
  if (parsed.point)   crumbs.push({ label: 'Điểm',  value: parsed.point });

  return crumbs;
}

/** Color class map for node label badges */
export const LABEL_COLOR_MAP: Record<string, string> = {
  Article:  'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200 border-orange-300 dark:border-orange-700',
  Clause:   'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200 border-purple-300 dark:border-purple-700',
  Point:    'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 border-blue-300 dark:border-blue-700',
  Chapter:  'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200 border-amber-300 dark:border-amber-700',
  Section:  'bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200 border-teal-300 dark:border-teal-700',
  Part:     'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 border-green-300 dark:border-green-700',
  Document: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200 border-gray-300 dark:border-gray-600',
};