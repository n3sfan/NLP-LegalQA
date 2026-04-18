import { uidToBreadcrumb } from '@/lib/uid';
import { Separator } from '@/components/ui/separator';

interface HierarchyBreadcrumbProps {
  uid: string;
  docIdentity?: string;
}

export function HierarchyBreadcrumb({ uid, docIdentity }: HierarchyBreadcrumbProps) {
  const crumbs = uidToBreadcrumb(uid);

  if (!crumbs.length && !docIdentity) return null;

  return (
    <nav className="flex flex-wrap items-center gap-1 text-xs text-muted-foreground">
      {docIdentity && (
        <span className="rounded bg-muted px-1.5 py-0.5 font-mono">{docIdentity}</span>
      )}
      {docIdentity && crumbs.length > 0 && <Separator orientation="vertical" className="h-3" />}
      {crumbs.map((crumb, i) => (
        <span key={i} className="flex items-center gap-1">
          <span className="font-medium text-foreground/70">{crumb.label}</span>
          <span className="font-semibold text-foreground">{crumb.value}</span>
          {i < crumbs.length - 1 && <span className="ml-1 text-muted-foreground/50">›</span>}
        </span>
      ))}
    </nav>
  );
}
