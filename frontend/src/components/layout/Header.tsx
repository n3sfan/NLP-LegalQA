import { Link, useLocation } from 'react-router-dom';
import { Scale, History, Home } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export function Header() {
  const location = useLocation();

  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto flex h-14 max-w-4xl items-center justify-between px-4">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2 font-semibold text-foreground">
          <Scale className="h-5 w-5 text-primary" />
          <span>Pháp Luật QA</span>
        </Link>

        {/* Nav */}
        <nav className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            asChild
            className={cn(
              'gap-1.5',
              location.pathname === '/' && 'bg-muted'
            )}
          >
            <Link to="/">
              <Home className="h-4 w-4" />
              Tra cứu
            </Link>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            asChild
            className={cn(
              'gap-1.5',
              location.pathname === '/history' && 'bg-muted'
            )}
          >
            <Link to="/history">
              <History className="h-4 w-4" />
              Lịch sử
            </Link>
          </Button>
        </nav>
      </div>
    </header>
  );
}
