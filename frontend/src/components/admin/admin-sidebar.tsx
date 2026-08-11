'use client';

import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { useSidebarCollapse } from "@/lib/hooks";
import {
  PanelLeftClose, PanelLeftOpen,
  Building2, Users, Database, Settings,
} from 'lucide-react';

interface AdminSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  userRole?: string;
}

const NAV_ITEMS = [
  { label: 'Orgs', href: '/dashboard/admin/orgs', icon: Building2 },
  { label: 'Users', href: '/dashboard/admin/users', icon: Users },
  { label: 'Data Stores', href: '/dashboard/admin/data-sources', icon: Database },
];

const SUPER_ADMIN_ITEMS = [
  { label: 'Settings', href: '/dashboard/admin/settings', icon: Settings },
];

// LLM config is managed per-organisation on the Orgs page (LLM Config button).
// The standalone /dashboard/admin/llm-config page does not exist.


export default function AdminSidebar({ isOpen, onClose, userRole }: AdminSidebarProps) {
  const pathname = usePathname();
  const { collapsed, toggleCollapse } = useSidebarCollapse("admin-sidebar-collapsed");

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/40 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={[
          'fixed inset-y-0 left-0 z-30 bg-card border-r flex flex-col',
          'transition-all duration-200 ease-in-out',
          'lg:relative lg:inset-auto lg:translate-x-0 lg:z-auto lg:h-full',
          collapsed ? 'w-12' : 'w-64',
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
        ].join(' ')}
      >
        {/* ── Collapsed ─── */}
        {collapsed && (
          <div className="flex flex-col items-center gap-2 py-3 flex-1">
            <button
              onClick={toggleCollapse}
              className="p-2 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
              aria-label="Expand sidebar"
            >
              <PanelLeftOpen className="h-4 w-4" />
            </button>
            <div className="w-5 h-px bg-border" />
            {NAV_ITEMS.map(({ href, icon: Icon }) => {
              const isActive = pathname === href || pathname.startsWith(href + '/');
              return (
                <Link
                  key={href}
                  href={href}
                  onClick={onClose}
                  className={[
                    'p-2 rounded-lg transition-colors',
                    isActive
                      ? 'bg-accent text-accent-foreground'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted',
                  ].join(' ')}
                  aria-label={href}
                  title={href}
                >
                  <Icon className="h-4 w-4" />
                </Link>
              );
            })}
            {userRole === 'super_admin' && SUPER_ADMIN_ITEMS.map(({ href, icon: Icon }) => {
              const isActive = pathname === href || pathname.startsWith(href + '/');
              return (
                <Link
                  key={href}
                  href={href}
                  onClick={onClose}
                  className={[
                    'p-2 rounded-lg transition-colors',
                    isActive
                      ? 'bg-accent text-accent-foreground'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted',
                  ].join(' ')}
                  aria-label={href}
                  title={href}
                >
                  <Icon className="h-4 w-4" />
                </Link>
              );
            })}
          </div>
        )}

        {/* ── Expanded ─── */}
        {!collapsed && (
          <>
            {/* Header */}
            <div className="flex items-center gap-1 px-3 pt-3 pb-2 shrink-0">
              <span className="flex-1 text-sm font-semibold truncate">Admin</span>
              <button
                onClick={toggleCollapse}
                className="hidden lg:flex p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground shrink-0"
                aria-label="Collapse sidebar"
              >
                <PanelLeftClose className="h-4 w-4" />
              </button>
              <button
                onClick={onClose}
                className="lg:hidden p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground shrink-0"
                aria-label="Close sidebar"
              >
                <PanelLeftClose className="h-4 w-4" />
              </button>
            </div>

            {/* Nav */}
            <nav className="flex-1 overflow-y-auto px-2 space-y-0.5">
              {NAV_ITEMS.map(({ label, href, icon: Icon }) => {
                const isActive = pathname === href || pathname.startsWith(href + '/');
                return (
                  <Link
                    key={href}
                    href={href}
                    onClick={onClose}
                    className={[
                      'flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm transition-colors',
                      isActive
                        ? 'bg-accent text-accent-foreground'
                        : 'hover:bg-accent/60 text-foreground',
                    ].join(' ')}
                  >
                    <Icon className="h-3.5 w-3.5 shrink-0 opacity-50" />
                    <span className="truncate">{label}</span>
                  </Link>
                );
              })}
              {userRole === 'super_admin' && (
                <div className="pt-2 mt-2 border-t">
                  {SUPER_ADMIN_ITEMS.map(({ label, href, icon: Icon }) => {
                    const isActive = pathname === href || pathname.startsWith(href + '/');
                    return (
                      <Link
                        key={href}
                        href={href}
                        onClick={onClose}
                        className={[
                          'flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm transition-colors',
                          isActive
                            ? 'bg-accent text-accent-foreground'
                            : 'hover:bg-accent/60 text-foreground',
                        ].join(' ')}
                      >
                        <Icon className="h-3.5 w-3.5 shrink-0 opacity-50" />
                        <span className="truncate">{label}</span>
                      </Link>
                    );
                  })}
                </div>
              )}
            </nav>
          </>
        )}
      </aside>
    </>
  );
}
