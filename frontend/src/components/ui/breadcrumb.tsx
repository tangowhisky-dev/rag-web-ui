"use client";

import { ChevronRight, Home } from "lucide-react";
import { APP_NAME, APP_ICON_SRC, APP_VERSION } from "@/lib/app-config";
import Link from "next/link";
import { usePathname } from "next/navigation";

interface BreadcrumbProps {
  overrideLastLabel?: string;
}

const Breadcrumb = ({ overrideLastLabel }: BreadcrumbProps) => {
  const pathname = usePathname();

  const generateBreadcrumbs = () => {
    const paths = pathname.split("/").filter(Boolean);
    return paths.map((path, index) => {
      const href = "/" + paths.slice(0, index + 1).join("/");
      const label =
        path.charAt(0).toUpperCase() + path.slice(1).replace(/-/g, " ");
      const isLast = index === paths.length - 1;
      const displayLabel = path.match(/^\[.*\]$/) ? "Details" : label;
      return {
        href,
        label: isLast && overrideLastLabel ? overrideLastLabel : displayLabel,
        isLast,
      };
    });
  };

  const breadcrumbs = generateBreadcrumbs();

  if (pathname === "/") return null;

  return (
    <nav className="flex items-center space-x-1 text-sm text-muted-foreground">
      {/* Home button: logo + app name on left, Home icon on right */}
      <Link
        href="/dashboard"
        className="flex items-center gap-2 rounded-lg px-2 py-1 hover:bg-accent hover:text-foreground transition-colors"
      >
        <img src={APP_ICON_SRC} alt={APP_NAME} className="h-5 w-5 rounded" />
        <div className="flex flex-col leading-none">
          <span className="font-semibold text-foreground hidden sm:inline">{APP_NAME}</span>
          <span className="text-[9px] text-muted-foreground/60 hidden sm:inline">v{APP_VERSION}</span>
        </div>
        <Home className="h-3.5 w-3.5" />
      </Link>

      {breadcrumbs.map((breadcrumb) => (
        <div key={breadcrumb.href} className="flex items-center">
          <ChevronRight className="h-3.5 w-3.5 mx-0.5 text-muted-foreground/50" />
          {breadcrumb.isLast ? (
            <span className="px-1 text-foreground font-medium">
              {breadcrumb.label}
            </span>
          ) : (
            <Link
              href={breadcrumb.href}
              className="px-1 hover:text-foreground transition-colors"
            >
              {breadcrumb.label}
            </Link>
          )}
        </div>
      ))}
    </nav>
  );
};

export default Breadcrumb;
