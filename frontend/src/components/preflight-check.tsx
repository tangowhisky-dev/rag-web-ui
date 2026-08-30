"use client";

import { useState, useEffect, useCallback } from "react";
import { usePathname } from "next/navigation";
import { api } from "@/lib/api";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Settings } from "lucide-react";

interface PreflightIssue {
  key: string;
  label: string;
  category: string;
  severity: string;
  message: string;
  who_can_fix: string;
  scope: string;
  is_set: boolean;
}

interface PreflightResult {
  role: string;
  org_id: number | null;
  ok: boolean;
  issues: PreflightIssue[];
}

function computeSettingsHref(role: string, orgId: number | null): string | null {
  if (role === "super_admin") return "/dashboard/admin/settings";
  if (role === "admin") return `/dashboard/admin/orgs/${orgId}/settings`;
  return null;
}

function IssueCard({
  issue,
  variant,
}: {
  issue: PreflightIssue;
  variant: "error" | "warning";
}) {
  const isError = variant === "error";
  const cardClass = isError
    ? "rounded-md border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/30 p-3"
    : "rounded-md border border-amber-200 dark:border-amber-900 bg-amber-50 dark:bg-amber-950/30 p-3";
  const badgeClass = isError
    ? issue.who_can_fix === "super_admin"
      ? "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300"
      : "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300"
    : "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300";
  const badgeLabel = isError
    ? issue.who_can_fix === "super_admin"
      ? "Super Admin"
      : "Org Admin"
    : "Super Admin";

  return (
    <div key={issue.key} className={cardClass}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1">
          <p className="text-sm font-medium">{issue.label}</p>
          <p className="text-xs text-muted-foreground mt-0.5">{issue.message}</p>
        </div>
        <span
          className={`text-[10px] font-medium rounded px-1.5 py-0.5 shrink-0 ${badgeClass}`}
        >
          {badgeLabel}
        </span>
      </div>
    </div>
  );
}

function SettingsLink({
  href,
  onNavigate,
}: {
  href: string;
  onNavigate: () => void;
}) {
  return (
    <a href={href} onClick={onNavigate}>
      <Button size="sm">
        <Settings className="h-3.5 w-3.5 mr-1" />
        Go to Settings
      </Button>
    </a>
  );
}

function preflightDescription(hasErrors: boolean): string {
  return hasErrors
    ? "Some required settings are missing. The application will not work correctly until they are configured."
    : "Some optional settings are not configured. Ingestion features may be limited.";
}

export function PreflightCheck() {
  const [result, setResult] = useState<PreflightResult | null>(null);
  const [dismissed, setDismissed] = useState(false);
  const pathname = usePathname();

  const checkPreflight = useCallback(async () => {
    try {
      const data = await api.get("/api/auth/preflight") as PreflightResult;
      setResult(data);
    } catch {
      // Silently fail — don't block login over a preflight error
    }
  }, [setResult]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      await checkPreflight();
      if (cancelled) return;
    })();
    return () => { cancelled = true; };
  }, [checkPreflight]);

  // Auto-dismiss when navigating to any settings page
  useEffect(() => {
    if (pathname?.includes("/settings")) {
      // Defer to avoid synchronous setState in effect
      Promise.resolve().then(() => setDismissed(true));
    }
  }, [pathname]);

  const errorIssues = result?.issues.filter((i) => i.severity === "error") ?? [];
  const warningIssues = result?.issues.filter((i) => i.severity === "warning") ?? [];
  const hasIssues = errorIssues.length > 0 || warningIssues.length > 0;

  if (!result || !hasIssues) return null;

  const isOpen = hasIssues && !dismissed;
  const settingsHref = computeSettingsHref(result.role, result.org_id);
  const isAdmin = result.role === "admin" || result.role === "super_admin";

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && setDismissed(true)}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-amber-500" />
            Configuration Required
          </DialogTitle>
          <DialogDescription>
            {preflightDescription(errorIssues.length > 0)}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 py-2 max-h-[40vh] overflow-y-auto">
          {errorIssues.map((issue) => (
            <IssueCard key={issue.key} issue={issue} variant="error" />
          ))}

          {warningIssues.map((issue) => (
            <IssueCard key={issue.key} issue={issue} variant="warning" />
          ))}
        </div>

        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={() => setDismissed(true)}>
            Dismiss
          </Button>
          {isAdmin && settingsHref && (
            <SettingsLink href={settingsHref} onNavigate={() => setDismissed(true)} />
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
