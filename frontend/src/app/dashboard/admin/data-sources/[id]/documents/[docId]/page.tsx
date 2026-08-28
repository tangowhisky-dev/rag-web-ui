"use client";

import "@uiw/react-md-editor/markdown-editor.css";
import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { useTheme } from "next-themes";
import { api, ApiError } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { useToast } from "@/components/ui/use-toast";
import {
  ArrowLeft,
  Save,
  RotateCcw,
  RefreshCw,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

// react-md-editor is a client-only component — load it dynamically
// to avoid SSR issues with Next.js.
const MDEditor = dynamic(() => import("@uiw/react-md-editor"), { ssr: false });

interface MarkdownResponse {
  document_id: number;
  markdown: string;
  conversion_status: string | null;
  lock_version: number;
  title: string | null;
}

interface IngestStatusResponse {
  document_id: number;
  conversion_status: string | null;
  conversion_error: string | null;
  ingest_status: string | null;
  ingest_progress: number;
  ingest_message: string | null;
  ingest_error: string | null;
  graph_status: string | null;
  graph_error: string | null;
  chunk_count: number;
  lock_version: number;
}

type PhaseState = "idle" | "pending" | "processing" | "completed" | "failed" | "error";

function PhaseDot({ state, label }: { state: PhaseState; label: string }) {
  const colors: Record<PhaseState, string> = {
    idle: "bg-muted-foreground/30",
    pending: "bg-yellow-500",
    processing: "bg-blue-500 animate-pulse",
    completed: "bg-green-500",
    failed: "bg-red-500",
    error: "bg-red-500",
  };
  const icons: Record<PhaseState, React.ReactNode> = {
    idle: null,
    pending: <AlertCircle className="h-3 w-3" />,
    processing: <Loader2 className="h-3 w-3 animate-spin" />,
    completed: <CheckCircle2 className="h-3 w-3" />,
    failed: <XCircle className="h-3 w-3" />,
    error: <XCircle className="h-3 w-3" />,
  };
  return (
    <div className="flex items-center gap-1.5">
      <div className={cn("h-2.5 w-2.5 rounded-full", colors[state])} />
      {icons[state] && <span className="text-muted-foreground">{icons[state]}</span>}
      <span className="text-xs text-muted-foreground">{label}</span>
    </div>
  );
}

export default function DocumentEditorPage() {
  const params = useParams();
  const router = useRouter();
  const { toast } = useToast();
  const { resolvedTheme } = useTheme();
  const datastoreId = params!.id as string;
  const docId = params!.docId as string;

  const [markdown, setMarkdown] = useState("");
  const [originalMarkdown, setOriginalMarkdown] = useState("");
  const [lockVersion, setLockVersion] = useState(0);
  const [title, setTitle] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [reconverting, setReconverting] = useState(false);
  const [status, setStatus] = useState<IngestStatusResponse | null>(null);
  const [showUnsavedDialog, setShowUnsavedDialog] = useState(false);
  const [pendingNavigation, setPendingNavigation] = useState<string | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const isDirty = markdown !== originalMarkdown;

  // Load markdown
  const loadMarkdown = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = (await api.get(
        `/api/admin/datastores/${datastoreId}/documents/${docId}/markdown`,
      )) as MarkdownResponse;
      setMarkdown(resp.markdown);
      setOriginalMarkdown(resp.markdown);
      setLockVersion(resp.lock_version);
      setTitle(resp.title);
    } catch (e) {
      if (e instanceof ApiError) {
        if (e.status === 409) {
          setError(e.message + " — poll ingest-status to track progress.");
        } else if (e.status === 422) {
          setError(e.message);
        } else {
          setError(e.message);
        }
      } else {
        setError("Failed to load markdown");
      }
    } finally {
      setLoading(false);
    }
  }, [datastoreId, docId]);

  // Poll ingest status
  const pollStatus = useCallback(async () => {
    try {
      const resp = (await api.get(
        `/api/admin/datastores/${datastoreId}/documents/${docId}/ingest-status`,
      )) as IngestStatusResponse;
      setStatus(resp);
      // Stop polling when everything is done (or failed)
      const active =
        resp.conversion_status === "pending" ||
        resp.conversion_status === "processing" ||
        resp.ingest_status === "pending" ||
        resp.ingest_status === "processing" ||
        resp.graph_status === "pending";
      if (!active && pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    } catch {
      // ignore polling errors
    }
  }, [datastoreId, docId]);

  useEffect(() => {
    loadMarkdown();
    pollStatus();
    pollRef.current = setInterval(pollStatus, 2000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [loadMarkdown, pollStatus]);

  // beforeunload guard
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [isDirty]);

  // In-app navigation guard
  const handleBack = useCallback(() => {
    if (isDirty) {
      setShowUnsavedDialog(true);
      setPendingNavigation("back");
    } else {
      router.push(`/dashboard/admin/data-sources/${datastoreId}`);
    }
  }, [isDirty, router, datastoreId]);

  const confirmNavigation = useCallback(() => {
    setShowUnsavedDialog(false);
    if (pendingNavigation === "back") {
      router.push(`/dashboard/admin/data-sources/${datastoreId}`);
    }
    setPendingNavigation(null);
  }, [pendingNavigation, router, datastoreId]);

  // Save
  const handleSave = useCallback(async () => {
    if (!markdown.trim()) {
      toast({ title: "Cannot save empty markdown", variant: "destructive" });
      return;
    }
    setSaving(true);
    try {
      const resp = (await api.put(
        `/api/admin/datastores/${datastoreId}/documents/${docId}/markdown`,
        { markdown, lock_version: lockVersion },
      )) as any;
      setLockVersion(resp.lock_version);
      setOriginalMarkdown(markdown);
      toast({ title: "Markdown saved", description: "File will be re-ingested on next process cycle." });
      // Start polling
      if (!pollRef.current) {
        pollStatus();
        pollRef.current = setInterval(pollStatus, 2000);
      }
    } catch (e) {
      if (e instanceof ApiError) {
        toast({ title: "Save failed", description: e.message, variant: "destructive" });
      } else {
        toast({ title: "Save failed", variant: "destructive" });
      }
    } finally {
      setSaving(false);
    }
  }, [markdown, lockVersion, datastoreId, docId, toast, pollStatus]);

  // Discard changes — revert to original markdown
  const handleDiscard = useCallback(() => {
    setMarkdown(originalMarkdown);
  }, [originalMarkdown]);

  // Re-convert
  const handleReconvert = useCallback(async () => {
    setReconverting(true);
    try {
      await api.post(
        `/api/admin/datastores/${datastoreId}/documents/${docId}/reconvert`,
      );
      toast({
        title: "Re-convert queued",
        description: "Current edits will be overwritten when conversion completes.",
      });
      // Start polling
      if (!pollRef.current) {
        pollStatus();
        pollRef.current = setInterval(pollStatus, 2000);
      }
    } catch (e) {
      if (e instanceof ApiError) {
        toast({ title: "Re-convert failed", description: e.message, variant: "destructive" });
      } else {
        toast({ title: "Re-convert failed", variant: "destructive" });
      }
    } finally {
      setReconverting(false);
    }
  }, [datastoreId, docId, toast, pollStatus]);

  if (loading) {
    return (
      <div className="flex h-[calc(100vh-4rem)] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error && !markdown) {
    return (
      <div className="flex h-[calc(100vh-4rem)] flex-col items-center justify-center gap-4">
        <XCircle className="h-8 w-8 text-destructive" />
        <p className="text-sm text-muted-foreground max-w-md text-center">{error}</p>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleBack}>
            <ArrowLeft className="mr-2 h-4 w-4" /> Back
          </Button>
          <Button onClick={loadMarkdown}>Retry</Button>
        </div>
      </div>
    );
  }

  const conversionState: PhaseState =
    status?.conversion_status === "completed" ? "completed" :
    status?.conversion_status === "processing" ? "processing" :
    status?.conversion_status === "pending" ? "pending" :
    status?.conversion_status === "error" ? "error" : "idle";

  const ingestState: PhaseState =
    status?.ingest_status === "completed" ? "completed" :
    status?.ingest_status === "processing" ? "processing" :
    status?.ingest_status === "pending" ? "pending" :
    status?.ingest_status === "failed" ? "failed" : "idle";

  const graphState: PhaseState =
    status?.graph_status === "completed" ? "completed" :
    status?.graph_status === "pending" ? "pending" :
    status?.graph_status === "failed" ? "failed" : "idle";

  const isIngestActive =
    conversionState === "processing" || conversionState === "pending" ||
    ingestState === "processing" || ingestState === "pending";

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col">
      {/* Header */}
      <div className="border-b px-4 py-3">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <Button variant="ghost" size="sm" onClick={handleBack} className="h-7 px-2">
              <ArrowLeft className="mr-1 h-3.5 w-3.5" />
              Back
            </Button>
            <div className="min-w-0">
              <h1 className="text-sm font-medium truncate">{title || `Document ${docId}`}</h1>
              <p className="text-xs text-muted-foreground truncate">doc_id={docId}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleReconvert}
              disabled={isIngestActive || reconverting}
              className="h-7"
            >
              {reconverting ? (
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              ) : (
                <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
              )}
              Re-convert
            </Button>
          </div>
        </div>

        {/* Phase status bar — reserve height for action buttons to prevent wobble */}
        <div className="mt-3 flex items-center gap-6 min-h-[28px]">
          <PhaseDot state={conversionState} label="Convert" />
          <PhaseDot state={ingestState} label="Ingest" />
          <PhaseDot state={graphState} label="Graph" />
          {status && status.chunk_count > 0 && (
            <Badge variant="secondary" className="text-xs">
              {status.chunk_count} chunks
            </Badge>
          )}
          <div className="flex items-center gap-2 ml-auto h-7">
            {isDirty && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDiscard}
                  disabled={saving || isIngestActive}
                  className="h-7 px-2.5"
                >
                  <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
                  Discard
                </Button>
                <Button
                  size="sm"
                  onClick={handleSave}
                  disabled={saving || isIngestActive}
                  className="h-7 px-2.5"
                >
                  {saving ? (
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Save className="mr-1.5 h-3.5 w-3.5" />
                  )}
                  Save
                </Button>
              </>
            )}
          </div>
        </div>

        {/* Progress bar */}
        {status && status.ingest_progress > 0 && status.ingest_progress < 100 && (
          <div className="mt-2 flex items-center gap-2">
            <Progress value={status.ingest_progress} className="h-1.5 flex-1" />
            <span className="text-xs text-muted-foreground whitespace-nowrap">
              {status.ingest_message || `${status.ingest_progress}%`}
            </span>
          </div>
        )}

        {/* Error messages */}
        {status?.conversion_error && (
          <p className="mt-2 text-xs text-destructive">
            Convert error: {status.conversion_error}
          </p>
        )}
        {status?.ingest_error && (
          <p className="mt-2 text-xs text-destructive">
            Ingest error: {status.ingest_error}
          </p>
        )}
        {status?.graph_error && (
          <p className="mt-2 text-xs text-destructive">
            Graph error: {status.graph_error}
          </p>
        )}
      </div>

      {/* Editor with built-in live preview */}
      <div className="flex-1 overflow-hidden" data-color-mode={resolvedTheme === "dark" ? "dark" : "light"}>
        <MDEditor
          value={markdown}
          onChange={(val) => setMarkdown(val ?? "")}
          preview="live"
          height="100%"
          style={{ height: "100%", borderRadius: 0 }}
          hideToolbar={false}
        />
      </div>

      {/* Unsaved changes dialog */}
      <ConfirmDialog
        open={showUnsavedDialog}
        title="Unsaved changes"
        description="You have unsaved edits. Leave without saving?"
        confirmText="Discard & leave"
        cancelText="Stay"
        destructive
        onConfirm={confirmNavigation}
        onCancel={() => {
          setShowUnsavedDialog(false);
          setPendingNavigation(null);
        }}
      />
    </div>
  );
}
