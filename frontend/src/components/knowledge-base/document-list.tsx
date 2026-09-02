"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { formatDistanceToNow } from "date-fns";
import { api, ApiError } from "@/lib/api";
import { FileIcon, defaultStyles } from "react-file-icon";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { FileText, Trash2, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { LoadingDots } from "@/components/ui/loading-dots";

interface ProcessingTask {
  id: number;
  status: string;
  error_message: string | null;
  progress?: number;
  progress_message?: string;
  graph_status?: string | null;
  graph_error?: string | null;
  graph_progress?: number | null;
  graph_progress_message?: string | null;
}

interface Document {
  id: number;
  file_name: string;
  file_path: string;
  file_size: number;
  content_type: string;
  created_at: string;
  processing_tasks: ProcessingTask[];
}

interface DocumentListProps {
  knowledgeBaseId: number;
  refreshKey?: number;
}

const POLL_INTERVAL = 3000;
const POLL_RETRY_DELAY = 8000;

function isRunningTask(t: ProcessingTask): boolean {
  return t.status === "pending" || t.status === "processing" || t.graph_status === "pending";
}

function hasTerminalStatus(info: ProcessingTask): boolean {
  return info.status === "completed" || info.status === "failed";
}

function collectPendingTaskIds(
  docs: Document[],
  taskProgressRef: { current: Record<number, ProcessingTask> }
): number[] {
  const taskIds: number[] = [];
  for (const doc of docs) {
    for (const t of doc.processing_tasks) {
      if (isRunningTask(t)) taskIds.push(t.id);
    }
  }

  // Also poll any task we're tracking in taskProgress that's still running
  // (handles the case where the dialog was closed before the KB list refreshed)
  for (const [idStr, t] of Object.entries(taskProgressRef.current)) {
    if (isRunningTask(t)) {
      const id = parseInt(idStr);
      if (!taskIds.includes(id)) taskIds.push(id);
    }
  }
  return taskIds;
}

function processTaskResponse(
  resp: Record<string, ProcessingTask>,
  setTaskProgress: (updater: (prev: Record<number, ProcessingTask>) => Record<number, ProcessingTask>) => void
): { anyRunning: boolean; anyFinished: boolean } {
  const updates: Record<number, ProcessingTask> = {};
  let anyRunning = false;
  let anyFinished = false;
  for (const [idStr, info] of Object.entries(resp)) {
    const id = parseInt(idStr);
    updates[id] = info;
    if (isRunningTask(info)) anyRunning = true;
    if (hasTerminalStatus(info)) anyFinished = true;
  }
  setTaskProgress((prev) => ({ ...prev, ...updates }));
  return { anyRunning, anyFinished };
}

export function DocumentList({ knowledgeBaseId, refreshKey }: DocumentListProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [confirmDoc, setConfirmDoc] = useState<Document | null>(null);
  // live progress keyed by task id — overlays the static data from KB GET
  const [taskProgress, setTaskProgress] = useState<Record<number, ProcessingTask>>({});
  const taskProgressRef = useRef(taskProgress);
  useEffect(() => { taskProgressRef.current = taskProgress; }, [taskProgress]);
  const { toast } = useToast();

  // Stable refs so the poll loop never has stale closures
  const docsRef = useRef<Document[]>([]);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const knowledgeBaseIdRef = useRef(knowledgeBaseId);
  useEffect(() => {
    knowledgeBaseIdRef.current = knowledgeBaseId;
  }, [knowledgeBaseId]);

  const clearPoll = () => {
    if (pollTimerRef.current) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const fetchDocuments = useCallback(async () => {
    try {
      const data = await api.get(`/api/knowledge-base/${knowledgeBaseIdRef.current}`);
      const docs: Document[] = data.documents;
      docsRef.current = docs;
      setDocuments(docs);
      return docs;
    } catch (err) {
      if (err instanceof ApiError) setError(err.message);
      else setError("Failed to fetch documents");
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // The poll loop — reads from refs so it never captures stale state
  const schedulePoll = useCallback(() => {
    clearPoll();

    const run = async () => {
      const docs = docsRef.current;
      const taskIds = collectPendingTaskIds(docs, taskProgressRef);
      if (taskIds.length === 0) return; // nothing in progress, stop polling

      try {
        const resp = await api.get(
          `/api/knowledge-base/${knowledgeBaseIdRef.current}/documents/tasks?task_ids=${taskIds.join(",")}`
        );
        const { anyRunning, anyFinished } = processTaskResponse(
          resp as Record<string, ProcessingTask>,
          setTaskProgress
        );

        if (anyFinished) {
          // Refresh the doc list so status badges update; restart poll after
          const fresh = await fetchDocuments();
          if (fresh) docsRef.current = fresh;
        }

        if (anyRunning) {
          pollTimerRef.current = setTimeout(run, POLL_INTERVAL);
        }
      } catch {
        // Network blip — back off and retry
        pollTimerRef.current = setTimeout(run, POLL_RETRY_DELAY);
      }
    };

    pollTimerRef.current = setTimeout(run, POLL_INTERVAL);
  }, [fetchDocuments, setTaskProgress]);

  // Initial load + reload when refreshKey changes
  useEffect(() => {
    clearPoll();
    let cancelled = false;
    (async () => {
      // Set loading before fetch — use a microtask to avoid synchronous setState in effect
      await Promise.resolve();
      if (cancelled) return;
      setLoading(true);
      const docs = await fetchDocuments();
      if (cancelled || !docs) return;
      docsRef.current = docs;
      const hasInProgress = docs.some((d) =>
        d.processing_tasks.some(
          (t) =>
            t.status === "pending" ||
            t.status === "processing" ||
            t.graph_status === "pending"
        )
      );
      if (hasInProgress) schedulePoll();
    })();
    return () => { cancelled = true; clearPoll(); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [knowledgeBaseId, refreshKey]);

  const handleDelete = async (doc: Document) => {
    setDeletingId(doc.id);
    try {
      await api.delete(`/api/knowledge-base/${knowledgeBaseId}/documents/${doc.id}`);
      setDocuments((prev) => prev.filter((d) => d.id !== doc.id));
      toast({ title: "Document deleted", description: `"${doc.file_name}" has been removed.` });
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Failed to delete document";
      toast({ title: "Delete failed", description: msg, variant: "destructive" });
    } finally {
      setDeletingId(null);
    }
  };

  const getTaskDisplay = (doc: Document): ProcessingTask | null => {
    const task = doc.processing_tasks[0];
    if (!task) return null;
    const live = taskProgress[task.id];
    // Merge: prefer live status/progress but fall back to static if no live data yet
    if (!live) return task;
    return { ...task, ...live };
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center p-8">
        <LoadingDots label="Loading documents" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center p-8">
        <p className="text-destructive">{error}</p>
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] p-8">
        <div className="flex flex-col items-center max-w-[420px] text-center space-y-6">
          <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center">
            <FileText className="w-10 h-10 text-muted-foreground" />
          </div>
          <div className="space-y-2">
            <h3 className="text-xl font-semibold">No documents</h3>
            <p className="text-muted-foreground">
              Upload document(s) to build/ enhance your knowledge base.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Size</TableHead>
          <TableHead>Created</TableHead>
          <TableHead>Status</TableHead>
          <TableHead className="w-16" />
        </TableRow>
      </TableHeader>
      <TableBody>
        {documents.map((doc) => {
          const taskDisplay = getTaskDisplay(doc);
          const isInProgress =
            taskDisplay?.status === "pending" || taskDisplay?.status === "processing";
          return (
            <TableRow key={doc.id}>
              <TableCell className="font-medium">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6">
                    {(() => {
                      const ext = (doc.file_name.split(".").pop() || "").toLowerCase();
                      const style = defaultStyles[ext as keyof typeof defaultStyles];
                      return style
                        ? <FileIcon extension={ext} {...style} />
                        : <FileIcon extension={ext} color="#E2E8F0" labelColor="#94A3B8" />;
                    })()}
                  </div>
                  {doc.file_name}
                </div>
              </TableCell>
              <TableCell>{(doc.file_size / 1024 / 1024).toFixed(2)} MB</TableCell>
              <TableCell>
                {formatDistanceToNow(new Date(doc.created_at), { addSuffix: true })}
              </TableCell>
              <TableCell className="min-w-[200px]">
                {taskDisplay && (
                  <div className="space-y-1">
                    <div className="flex items-center gap-1.5">
                      <Badge
                        variant={
                          taskDisplay.status === "completed"
                            ? "secondary"
                            : taskDisplay.status === "failed"
                            ? "destructive"
                            : "default"
                        }
                      >
                        {taskDisplay.status}
                      </Badge>
                      {taskDisplay.graph_status === "pending" && (
                        <span title={taskDisplay.graph_progress_message || "Building knowledge graph"}>
                          <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                        </span>
                      )}
                      {taskDisplay.graph_status === "completed" && (
                        <span title="Knowledge graph built">
                          <CheckCircle2 className="h-3.5 w-3.5 text-muted-foreground" />
                        </span>
                      )}
                      {taskDisplay.graph_status === "failed" && (
                        <span title={taskDisplay.graph_error || "Graph extraction failed"}>
                          <AlertCircle className="h-3.5 w-3.5 text-amber-500" />
                        </span>
                      )}
                    </div>
                    {isInProgress && (
                      <div className="space-y-0.5">
                        <Progress
                          value={taskDisplay.progress ?? 5}
                          className="h-1.5 w-full"
                        />
                        {taskDisplay.progress_message && (
                          <p className="text-xs text-muted-foreground truncate max-w-[220px]">
                            {taskDisplay.progress_message}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </TableCell>
              <TableCell>
                <Button
                  variant="ghost"
                  size="icon"
                  disabled={deletingId === doc.id || !!isInProgress}
                  onClick={() => setConfirmDoc(doc)}
                  className="h-8 w-8 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                  title={isInProgress ? "Cannot delete while processing" : "Delete document"}
                >
                  {deletingId === doc.id ? (
                    <span
                      className="w-3 h-3 rounded-full bg-current"
                      style={{ animation: "loading-dot-pulse 1.4s ease-in-out infinite" }}
                    />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                </Button>
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
      <ConfirmDialog
        open={confirmDoc !== null}
        title="Delete document"
        description={`Delete "${confirmDoc?.file_name}"? This will remove the file and all its indexed chunks.`}
        confirmText="Delete"
        destructive
        onConfirm={() => {
          if (confirmDoc) handleDelete(confirmDoc);
          setConfirmDoc(null);
        }}
        onCancel={() => setConfirmDoc(null)}
      />
    </>
  );
}
