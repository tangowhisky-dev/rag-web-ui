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
import { FileText, Trash2 } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

interface ProcessingTask {
  id: number;
  status: string;
  error_message: string | null;
  progress?: number;
  progress_message?: string;
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

export function DocumentList({ knowledgeBaseId, refreshKey }: DocumentListProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  // live progress keyed by task id — overlays the static data from KB GET
  const [taskProgress, setTaskProgress] = useState<Record<number, ProcessingTask>>({});
  const { toast } = useToast();

  // Stable refs so the poll loop never has stale closures
  const docsRef = useRef<Document[]>([]);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const knowledgeBaseIdRef = useRef(knowledgeBaseId);
  knowledgeBaseIdRef.current = knowledgeBaseId;

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
      const taskIds: number[] = [];
      for (const doc of docs) {
        for (const t of doc.processing_tasks) {
          if (t.status === "pending" || t.status === "processing") {
            taskIds.push(t.id);
          }
        }
      }

      // Also poll any task we're tracking in taskProgress that's still running
      // (handles the case where the dialog was closed before the KB list refreshed)
      setTaskProgress((prev) => {
        for (const [idStr, t] of Object.entries(prev)) {
          if (t.status === "pending" || t.status === "processing") {
            const id = parseInt(idStr);
            if (!taskIds.includes(id)) taskIds.push(id);
          }
        }
        return prev;
      });

      if (taskIds.length === 0) return; // nothing in progress, stop polling

      try {
        const resp = await api.get(
          `/api/knowledge-base/${knowledgeBaseIdRef.current}/documents/tasks?task_ids=${taskIds.join(",")}`
        );
        const updates: Record<number, ProcessingTask> = {};
        let anyRunning = false;
        let anyFinished = false;
        for (const [idStr, info] of Object.entries(resp as Record<string, ProcessingTask>)) {
          const id = parseInt(idStr);
          updates[id] = info as ProcessingTask;
          if (info.status === "pending" || info.status === "processing") anyRunning = true;
          if (info.status === "completed" || info.status === "failed") anyFinished = true;
        }
        setTaskProgress((prev) => ({ ...prev, ...updates }));

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
  }, [fetchDocuments]);

  // Initial load + reload when refreshKey changes
  useEffect(() => {
    clearPoll();
    setLoading(true);
    fetchDocuments().then((docs) => {
      if (docs) {
        docsRef.current = docs;
        const hasInProgress = docs.some((d) =>
          d.processing_tasks.some(
            (t) => t.status === "pending" || t.status === "processing"
          )
        );
        if (hasInProgress) schedulePoll();
      }
    });
    return clearPoll;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [knowledgeBaseId, refreshKey]);

  const handleDelete = async (doc: Document) => {
    if (!confirm(`Delete "${doc.file_name}"? This will remove the file and all its indexed chunks.`)) return;
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
        <div className="space-y-4">
          <div className="w-8 h-8 border-4 border-primary/30 border-t-primary rounded-full animate-spin mx-auto" />
          <p className="text-muted-foreground animate-pulse">Loading documents...</p>
        </div>
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
            <h3 className="text-xl font-semibold">No documents yet</h3>
            <p className="text-muted-foreground">
              Upload your first document to start building your knowledge base.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
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
                  onClick={() => handleDelete(doc)}
                  className="h-8 w-8 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                  title={isInProgress ? "Cannot delete while processing" : "Delete document"}
                >
                  {deletingId === doc.id ? (
                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
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
  );
}
