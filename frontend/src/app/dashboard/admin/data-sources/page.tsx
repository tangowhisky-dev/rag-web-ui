'use client';

import { useState, useEffect, useRef } from 'react';
import { api, ApiError } from '@/lib/api';
import { fetchTokenClaims } from '@/lib/auth';

import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToast } from '@/components/ui/use-toast';
import { LoadingDots } from '@/components/ui/loading-dots';
import { Loader2, CheckCircle2, AlertCircle, Pause } from 'lucide-react';

interface Org {
  id: number;
  name: string;
}

interface DataStore {
  id: number;
  name: string;
  description: string | null;
  folder_path: string;
  scan_pattern: string;
  is_active: boolean;
  auto_process_enabled: boolean;
  auto_process_interval_minutes: number;
  last_scan_at: string | null;
  last_scan_status: string;
  last_scan_error: string | null;
  last_scan_total_files: number;
  last_scan_processed: number;
  last_scan_new: number;
  last_scan_modified: number;
  last_scan_skipped: number;
  last_scan_errors: number;
  assigned_orgs: Array<{ id: number; name: string }>;
  created_at: string;
  updated_at: string;
  last_recovered_at: string | null;
  // Real-time scan progress (populated when a scan is running)
  scan_progress?: {
    total_files: number;
    processed_files: number;
    status: string;
    new_files: number;
    skipped_files: number;
    error_files: number;
  };
  pending_changes: number;
  // Whether changes are currently being processed (event-driven ingestion)
  processing: boolean;
  // Aggregated graph build status across all documents
  graph_summary?: {
    total: number;
    pending: number;
    completed: number;
    failed: number;
    status: string; // "idle" | "running" | "completed" | "failed"
  } | null;
  graph_ingestion_paused: boolean;
}

interface ScanProgress {
  total_files: number;
  processed_files: number;
  status: string;
  scanned?: number;
  new_files?: number;
  modified_files?: number;
  skipped_files?: number;
  error_files?: number;
  error_message?: string;
}

interface RecoveryProgress {
  status: string;
  new_files?: number;
  modified?: number;
  deleted?: number;
}

interface RecoveryStatus {
  id: number;
  name: string;
  recovery_status: string;
  scan_id: number | null;
  total_files: number;
  processed_files: number;
  new_files: number;
  modified_files: number;
  deleted_files: number;
  started_at: string | null;
  error_message: string | null;
  last_recovered_at: string | null;
}

const STATUS_CONFIG: Record<string, { cls: string; label: string }> = {
  never: { cls: 'bg-gray-100 text-gray-600', label: '—' },
  running: { cls: 'bg-blue-100 text-blue-700', label: 'Running' },
  completed: { cls: 'bg-green-100 text-green-700', label: 'Completed' },
  error: { cls: 'bg-red-100 text-red-700', label: 'Error' },
  idle: { cls: 'bg-gray-100 text-gray-600', label: '—' },
  cancelled: { cls: 'bg-yellow-100 text-yellow-700', label: 'Cancelled' },
  paused: { cls: 'bg-amber-100 text-amber-700', label: 'Paused' },
};

const RECOVERY_STATUS_CONFIG: Record<string, { cls: string; label: string }> = {
  idle: { cls: 'bg-gray-100 text-gray-600', label: 'Idle' },
  running: { cls: 'bg-blue-100 text-blue-700', label: 'Recovering' },
  complete: { cls: 'bg-green-100 text-green-700', label: 'Complete' },
  error: { cls: 'bg-red-100 text-red-700', label: 'Error' },
};

function StatusBadge({ status, isRunning }: { status: string; isRunning?: boolean }) {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.never;
  return (
    <Badge variant="secondary" className={`${config.cls} ${isRunning ? 'animate-pulse' : ''}`}>
      {config.label}
    </Badge>
  );
}

export default function DataSourcesPage() {
  const { toast } = useToast();

  const [datastores, setDatastores] = useState<DataStore[]>([]);
  const [orgs, setOrgs] = useState<Org[]>([]);
  const [loading, setLoading] = useState(true);
  const [isSuperAdmin, setIsSuperAdmin] = useState(false);
  const [triggering, setTriggering] = useState<Set<number>>(new Set());
  const [flushing, setFlushing] = useState<Set<number>>(new Set());
  const [scanProgress, setScanProgress] = useState<Record<number, ScanProgress | undefined>>({});
  const [recoveryProgress, setRecoveryProgress] = useState<Record<number, RecoveryProgress | undefined>>({});
  const [recoveryStatuses, setRecoveryStatuses] = useState<Record<number, RecoveryStatus>>({});
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const scanEventSourceRef = useRef<EventSource | null>(null);

  // Default form values — source of truth for the create/edit dialog
  const formDefaults = {
    name: '',
    description: '',
    folder_path: '',
    scan_pattern: '*',
    auto_process_enabled: false,
    auto_process_interval_minutes: 30,
  };

  // Create/Edit dialog
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState(formDefaults);

  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Assign dialog
  const [assignOpen, setAssignOpen] = useState(false);
  const [assigningId, setAssigningId] = useState<number | null>(null);
  const [selectedOrgIds, setSelectedOrgIds] = useState<number[]>([]);

  // Fetch user role on mount
  useEffect(() => {
    fetchTokenClaims().then((claims) => {
      setIsSuperAdmin(claims?.role === 'super_admin');
    });
  }, []);

  // Poll for scan progress when ANY datastore is processing or scanning
  // Note: SSE is used for real-time progress during manual scans, but we
  // still poll for event-driven processing and background scans.
  useEffect(() => {
    const hasProcessing = datastores.some(
      (ds) => ds.processing
    );
    const hasRunningScan = datastores.some(
      (ds) => ds.last_scan_status === 'running' || ds.scan_progress?.status === 'running'
    );
    const hasRunningGraph = datastores.some(
      (ds) => ds.graph_summary?.status === 'running'
    );
    if (hasProcessing || hasRunningScan || hasRunningGraph) {
      // Poll every 2-5 seconds for event-driven processing and background scans
      pollingRef.current = setInterval(() => {
        fetchData();
      }, hasProcessing ? 2000 : 5000);
    } else {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
      // NOTE: Do NOT close scanEventSourceRef here.  The SSE connection is
      // managed by handleTriggerScan (opens) and its onmessage/onerror
      // handlers (closes).  Closing it here kills the SSE stream every time
      // datastores changes (which happens on every poll), leaving
      // scanProgress stuck at 0/0 until the page is refreshed.
    };
  }, [triggering, datastores]);

  // Poll recovery status for all datastores
  useEffect(() => {
    if (datastores.length === 0) return;

    const fetchRecoveryStatuses = async () => {
      try {
        const statuses = await api.get('/api/admin/datastores/recovery-status') as RecoveryStatus[];
        const statusMap: Record<number, RecoveryStatus> = {};
        const progressMap: Record<number, RecoveryProgress | undefined> = {};
        for (const s of statuses) {
          statusMap[s.id] = s;
          const st = s.recovery_status;
          if (st === 'running' || st === 'complete' || st === 'error') {
            progressMap[s.id] = {
              status: st,
              new_files: s.new_files ?? 0,
              modified: s.modified_files ?? 0,
              deleted: s.deleted_files ?? 0,
            };
          }
        }
        setRecoveryStatuses(statusMap);
        setRecoveryProgress(progressMap);
      } catch {
        // Recovery service may not be ready yet
      }
    };

    fetchRecoveryStatuses();
    const id = setInterval(fetchRecoveryStatuses, 5000);
    return () => clearInterval(id);
  }, [datastores.length]);

  // Refresh datastores list when recovery completes or fails
  useEffect(() => {
    const interval = setInterval(() => {
      for (const ds of datastores) {
        const st = recoveryStatuses[ds.id];
        if (!st) continue;
        if (st.recovery_status === 'complete' || st.recovery_status === 'error') {
          fetchData();
          break;
        }
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [recoveryStatuses, datastores.length]);

  useEffect(() => {
    fetchData();
  }, []);

  async function fetchData() {
    try {
      const [dsData, orgData] = await Promise.all([
        api.get('/api/admin/datastores?limit=200') as Promise<{ items: DataStore[]; total: number } | DataStore[]>,
        api.get('/api/admin/orgs') as Promise<Org[]>,
      ]);
      // Handle both paginated {items: [...]} and legacy [...] responses
      const dsList = Array.isArray(dsData) ? dsData : dsData.items;
      setDatastores(dsList);
      setOrgs(orgData);
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to load data stores',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  }

  function openCreate() {
    setEditingId(null);
    setForm(formDefaults);
    setDialogOpen(true);
  }

  function openEdit(ds: DataStore) {
    setEditingId(ds.id);
    setForm({ ...formDefaults, name: ds.name, description: ds.description ?? '', folder_path: ds.folder_path, scan_pattern: ds.scan_pattern, auto_process_enabled: ds.auto_process_enabled, auto_process_interval_minutes: ds.auto_process_interval_minutes });
    setDialogOpen(true);
  }

  async function handleSave() {
    if (!form.name.trim() || !form.folder_path.trim()) return;
    setSaving(true);
    try {
      if (editingId) {
        await api.patch(`/api/admin/datastores/${editingId}`, form);
        toast({ title: 'Data store updated' });
      } else {
        const result = await api.post('/api/admin/datastores', form);
        // Show file count after creation
        const fileCount = (result as DataStore).last_scan_total_files || 0;
        toast({ 
          title: 'Data store created',
          description: `Found ${fileCount} files in folder`,
        });
      }
      setDialogOpen(false);
      await fetchData();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? 'Failed to save data store',
        variant: 'destructive',
      });
    } finally {
      setSaving(false);
    }
  }

  function openDelete(ds: DataStore) {
    setDeletingId(ds.id);
  }

  async function handleDelete() {
    if (!deletingId) return;
    setDeleting(true);
    try {
      await api.delete(`/api/admin/datastores/${deletingId}`);
      toast({ title: 'Data store deleted' });
      setDeletingId(null);
      await fetchData();
    } catch (err) {
      const apiErr = err as ApiError;
      toast({
        title: 'Error',
        description: apiErr.message ?? 'Failed to delete data store',
        variant: 'destructive',
      });
    } finally {
      setDeleting(false);
      setDeletingId(null);
    }
  }

  async function handleTriggerScan(dsId: number) {
    setTriggering((prev) => new Set(prev).add(dsId));
    setScanProgress((prev) => ({ ...prev, [dsId]: { total_files: 0, processed_files: 0, status: 'running' } }));

    // Close any existing SSE connection before opening a new one.
    if (scanEventSourceRef.current) {
      scanEventSourceRef.current.close();
      scanEventSourceRef.current = null;
    }

    try {
      // Start the scan
      await api.post(`/api/admin/datastores/${dsId}/scan`);

      // Refresh datastores so polling effect detects the running scan
      // even if SSE is buffered/broken.
      fetchData();

      // Clear triggering immediately — the button state is now driven by
      // scanProgress and ds.last_scan_status, not by the triggering set.
      // Keeping it locked until SSE cleanup makes the stop button unclickable
      // if SSE is buffered/broken.
      setTriggering((prev) => {
        const next = new Set(prev);
        next.delete(dsId);
        return next;
      });

      // Subscribe to SSE stream for real-time progress updates.
      const es = new EventSource(`/api/admin/datastores/${dsId}/scan-progress-stream`);
      scanEventSourceRef.current = es;

      const cleanup = () => {
        es.close();
        if (scanEventSourceRef.current === es) {
          scanEventSourceRef.current = null;
        }
      };

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as ScanProgress;

          // "waiting" status is a keep-alive from the backend while the
          // scan registers — don't update UI for it.
          if (data.status === 'waiting') return;

          setScanProgress((prev) => ({
            ...prev,
            [dsId]: {
              total_files: data.total_files || 0,
              processed_files: data.processed_files || 0,
              status: data.status || 'running',
              new_files: data.new_files || 0,
              modified_files: data.modified_files || 0,
              skipped_files: data.skipped_files || 0,
              error_files: data.error_files || 0,
              error_message: data.error_message,
            },
          }));

          if (data.status === 'completed') {
            const parts = [
              `Scanned: ${data.processed_files || 0}`,
              `New: ${data.new_files || 0}`,
              `Modified: ${data.modified_files || 0}`,
              `Skipped: ${data.skipped_files || 0}`,
            ];
            if (data.error_files && data.error_files > 0) {
              parts.push(`Errors: ${data.error_files}`);
            }
            toast({ title: 'Processing completed (less graph ingestion)', description: parts.join(' | ') });
            setScanProgress((prev) => ({ ...prev, [dsId]: undefined }));
            cleanup();
            fetchData();
          } else if (data.status === 'error') {
            const errorMsg = data.error_message || `Errors: ${data.error_files || 1}`;
            toast({ title: 'Processing failed', description: errorMsg, variant: 'destructive' });
            setScanProgress((prev) => ({ ...prev, [dsId]: undefined }));
            cleanup();
            fetchData();
          } else if (data.status === 'cancelled') {
            setScanProgress((prev) => ({ ...prev, [dsId]: undefined }));
            cleanup();
            fetchData();
          } else if (data.status === 'paused') {
            setScanProgress((prev) => ({ ...prev, [dsId]: undefined }));
            cleanup();
            fetchData();
          }
        } catch {
          // Ignore malformed events
        }
      };

      es.onerror = () => {
        // EventSource auto-reconnects, but if the scan is done the server
        // closes the stream.  Clean up and refresh to get final state.
        if (es.readyState === EventSource.CLOSED) {
          setScanProgress((prev) => ({ ...prev, [dsId]: undefined }));
          cleanup();
          fetchData();
        }
      };
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? 'Failed to start scan',
        variant: 'destructive',
      });
      setScanProgress((prev) => ({ ...prev, [dsId]: undefined }));
      setTriggering((prev) => {
        const next = new Set(prev);
        next.delete(dsId);
        return next;
      });
    }
  }

  async function handleFlushChanges(dsId: number) {
    setFlushing((prev) => new Set(prev).add(dsId));
    try {
      const result = (await api.post(
        `/api/admin/datastores/${dsId}/flush`,
      )) as { pending_processed: number; processing: boolean };
      toast({
        title: 'Changes flushed',
        description: `Processed: ${result.pending_processed} pending change(s)`,
      });
      await fetchData();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? 'Failed to flush changes',
        variant: 'destructive',
      });
    }
    // Clear flushing so button re-enables
    setFlushing((prev) => {
      const next = new Set(prev);
      next.delete(dsId);
      return next;
    });
  }

  async function handlePauseScan(dsId: number) {
    try {
      const resp = (await api.post(
        `/api/admin/datastores/${dsId}/stop-scan?pause=true`,
      )) as { message: string };
      toast({
        title: 'Scan paused',
        description: resp.message,
      });
      // Close the SSE stream and clear progress state
      if (scanEventSourceRef.current) {
        scanEventSourceRef.current.close();
        scanEventSourceRef.current = null;
      }
      setScanProgress((prev) => ({ ...prev, [dsId]: undefined }));
      await fetchData();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? 'Failed to pause scan',
        variant: 'destructive',
      });
    }
  }

  async function handleGraphToggle(dsId: number, isPaused: boolean) {
    const endpoint = isPaused ? 'graph-resume' : 'graph-pause';
    try {
      const resp = (await api.post(
        `/api/admin/datastores/${dsId}/${endpoint}`,
      )) as { message: string };
      toast({
        title: isPaused ? 'Graph ingestion resumed' : 'Graph ingestion paused',
        description: resp.message,
      });
      await fetchData();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? `Failed to ${isPaused ? 'resume' : 'pause'} graph ingestion`,
        variant: 'destructive',
      });
    }
  }

  async function openAssign(dsId: number) {
    setAssigningId(dsId);
    // Fetch current assignments directly instead of relying on stale datastores state
    try {
      const ds = await api.get(`/api/admin/datastores/${dsId}`) as DataStore;
      setSelectedOrgIds(ds?.assigned_orgs?.map((o) => o.id) ?? []);
    } catch {
      // Fallback to state if fetch fails
      const ds = datastores.find((d) => d.id === dsId);
      setSelectedOrgIds(ds?.assigned_orgs?.map((o) => o.id) ?? []);
    }
    setAssignOpen(true);
  }

  async function handleAssign() {
    if (!assigningId) return;
    try {
      await api.post(`/api/admin/datastores/${assigningId}/assign`, {
        org_ids: selectedOrgIds,
        force_clear: selectedOrgIds.length === 0,
      });
      toast({
        title: 'Assignments updated',
        description: selectedOrgIds.length === 0 ? 'All organisations removed' : undefined,
      });
      setAssignOpen(false);
      await fetchData();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? 'Failed to update assignments',
        variant: 'destructive',
      });
    }
  }

  const formatLastScan = (timestamp: string | null) => {
    if (!timestamp) return 'never';
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 pt-16 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Data Stores</h1>
          <p className="text-muted-foreground">
            Manage local folders for automatic document ingestion
          </p>
        </div>
        {isSuperAdmin && <Button onClick={openCreate}>+ New Data Store</Button>}
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Folder Path</TableHead>
              <TableHead>Org(s)</TableHead>
              <TableHead>Background Processing</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Files</TableHead>
              <TableHead>Recovery </TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {datastores.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="text-center text-muted-foreground">
                  No data stores configured. Click &ldquo;+ New Data Store&rdquo; to add one.
                </TableCell>
              </TableRow>
            ) : (
              datastores.map((ds) => (
                <TableRow key={ds.id}>
                  <TableCell className="font-medium">
                    <a href={`/dashboard/admin/data-sources/${ds.id}`} className="hover:underline">
                      {ds.name}
                    </a>
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground max-w-[200px] truncate">
                    {ds.folder_path}
                  </TableCell>
                  <TableCell>
                    {ds.assigned_orgs.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {ds.assigned_orgs.map((org) => (
                          <Badge key={org.id} variant="secondary" className="text-xs">
                            {org.name}
                          </Badge>
                        ))}
                      </div>
                    ) : (
                      <span className="text-xs text-muted-foreground">Not assigned</span>
                    )}
                  </TableCell>
                  <TableCell>
                    {ds.auto_process_enabled ? (
                      <Badge variant="secondary" className="bg-green-100 text-green-700">
                        Immediate + auto-scan
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="bg-gray-100 text-gray-500">
                        Manual only
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <StatusBadge status={ds.last_scan_status} isRunning={ds.last_scan_status === 'running'} />
                      {ds.processing && (
                        <span className="inline-flex items-center gap-1 text-[10px] text-orange-600">
                          <span className="w-1.5 h-1.5 rounded-full bg-orange-500 animate-pulse"></span>
                          Processing
                        </span>
                      )}
                      {ds.pending_changes > 0 && ds.last_scan_status !== 'running' && ds.last_scan_status !== 'idle' && (
                        <span className="inline-flex items-center gap-1 text-[10px] text-yellow-600">
                          <span className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-pulse"></span>
                          {ds.pending_changes}
                        </span>
                      )}
                    </div>
                    <div className="mt-1">{formatLastScan(ds.last_scan_at)}</div>
                    {ds.last_scan_status === 'error' && ds.last_scan_error && (
                      <div className="mt-1 text-xs text-red-500 truncate max-w-[180px]" title={ds.last_scan_error}>
                        {ds.last_scan_error}
                      </div>
                    )}
                  </TableCell>
                  <TableCell>
                    {(() => {
                      const progress = scanProgress[ds.id];
                      if (progress && progress.status !== 'completed' && progress.status !== 'paused') {
                        const pct = progress.total_files > 0
                          ? Math.min((progress.processed_files / Math.max(progress.total_files, 1)) * 100, 100)
                          : 0;
                        const finalizing = pct >= 100 && progress.status === 'running';
                        return (
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <LoadingDots size="sm" />
                              <span className="text-xs text-blue-600">{finalizing ? 'Finalizing ingestion...' : 'Processing...'}</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div 
                                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${pct}%` }}
                              ></div>
                            </div>
                            <div className="flex justify-between text-xs text-muted-foreground">
                              <span>{progress.processed_files} / {progress.total_files}</span>
                              <span>{pct.toFixed(0)}%</span>
                            </div>
                            <div className="flex flex-wrap gap-2 text-[10px] text-muted-foreground">
                              {progress.new_files != null && progress.new_files > 0 && <span>New: {progress.new_files}</span>}
                              {progress.modified_files != null && progress.modified_files > 0 && <span>Modified: {progress.modified_files}</span>}
                              {progress.skipped_files != null && progress.skipped_files > 0 && <span>Skipped: {progress.skipped_files}</span>}
                              {progress.error_files != null && progress.error_files > 0 && <span className="text-red-500">Errors: {progress.error_files}</span>}
                            </div>
                          </div>
                        );
                      }
                      if (ds.last_scan_status === 'paused') {
                        const pct = ds.last_scan_total_files > 0
                          ? Math.min((ds.last_scan_processed / Math.max(ds.last_scan_total_files, 1)) * 100, 100)
                          : 0;
                        return (
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-amber-600 font-medium">Paused</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div 
                                className="bg-amber-500 h-2 rounded-full"
                                style={{ width: `${pct}%` }}
                              ></div>
                            </div>
                            <div className="flex justify-between text-xs text-muted-foreground">
                              <span>{ds.last_scan_processed} / {ds.last_scan_total_files}</span>
                              <span>{pct.toFixed(0)}%</span>
                            </div>
                          </div>
                        );
                      }
                      if (ds.last_scan_status === 'running') {
                        const pct = ds.last_scan_total_files > 0
                          ? Math.min((ds.last_scan_processed / Math.max(ds.last_scan_total_files, 1)) * 100, 100)
                          : 0;
                        const finalizing = pct >= 100;
                        return (
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <LoadingDots size="sm" />
                              <span className="text-xs text-blue-600">{finalizing ? 'Finalizing ingestion...' : 'Processing...'}</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div 
                                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${pct}%` }}
                              ></div>
                            </div>
                            <div className="flex justify-between text-xs text-muted-foreground">
                              <span>{ds.last_scan_processed} / {ds.last_scan_total_files}</span>
                              <span>{pct.toFixed(0)}%</span>
                            </div>
                          </div>
                        );
                      }
                      if (ds.last_scan_status === 'error') {
                        return (
                          <div className="space-y-1">
                            <div className="text-xs text-red-600">Error</div>
                            <div className="text-xs text-muted-foreground truncate max-w-[180px]" title={ds.last_scan_error || ''}>
                              {ds.last_scan_error}
                            </div>
                          </div>
                        );
                      }
                      return (
                        <div className="text-xs">
                          {ds.last_scan_total_files} files
                          <br />
                          {ds.last_scan_processed} processed
                          {ds.processing && (
                            <div className="mt-1 flex items-center gap-1">
                              <div className="w-2 h-2 rounded-full bg-orange-500 animate-pulse"></div>
                              <span className="text-orange-600">Processing changes...</span>
                            </div>
                          )}
                          {ds.pending_changes > 0 && (
                            <div className="mt-1 flex items-center gap-1">
                              <div className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse"></div>
                              <span className="text-yellow-600">{ds.pending_changes} pending</span>
                            </div>
                          )}
                        </div>
                      );
                    })()}
                    {ds.graph_summary && ds.graph_summary.total > 0 && (
                      <div className="mt-2 flex items-center gap-1.5 text-xs">
                        {ds.graph_ingestion_paused && (
                          <>
                            <Pause className="h-3 w-3 text-amber-500" />
                            <span className="text-amber-600">
                              Graph paused ({ds.graph_summary.pending} pending)
                            </span>
                          </>
                        )}
                        {!ds.graph_ingestion_paused && ds.graph_summary.status === 'running' && (
                          <>
                            <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                            <span className="text-muted-foreground">
                              Graph {ds.graph_summary.completed}/{ds.graph_summary.total}
                            </span>
                          </>
                        )}
                        {ds.graph_summary.status === 'completed' && (
                          <>
                            <CheckCircle2 className="h-3 w-3 text-muted-foreground" />
                            <span className="text-muted-foreground">
                              Graph {ds.graph_summary.completed}/{ds.graph_summary.total}
                            </span>
                          </>
                        )}
                        {ds.graph_summary.status === 'failed' && (
                          <>
                            <AlertCircle className="h-3 w-3 text-amber-500" />
                            <span className="text-muted-foreground">
                              Graph {ds.graph_summary.completed}/{ds.graph_summary.total}
                              {ds.graph_summary.failed > 0 && ` (${ds.graph_summary.failed} failed)`}
                            </span>
                          </>
                        )}
                      </div>
                    )}
                  </TableCell>
                  <TableCell>
                    {(() => {
                      const rp = recoveryProgress[ds.id];
                      const rs = recoveryStatuses[ds.id];
                      if (!rp) {
                        return (
                          <div className="flex items-center gap-1 text-xs text-muted-foreground">
                            <span className="text-gray-400">⏸</span> Idle
                          </div>
                        );
                      }
                      const rc = RECOVERY_STATUS_CONFIG[rp.status] ?? RECOVERY_STATUS_CONFIG.idle;
                      return (
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            {rp.status === 'running' && (
                              <LoadingDots size="sm" />
                            )}
                            <Badge variant="secondary" className={`${rc.cls} text-[10px]`}>{rc.label}</Badge>
                          </div>
                          {rp.status === 'running' && (
                            <div className="text-[10px] text-muted-foreground">
                              {rp.new_files! > 0 && <span className="text-green-600">{rp.new_files} new </span>}
                              {rp.modified! > 0 && <span className="text-yellow-600">{rp.modified} modified </span>}
                              {rp.deleted! > 0 && <span className="text-red-600">{rp.deleted} deleted</span>}
                            </div>
                          )}
                          {rp.status === 'complete' && (
                            <div className="text-[10px] text-muted-foreground">
                              {rs?.last_recovered_at ? (
                                <>Last recover: {new Date(rs.last_recovered_at).toLocaleString()}</>
                              ) : (
                                'No recovery yet'
                              )}
                              {rp.new_files! > 0 && <span className="text-green-600"> {rp.new_files} new </span>}
                              {rp.modified! > 0 && <span className="text-yellow-600">{rp.modified} modified </span>}
                              {rp.deleted! > 0 && <span className="text-red-600">{rp.deleted} deleted</span>}
                            </div>
                          )}
                          {rp.status === 'error' && (
                            <div className="text-[10px] text-red-500">Recovery failed</div>
                          )}
                        </div>
                      );
                    })()}
                  </TableCell>
                  <TableCell className="space-x-1">
                    {isSuperAdmin ? (
                      <>
                        {(() => {
                          const isRunning = ds.last_scan_status === 'running' || ds.scan_progress?.status === 'running' || (scanProgress[ds.id]?.status !== 'completed' && scanProgress[ds.id]?.status !== 'paused' && scanProgress[ds.id]);
                          const isPaused = ds.last_scan_status === 'paused' || scanProgress[ds.id]?.status === 'paused';

                          if (isRunning) {
                            return (
                              <Button
                                variant="secondary"
                                size="sm"
                                onClick={() => handlePauseScan(ds.id)}
                                disabled={triggering.has(ds.id)}
                                title="Pause processing — can be resumed later"
                              >
                                Pause
                              </Button>
                            );
                          }
                          if (isPaused) {
                            return (
                              <Button
                                variant="default"
                                size="sm"
                                onClick={() => handleTriggerScan(ds.id)}
                                disabled={triggering.has(ds.id)}
                                title="Resume processing"
                              >
                                Resume
                              </Button>
                            );
                          }
                          return (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleTriggerScan(ds.id)}
                              disabled={triggering.has(ds.id)}
                              title="Trigger manual processing"
                            >
                              Process
                            </Button>
                          );
                        })()}
                        {ds.graph_summary && ds.graph_summary.total > 0 && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleGraphToggle(ds.id, ds.graph_ingestion_paused)}
                            title={ds.graph_ingestion_paused ? 'Resume graph ingestion' : 'Pause graph ingestion'}
                          >
                            {ds.graph_ingestion_paused ? 'Resume Graph' : 'Pause Graph'}
                          </Button>
                        )}
                        {ds.pending_changes > 0 && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleFlushChanges(ds.id)}
                            disabled={flushing.has(ds.id)}
                            title="Flush pending changes"
                          >
                            {flushing.has(ds.id) ? '⏳' : '⟳'}
                          </Button>
                        )}
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => openAssign(ds.id)}
                          title="Assign to organisations"
                        >
                          Assign
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => openEdit(ds)}
                          title="Edit data store"
                        >
                          Edit
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => openDelete(ds)}
                          title="Delete data store"
                        >
                          Delete
                        </Button>
                      </>
                    ) : (
                      <span className="text-xs text-muted-foreground">Read-only</span>
                    )}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog open={deletingId !== null} onOpenChange={(open) => !open && setDeletingId(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="text-destructive">Permanently Delete Data Store</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <p className="text-sm">
              Are you sure you want to permanently delete this data store? This action:
            </p>
            <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
              <li>Stops all running ingestion and graph builds for this data store</li>
              <li>Removes the data store permanently (cannot be undone)</li>
              <li>Does NOT delete files from the folder</li>
              <li>Unassigns the data store from all organizations</li>
            </ul>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeletingId(null)} disabled={deleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={deleting}>
              {deleting ? 'Deleting...' : 'Delete'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create/Edit Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {editingId ? 'Edit Data Store' : 'New Data Store'}
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div className="space-y-1">
              <Label htmlFor="ds-name">Name</Label>
              <Input
                id="ds-name"
                placeholder="e.g. HQ Documents"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="ds-desc">Description</Label>
              <Input
                id="ds-desc"
                placeholder="Optional description"
                value={form.description}
                onChange={(e) => setForm({ ...form, description: e.target.value })}
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="ds-path">Folder Path</Label>
              <Input
                id="ds-path"
                placeholder="/app/data/my-folder"
                value={form.folder_path}
                onChange={(e) => setForm({ ...form, folder_path: e.target.value })}
              />
              <p className="text-xs text-muted-foreground">
                Folder must exist in the Docker container. Example: /app/data/my-folder
              </p>
            </div>
            <div className="space-y-1">
              <Label htmlFor="ds-pattern">Scan Pattern</Label>
              <Input
                id="ds-pattern"
                placeholder="*"
                value={form.scan_pattern}
                onChange={(e) => setForm({ ...form, scan_pattern: e.target.value })}
              />
            </div>
            {editingId && (
              <>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="ds-auto"
                    checked={form.auto_process_enabled}
                    onChange={(e) =>
                      setForm({ ...form, auto_process_enabled: e.target.checked })
                    }
                    className="h-4 w-4 rounded border-gray-300"
                  />
                  <Label htmlFor="ds-auto">Background processing enabled</Label>
                </div>
                {form.auto_process_enabled && (
                  <div className="space-y-1">
                    <Label htmlFor="ds-interval">Process Interval (minutes)</Label>
                    <p className="text-xs text-muted-foreground">
                      File changes are automatically processed every {form.auto_process_interval_minutes} minutes
                    </p>
                    <Input
                      id="ds-interval"
                      type="number"
                      min={1}
                      max={1440}
                      value={form.auto_process_interval_minutes}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          auto_process_interval_minutes: e.target.value
                            ? parseInt(e.target.value, 10) || formDefaults.auto_process_interval_minutes
                            : form.auto_process_interval_minutes,
                        })
                      }
                    />
                  </div>
                )}
              </>
            )}
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDialogOpen(false)}
              disabled={saving}
            >
              Cancel
            </Button>
            <Button
              onClick={handleSave}
              disabled={saving || !form.name.trim() || !form.folder_path.trim()}
            >
              {editingId ? 'Save' : 'Create'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Assign Dialog */}
      <Dialog open={assignOpen} onOpenChange={setAssignOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Assign to Organisations</DialogTitle>
          </DialogHeader>
          <div className="space-y-2 py-2 max-h-60 overflow-y-auto">
            {orgs.map((org) => (
              <div key={org.id} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id={`assign-${org.id}`}
                  checked={selectedOrgIds.includes(org.id)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedOrgIds([...selectedOrgIds, org.id]);
                    } else {
                      setSelectedOrgIds(selectedOrgIds.filter((id) => id !== org.id));
                    }
                  }}
                  className="h-4 w-4 rounded border-gray-300"
                />
                <Label htmlFor={`assign-${org.id}`}>{org.name}</Label>
              </div>
            ))}
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setAssignOpen(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleAssign}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
