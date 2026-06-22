'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { api, ApiError } from '@/lib/api';
import { getTokenClaims } from '@/lib/auth';
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
  auto_scan_enabled: boolean;
  auto_scan_interval_minutes: number;
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
}

interface ScanResult {
  scanned: number;
  new: number;
  modified: number;
  skipped: number;
  errors: number;
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

const STATUS_CONFIG: Record<string, { cls: string; label: string }> = {
  never: { cls: 'bg-gray-100 text-gray-600', label: '—' },
  running: { cls: 'bg-blue-100 text-blue-700', label: 'Running' },
  completed: { cls: 'bg-green-100 text-green-700', label: 'Completed' },
  error: { cls: 'bg-red-100 text-red-700', label: 'Error' },
  idle: { cls: 'bg-gray-100 text-gray-600', label: '—' },
  cancelled: { cls: 'bg-yellow-100 text-yellow-700', label: 'Cancelled' },
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
  const router = useRouter();
  const { toast } = useToast();

  // Auth check
  useEffect(() => {
    const claims = getTokenClaims();
    if (!claims || claims.role !== 'super_admin') {
      router.push('/dashboard');
    }
  }, [router]);

  const [datastores, setDatastores] = useState<DataStore[]>([]);
  const [orgs, setOrgs] = useState<Org[]>([]);
  const [loading, setLoading] = useState(true);
  const [triggering, setTriggering] = useState<Set<number>>(new Set());
  const [flushing, setFlushing] = useState<Set<number>>(new Set());
  const [scanProgress, setScanProgress] = useState<Record<number, ScanProgress | undefined>>({});
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const scanPollRef = useRef<{ active: boolean; dsId: number | null }>({ active: false, dsId: null });

  // Default form values — source of truth for the create/edit dialog
  const formDefaults = {
    name: '',
    description: '',
    folder_path: '',
    scan_pattern: '*',
    auto_scan_enabled: false,
    auto_scan_interval_minutes: 30,
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
    const hasManualTrigger = triggering.size > 0;

    if (hasProcessing || hasRunningScan) {
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
    };
  }, [triggering, datastores]);

  useEffect(() => {
    fetchData();
  }, []);

  async function fetchData() {
    try {
      const [dsData, orgData] = await Promise.all([
        api.get('/api/admin/datastores') as Promise<DataStore[]>,
        api.get('/api/admin/orgs') as Promise<Org[]>,
      ]);
      setDatastores(dsData);
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
    setForm({ ...formDefaults, name: ds.name, description: ds.description ?? '', folder_path: ds.folder_path, scan_pattern: ds.scan_pattern, auto_scan_enabled: ds.auto_scan_enabled, auto_scan_interval_minutes: ds.auto_scan_interval_minutes });
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
    scanPollRef.current = { active: true, dsId };

    try {
      // Start the scan
      const token = typeof window !== 'undefined' ? localStorage.getItem('token') || '' : '';
      const scanResp = await fetch(`/api/admin/datastores/${dsId}/scan`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!scanResp.ok) {
        throw new Error(`Scan start failed with status ${scanResp.status}: ${scanResp.statusText}`);
      }

      // Poll for scan progress instead of using SSE (SSE doesn't work through
      // Next.js rewrites which buffer streaming responses).
      const pollInterval = 500;
      const timeout = 120_000; // 2 minutes max
      const startTime = Date.now();

      while (scanPollRef.current.active && scanPollRef.current.dsId === dsId && Date.now() - startTime < timeout) {
        await new Promise((resolve) => setTimeout(resolve, pollInterval));

        const progressResp = await fetch(`/api/admin/datastores/${dsId}/scan-progress`, {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (!progressResp.ok) {
          continue;
        }

        const data = (await progressResp.json()) as ScanProgress;

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
          toast({
            title: 'Scan completed',
            description: parts.join(' | '),
          });
          setScanProgress((prev) => ({ ...prev, [dsId]: undefined }));
          await fetchData();
          break;
        } else if (data.status === 'error') {
          const errorMsg = data.error_message || `Errors: ${data.error_files || 1}`;
          toast({
            title: 'Scan failed',
            description: errorMsg,
            variant: 'destructive',
          });
          setScanProgress((prev) => ({ ...prev, [dsId]: undefined }));
          await fetchData();
          break;
        } else if (data.status === 'cancelled') {
          setScanProgress((prev) => ({ ...prev, [dsId]: undefined }));
          await fetchData();
          break;
        }
      }

      // Timeout — clean up and refresh
      if (Date.now() - startTime >= timeout) {
        setScanProgress((prev) => ({ ...prev, [dsId]: undefined }));
        await fetchData();
      }
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? 'Failed to start scan',
        variant: 'destructive',
      });
      setScanProgress((prev) => ({ ...prev, [dsId]: undefined }));
    } finally {
      scanPollRef.current = { active: false, dsId: null };
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

  async function handleStopScan(dsId: number) {
    try {
      const resp = (await api.post(
        `/api/admin/datastores/${dsId}/stop-scan`,
      )) as { message: string };
      toast({
        title: 'Scan stopped',
        description: resp.message,
      });
      // Interrupt the polling loop and clear progress state
      scanPollRef.current = { active: false, dsId: null };
      setScanProgress((prev) => ({ ...prev, [dsId]: undefined }));
      await fetchData();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? 'Failed to stop scan',
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
      });
      toast({ title: 'Assignments updated' });
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
        <Button onClick={openCreate}>+ New Data Store</Button>
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Folder Path</TableHead>
              <TableHead>Assigned Orgs</TableHead>
              <TableHead>Background Processing</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Files</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {datastores.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="text-center text-muted-foreground">
                  No data stores configured. Click &ldquo;+ New Data Store&rdquo; to add one.
                </TableCell>
              </TableRow>
            ) : (
              datastores.map((ds) => (
                <TableRow key={ds.id}>
                  <TableCell className="font-medium">{ds.name}</TableCell>
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
                    {ds.auto_scan_enabled ? (
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
                      if (progress && progress.status !== 'completed') {
                        const pct = progress.total_files > 0
                          ? Math.min((progress.processed_files / Math.max(progress.total_files, 1)) * 100, 100)
                          : 0;
                        return (
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                              <span className="text-xs text-blue-600">Processing...</span>
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
                      if (ds.last_scan_status === 'running') {
                        const pct = ds.last_scan_total_files > 0
                          ? Math.min((ds.last_scan_processed / Math.max(ds.last_scan_total_files, 1)) * 100, 100)
                          : 0;
                        return (
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                              <span className="text-xs text-blue-600">Processing...</span>
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
                  </TableCell>
                  <TableCell className="space-x-1">
                    <Button
                      variant={ds.last_scan_status === 'running' || ds.scan_progress?.status === 'running' || (scanProgress[ds.id]?.status !== 'completed' && scanProgress[ds.id]) ? 'destructive' : 'outline'}
                      size="sm"
                      onClick={() => {
                        if (ds.last_scan_status === 'running' || ds.scan_progress?.status === 'running' || (scanProgress[ds.id]?.status !== 'completed' && scanProgress[ds.id])) {
                          handleStopScan(ds.id);
                        } else {
                          handleTriggerScan(ds.id);
                        }
                      }}
                      disabled={triggering.has(ds.id)}
                      title={ds.last_scan_status === 'running' || ds.scan_progress?.status === 'running' || (scanProgress[ds.id]?.status !== 'completed' && scanProgress[ds.id]) ? 'Stop scan' : 'Trigger manual scan'}
                    >
                      {ds.last_scan_status === 'running' || ds.scan_progress?.status === 'running' || (scanProgress[ds.id]?.status !== 'completed' && scanProgress[ds.id]) ? 'Stop' : 'Scan'}
                    </Button>
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
                    checked={form.auto_scan_enabled}
                    onChange={(e) =>
                      setForm({ ...form, auto_scan_enabled: e.target.checked })
                    }
                    className="h-4 w-4 rounded border-gray-300"
                  />
                  <Label htmlFor="ds-auto">Background processing enabled</Label>
                </div>
                {form.auto_scan_enabled && (
                  <div className="space-y-1">
                    <Label htmlFor="ds-interval">Process Interval (minutes)</Label>
                    <p className="text-xs text-muted-foreground">
                      File changes are automatically processed every {form.auto_scan_interval_minutes} minutes
                    </p>
                    <Input
                      id="ds-interval"
                      type="number"
                      min={1}
                      max={1440}
                      value={form.auto_scan_interval_minutes}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          auto_scan_interval_minutes: e.target.value
                            ? parseInt(e.target.value, 10) || formDefaults.auto_scan_interval_minutes
                            : form.auto_scan_interval_minutes,
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
