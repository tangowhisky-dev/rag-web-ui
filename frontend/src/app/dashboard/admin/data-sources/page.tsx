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
  assigned_orgs: Array<{ id: number; name: string }>;
  created_at: string;
  updated_at: string;
}

interface ScanResult {
  scanned: number;
  new: number;
  skipped: number;
  errors: number;
}

const STATUS_CONFIG: Record<string, { cls: string; label: string }> = {
  never: { cls: 'bg-gray-100 text-gray-600', label: '—' },
  running: { cls: 'bg-blue-100 text-blue-700', label: 'Running' },
  completed: { cls: 'bg-green-100 text-green-700', label: 'Completed' },
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
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Create/Edit dialog
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState({
    name: '',
    description: '',
    folder_path: '',
    scan_pattern: '*',
    auto_scan_enabled: false,
    auto_scan_interval_minutes: 60,
  });

  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Assign dialog
  const [assignOpen, setAssignOpen] = useState(false);
  const [assigningId, setAssigningId] = useState<number | null>(null);
  const [selectedOrgIds, setSelectedOrgIds] = useState<number[]>([]);

  // Poll for status updates when running
  useEffect(() => {
    if (triggering.size > 0) {
      // Poll every 2 seconds while scanning
      pollingRef.current = setInterval(() => {
        fetchData();
      }, 2000);
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
  }, [triggering]);

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
        description: (err as { message?: string }).message ?? 'Failed to load data sources',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  }

  function openCreate() {
    setEditingId(null);
    setForm({
      name: '',
      description: '',
      folder_path: '',
      scan_pattern: '*',
      auto_scan_enabled: false,
      auto_scan_interval_minutes: 60,
    });
    setDialogOpen(true);
  }

  function openEdit(ds: DataStore) {
    setEditingId(ds.id);
    setForm({
      name: ds.name,
      description: ds.description ?? '',
      folder_path: ds.folder_path,
      scan_pattern: ds.scan_pattern,
      auto_scan_enabled: ds.auto_scan_enabled,
      auto_scan_interval_minutes: ds.auto_scan_interval_minutes,
    });
    setDialogOpen(true);
  }

  async function handleSave() {
    if (!form.name.trim() || !form.folder_path.trim()) return;
    setSaving(true);
    try {
      if (editingId) {
        await api.patch(`/api/admin/datastores/${editingId}`, form);
        toast({ title: 'Data source updated' });
      } else {
        const result = await api.post('/api/admin/datastores', form);
        // Show file count after creation
        const fileCount = (result as DataStore).last_scan_total_files || 0;
        toast({ 
          title: 'Data source created',
          description: `Found ${fileCount} files in folder`,
        });
      }
      setDialogOpen(false);
      await fetchData();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? 'Failed to save data source',
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
      toast({ title: 'Data source deleted' });
      setDeletingId(null);
      await fetchData();
    } catch (err) {
      const apiErr = err as ApiError;
      toast({
        title: 'Error',
        description: apiErr.message ?? 'Failed to delete data source',
        variant: 'destructive',
      });
    } finally {
      setDeleting(false);
      setDeletingId(null);
    }
  }

  async function handleTriggerScan(dsId: number) {
    setTriggering((prev) => new Set(prev).add(dsId));
    try {
      const result = (await api.post(
        `/api/admin/datastores/${dsId}/scan`,
      )) as ScanResult;
      toast({
        title: 'Scan completed',
        description: `Scanned: ${result.scanned} | New: ${result.new} | Skipped: ${result.skipped} | Errors: ${result.errors}`,
      });
      await fetchData();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? 'Failed to trigger scan',
        variant: 'destructive',
      });
    }
    // Polling will clear triggering automatically
  }

  function openAssign(dsId: number) {
    setAssigningId(dsId);
    const ds = datastores.find((d) => d.id === dsId);
    const assignedIds = ds?.assigned_orgs.map((o) => o.id) ?? [];
    setSelectedOrgIds(assignedIds);
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
          <h1 className="text-3xl font-bold tracking-tight">Data Sources</h1>
          <p className="text-muted-foreground">
            Manage local folders for automatic document ingestion
          </p>
        </div>
        <Button onClick={openCreate}>+ New Data Source</Button>
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
                  No data sources configured. Click &ldquo;+ New Data Source&rdquo; to add one.
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
                        {ds.auto_scan_interval_minutes} minute(s)
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="bg-gray-100 text-gray-500">
                        Manual
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    <StatusBadge status={ds.last_scan_status} isRunning={ds.last_scan_status === 'running'} />
                    <div className="mt-1">{formatLastScan(ds.last_scan_at)}</div>
                  </TableCell>
                  <TableCell>
                    {ds.last_scan_status === 'running' ? (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                          <span className="text-xs text-blue-600">Processing...</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${Math.min((ds.last_scan_processed / Math.max(ds.last_scan_total_files, 1)) * 100, 100)}%` }}
                          ></div>
                        </div>
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>{ds.last_scan_processed} / {ds.last_scan_total_files}</span>
                        </div>
                      </div>
                    ) : (
                      <div className="text-xs">
                        {ds.last_scan_total_files} files
                        <br />
                        {ds.last_scan_processed} processed
                      </div>
                    )}
                  </TableCell>
                  <TableCell className="space-x-1">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleTriggerScan(ds.id)}
                      disabled={triggering.has(ds.id)}
                      title="Trigger manual scan"
                    >
                      {triggering.has(ds.id) ? 'Scanning…' : 'Scan'}
                    </Button>
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
                      title="Edit data source"
                    >
                      Edit
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => openDelete(ds)}
                      title="Delete data source"
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
            <DialogTitle className="text-destructive">Permanently Delete Data Source</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <p className="text-sm">
              Are you sure you want to permanently delete this data source? This action:
            </p>
            <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
              <li>Removes the data source permanently (cannot be undone)</li>
              <li>Does NOT delete files from the folder</li>
              <li>Unassigns the data source from all organizations</li>
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
              {editingId ? 'Edit Data Source' : 'New Data Source'}
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
                      auto_scan_interval_minutes: parseInt(e.target.value, 10) || 60,
                    })
                  }
                />
              </div>
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
