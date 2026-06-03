'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { FolderOpen, Play, Trash2 } from 'lucide-react';
import { api, ApiError } from '@/lib/api';
import { isAdmin } from '@/lib/auth';
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
import { useToast } from '@/components/ui/use-toast';

interface Org {
  id: number;
  name: string;
}

interface SMBShareStatus {
  host: string;
  share: string;
  connected: boolean;
  last_scan_at: number | null;
  last_error: string | null;
}

interface WatcherStatus {
  org_id: number;
  name: string;
  watch_dir: string | null;
  status: 'watching' | 'stopped' | 'not_configured';
  last_scan_at: number | null;
  files_scanned: number;
  smb_watches: SMBShareStatus[];
}

interface ScanResult {
  scanned: number;
  new: number;
  skipped: number;
  errors: number;
}

const STATUS_CONFIG: Record<string, { cls: string; label: string }> = {
  watching: { cls: 'bg-green-100 text-green-700', label: 'Watching' },
  stopped: { cls: 'bg-yellow-100 text-yellow-700', label: 'Stopped' },
  not_configured: { cls: 'bg-gray-100 text-gray-500', label: '—' },
};

const SMB_STATUS_CONFIG: Record<string, { cls: string; label: string }> = {
  connected: { cls: 'bg-emerald-100 text-emerald-700', label: 'Connected' },
  disconnected: { cls: 'bg-red-100 text-red-700', label: 'Disconnected' },
};

function StatusBadge({ status }: { status: string }) {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.not_configured;
  return (
    <Badge variant="secondary" className={config.cls}>
      {config.label}
    </Badge>
  );
}

function SMBStatusBadge({ share }: { share: SMBShareStatus }) {
  const connected = share.connected;
  const config = connected
    ? SMB_STATUS_CONFIG.connected
    : SMB_STATUS_CONFIG.disconnected;
  return (
    <div className="flex flex-col gap-1">
      <Badge variant="secondary" className={config.cls}>
        {config.label}
      </Badge>
      <span className="text-[10px] text-muted-foreground truncate max-w-[120px]" title={`${share.host}/${share.share}`}>
        {share.host}/{share.share}
      </span>
      {share.last_error && (
        <span className="text-[10px] text-red-500 truncate max-w-[120px]" title={share.last_error}>
          ⚠ {share.last_error.slice(0, 40)}
        </span>
      )}
    </div>
  );
}

export default function AdminWatcherPage() {
  const router = useRouter();
  const { toast } = useToast();

  const [orgs, setOrgs] = useState<Org[]>([]);
  const [watcherStatuses, setWatcherStatuses] = useState<Record<number, WatcherStatus>>({});
  const [loading, setLoading] = useState(true);
  const [triggering, setTriggering] = useState<Set<number>>(new Set());

  // Set watch dir dialog
  const [dirDialogOpen, setDirDialogOpen] = useState(false);
  const [selectedOrg, setSelectedOrg] = useState<Org | null>(null);
  const [watchDirInput, setWatchDirInput] = useState('');
  const [watchDirSaving, setWatchDirSaving] = useState(false);

  useEffect(() => {
    if (!isAdmin()) {
      router.replace('/dashboard');
      return;
    }
    fetchData();
  }, [router]);

  async function fetchData() {
    try {
      const [orgsData, statusData] = await Promise.all([
        api.get('/api/admin/orgs') as Promise<Org[]>,
        api.get('/api/admin/watcher-status-all') as Promise<{ orgs: WatcherStatus[] }>,
      ]);
      setOrgs(orgsData);

      const statusMap: Record<number, WatcherStatus> = {};
      (statusData.orgs ?? []).forEach((s) => {
        statusMap[s.org_id] = s;
      });
      setWatcherStatuses(statusMap);
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to load watcher status',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  }

  async function handleTriggerScan(orgId: number) {
    setTriggering((prev) => new Set(prev).add(orgId));
    try {
      const result = await api.post(
        `/api/admin/orgs/${orgId}/watcher-trigger`
      ) as ScanResult;
      toast({
        title: 'Scan completed',
        description:
          `Scanned: ${result.scanned} | New: ${result.new} | Skipped: ${result.skipped} | Errors: ${result.errors}`,
      });
      // Refresh status to update last_scan_at and files_scanned
      fetchData();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? 'Failed to trigger scan',
        variant: 'destructive',
      });
    } finally {
      setTriggering((prev) => {
        const next = new Set(prev);
        next.delete(orgId);
        return next;
      });
    }
  }

  function openSetDir(org: Org) {
    setSelectedOrg(org);
    const existing = watcherStatuses[org.id];
    setWatchDirInput(existing?.watch_dir ?? '');
    setDirDialogOpen(true);
  }

  async function handleSetDir() {
    if (!selectedOrg || !watchDirInput.trim()) return;
    setWatchDirSaving(true);
    try {
      await api.post(`/api/admin/orgs/${selectedOrg.id}/watch-dir`, {
        watch_dir: watchDirInput.trim(),
      });
      toast({ title: 'Watch directory set' });
      setDirDialogOpen(false);
      await fetchData();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? 'Failed to set watch directory',
        variant: 'destructive',
      });
    } finally {
      setWatchDirSaving(false);
    }
  }

  async function handleRemove(org: Org) {
    try {
      await api.delete(`/api/admin/orgs/${org.id}/watch-dir`);
      toast({ title: 'Watch directory removed' });
      await fetchData();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? 'Failed to remove watch directory',
        variant: 'destructive',
      });
    }
  }

  const formatLastScan = (timestamp: number | null) => {
    if (!timestamp) return 'never';
    return new Date(timestamp * 1000).toLocaleString();
  };

  const getDisplayRows = (): Array<WatcherStatus> => {
    const rows: WatcherStatus[] = [];
    for (const org of orgs) {
      const status = watcherStatuses[org.id];
      if (status) {
        rows.push(status);
      } else {
        rows.push({
          org_id: org.id,
          name: org.name,
          watch_dir: null,
          status: 'not_configured',
          last_scan_at: null,
          files_scanned: 0,
          smb_watches: [],
        });
      }
    }
    return rows;
  };

  const displayRows = getDisplayRows();

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Watcher Management</h1>
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Org Name</TableHead>
              <TableHead>Watch Dir</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Last Scan</TableHead>
              <TableHead>Files Scanned</TableHead>
              <TableHead>SMB Shares</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {displayRows.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="text-center text-muted-foreground">
                  No organisations configured.
                </TableCell>
              </TableRow>
            ) : (
              displayRows.map((row) => (
                <TableRow key={row.org_id}>
                  <TableCell className="font-medium">{row.name}</TableCell>
                  <TableCell className="text-xs text-muted-foreground max-w-[200px] truncate">
                    {row.watch_dir ?? '—'}
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={row.status} />
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {formatLastScan(row.last_scan_at)}
                  </TableCell>
                  <TableCell>{row.files_scanned}</TableCell>
                  <TableCell>
                    {row.smb_watches.length > 0 ? (
                      <div className="space-y-1">
                        {row.smb_watches.map((sw, i) => (
                          <SMBStatusBadge key={i} share={sw} />
                        ))}
                      </div>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                  </TableCell>
                  <TableCell className="space-x-1">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleTriggerScan(row.org_id)}
                      disabled={triggering.has(row.org_id) || row.status === 'not_configured'}
                      title="Trigger scan"
                    >
                      <Play className="h-3 w-3 mr-1" />
                      Trigger Scan
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => openSetDir({ id: row.org_id, name: row.name })}
                      title="Set watch directory"
                    >
                      <FolderOpen className="h-3 w-3 mr-1" />
                      Set Watch Dir
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => handleRemove({ id: row.org_id, name: row.name })}
                      disabled={row.status === 'not_configured'}
                      title="Remove watch directory"
                    >
                      <Trash2 className="h-3 w-3 mr-1" />
                      Remove
                    </Button>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      )}

      {/* Set Watch Dir dialog */}
      <Dialog open={dirDialogOpen} onOpenChange={setDirDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              Set Watch Directory — {selectedOrg?.name}
            </DialogTitle>
          </DialogHeader>
          <p className="text-xs text-muted-foreground pb-1">
            Enter the absolute path to the directory to watch for file changes.
          </p>
          <div className="py-2">
            <Input
              placeholder="/path/to/watch"
              value={watchDirInput}
              onChange={(e) => setWatchDirInput(e.target.value)}
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDirDialogOpen(false)}
              disabled={watchDirSaving}
            >
              Cancel
            </Button>
            <Button
              onClick={handleSetDir}
              disabled={watchDirSaving || !watchDirInput.trim()}
            >
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
