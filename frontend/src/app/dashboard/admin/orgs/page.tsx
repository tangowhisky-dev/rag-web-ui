'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { api, ApiError } from '@/lib/api';
import { isAdmin } from '@/lib/auth';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useToast } from '@/components/ui/use-toast';

interface Org {
  id: number;
  name: string;
  parent_id: number | null;
  path: string;
}

interface OrgIngestionStatus {
  org_id: number;
  status: 'idle' | 'running' | 'completed' | 'failed';
  total_docs: number;
  pending_docs: number;
  processing_docs: number;
  completed_docs: number;
  failed_docs: number;
  last_run_at: string | null;
}

const STATUS_COLORS: Record<string, string> = {
  idle: 'bg-gray-100 text-gray-600',
  running: 'bg-blue-100 text-blue-700',
  completed: 'bg-green-100 text-green-700',
  failed: 'bg-red-100 text-red-700',
};

function IngestionBadge({ status }: { status: OrgIngestionStatus | undefined }) {
  if (!status) return <span className="text-xs text-muted-foreground">—</span>;

  const cls = STATUS_COLORS[status.status] ?? STATUS_COLORS.idle;
  const lastRun = status.last_run_at
    ? new Date(status.last_run_at).toLocaleString()
    : 'never';
  const tooltip =
    `total: ${status.total_docs} | pending: ${status.pending_docs} | ` +
    `processing: ${status.processing_docs} | completed: ${status.completed_docs} | ` +
    `failed: ${status.failed_docs} | last run: ${lastRun}`;

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}
      title={tooltip}
    >
      {status.status}
      {status.total_docs > 0 && (
        <span className="opacity-70">{status.total_docs} docs</span>
      )}
    </span>
  );
}

export default function AdminOrgsPage() {
  const router = useRouter();
  const { toast } = useToast();

  const [orgs, setOrgs] = useState<Org[]>([]);
  const [loading, setLoading] = useState(true);
  const [ingestionStatuses, setIngestionStatuses] = useState<Record<number, OrgIngestionStatus>>({});

  // Create dialog
  const [createOpen, setCreateOpen] = useState(false);
  const [newOrgName, setNewOrgName] = useState('');
  const [newOrgParentId, setNewOrgParentId] = useState<string>('');
  const [creating, setCreating] = useState(false);

  // Rename dialog
  const [renameOpen, setRenameOpen] = useState(false);
  const [selectedOrg, setSelectedOrg] = useState<Org | null>(null);
  const [renameName, setRenameName] = useState('');
  const [renaming, setRenaming] = useState(false);

  // Delete dialog
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // LLM config dialog
  const [llmConfigOpen, setLlmConfigOpen] = useState(false);
  const [llmConfigOrg, setLlmConfigOrg] = useState<Org | null>(null);
  const [llmApiBase, setLlmApiBase] = useState('');
  const [llmModelName, setLlmModelName] = useState('');
  const [llmQueryModel, setLlmQueryModel] = useState('');
  const [llmSaving, setLlmSaving] = useState(false);

  useEffect(() => {
    if (!isAdmin()) {
      router.replace('/dashboard');
      return;
    }
    fetchOrgs();
  }, [router]);

  async function fetchIngestionStatuses(orgList: Org[]) {
    const results = await Promise.allSettled(
      orgList.map((org) =>
        api.get(`/api/admin/orgs/${org.id}/ingestion-status`) as Promise<OrgIngestionStatus>
      )
    );
    const map: Record<number, OrgIngestionStatus> = {};
    results.forEach((result, idx) => {
      if (result.status === 'fulfilled') {
        map[orgList[idx].id] = result.value;
      }
    });
    setIngestionStatuses(map);
  }

  async function fetchOrgs() {
    try {
      const data = await api.get('/api/admin/orgs');
      setOrgs(data);
      fetchIngestionStatuses(data);
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to load organisations',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  }

  async function handleCreate() {
    if (!newOrgName.trim()) return;
    setCreating(true);
    try {
      await api.post('/api/admin/orgs', {
        name: newOrgName.trim(),
        parent_id: newOrgParentId ? parseInt(newOrgParentId, 10) : null,
      });
      toast({ title: 'Organisation created' });
      setCreateOpen(false);
      setNewOrgName('');
      setNewOrgParentId('');
      await fetchOrgs();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to create organisation',
        variant: 'destructive',
      });
    } finally {
      setCreating(false);
    }
  }

  function openRename(org: Org) {
    setSelectedOrg(org);
    setRenameName(org.name);
    setRenameOpen(true);
  }

  async function handleRename() {
    if (!selectedOrg || !renameName.trim()) return;
    setRenaming(true);
    try {
      await api.patch(`/api/admin/orgs/${selectedOrg.id}`, { name: renameName.trim() });
      toast({ title: 'Organisation renamed' });
      setRenameOpen(false);
      await fetchOrgs();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to rename organisation',
        variant: 'destructive',
      });
    } finally {
      setRenaming(false);
    }
  }

  function openDelete(org: Org) {
    setSelectedOrg(org);
    setDeleteOpen(true);
  }

  async function handleDelete() {
    if (!selectedOrg) return;
    setDeleting(true);
    try {
      await api.delete(`/api/admin/orgs/${selectedOrg.id}`);
      toast({ title: 'Organisation deleted' });
      setDeleteOpen(false);
      await fetchOrgs();
    } catch (err) {
      const apiErr = err as ApiError;
      if (apiErr.status === 409) {
        toast({
          title: 'Cannot delete',
          description: 'Organisation has children or assigned users.',
          variant: 'destructive',
        });
      } else {
        toast({
          title: 'Error',
          description: apiErr.message ?? 'Failed to delete organisation',
          variant: 'destructive',
        });
      }
      setDeleteOpen(false);
    } finally {
      setDeleting(false);
    }
  }

  async function openLlmConfig(org: Org) {
    setLlmConfigOrg(org);
    setLlmApiBase('');
    setLlmModelName('');
    setLlmQueryModel('');
    try {
      const data = await api.get(`/api/admin/orgs/${org.id}/llm-config`);
      setLlmApiBase(data.api_base ?? '');
      setLlmModelName(data.model_name ?? '');
      setLlmQueryModel(data.query_model ?? '');
    } catch (err) {
      // 404 means no config yet — leave fields empty
      const apiErr = err as ApiError;
      if (apiErr.status !== 404) {
        toast({
          title: 'Error',
          description: apiErr.message ?? 'Failed to load LLM config',
          variant: 'destructive',
        });
        return;
      }
    }
    setLlmConfigOpen(true);
  }

  async function handleLlmConfigSave(clearAll: boolean) {
    if (!llmConfigOrg) return;
    setLlmSaving(true);
    try {
      await api.put(`/api/admin/orgs/${llmConfigOrg.id}/llm-config`, {
        api_base: clearAll ? null : (llmApiBase.trim() || null),
        model_name: clearAll ? null : (llmModelName.trim() || null),
        query_model: clearAll ? null : (llmQueryModel.trim() || null),
      });
      toast({ title: clearAll ? 'LLM config cleared' : 'LLM config saved' });
      setLlmConfigOpen(false);
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as ApiError).message ?? 'Failed to save LLM config',
        variant: 'destructive',
      });
    } finally {
      setLlmSaving(false);
    }
  }

  const parentName = (parentId: number | null) =>
    parentId ? (orgs.find((o) => o.id === parentId)?.name ?? String(parentId)) : '—';

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Organisations</h1>
        <Button onClick={() => setCreateOpen(true)}>+ New Org</Button>
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>ID</TableHead>
              <TableHead>Name</TableHead>
              <TableHead>Parent</TableHead>
              <TableHead>Path</TableHead>
              <TableHead>Ingestion Status</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {orgs.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="text-center text-muted-foreground">
                  No organisations yet.
                </TableCell>
              </TableRow>
            ) : (
              orgs.map((org) => (
                <TableRow key={org.id}>
                  <TableCell>{org.id}</TableCell>
                  <TableCell className="font-medium">{org.name}</TableCell>
                  <TableCell>{parentName(org.parent_id)}</TableCell>
                  <TableCell className="text-xs text-muted-foreground">{org.path}</TableCell>
                  <TableCell>
                    <IngestionBadge status={ingestionStatuses[org.id]} />
                  </TableCell>
                  <TableCell className="space-x-2">
                    <Button variant="outline" size="sm" onClick={() => openRename(org)}>
                      Rename
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => openLlmConfig(org)}>
                      LLM Config
                    </Button>
                    <Button variant="destructive" size="sm" onClick={() => openDelete(org)}>
                      Delete
                    </Button>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      )}

      {/* Create dialog */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>New Organisation</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <Input
              placeholder="Organisation name"
              value={newOrgName}
              onChange={(e) => setNewOrgName(e.target.value)}
            />
            <select
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={newOrgParentId}
              onChange={(e) => setNewOrgParentId(e.target.value)}
            >
              <option value="">No parent</option>
              {orgs.map((o) => (
                <option key={o.id} value={String(o.id)}>
                  {o.name}
                </option>
              ))}
            </select>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)} disabled={creating}>
              Cancel
            </Button>
            <Button onClick={handleCreate} disabled={creating || !newOrgName.trim()}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Rename dialog */}
      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename Organisation</DialogTitle>
          </DialogHeader>
          <div className="py-2">
            <Input
              placeholder="New name"
              value={renameName}
              onChange={(e) => setRenameName(e.target.value)}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenameOpen(false)} disabled={renaming}>
              Cancel
            </Button>
            <Button onClick={handleRename} disabled={renaming || !renameName.trim()}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* LLM config dialog */}
      <Dialog open={llmConfigOpen} onOpenChange={setLlmConfigOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>LLM Config — {llmConfigOrg?.name}</DialogTitle>
          </DialogHeader>
          <p className="text-xs text-muted-foreground pb-1">
            Leave fields blank to inherit from .env defaults.
          </p>
          <div className="space-y-3 py-2">
            <div className="space-y-1">
              <label className="text-sm font-medium">API Base URL</label>
              <Input
                placeholder="https://api.openai.com/v1"
                value={llmApiBase}
                onChange={(e) => setLlmApiBase(e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">Model Name</label>
              <Input
                placeholder="gpt-4o"
                value={llmModelName}
                onChange={(e) => setLlmModelName(e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">Query Model</label>
              <Input
                placeholder="gpt-4o-mini"
                value={llmQueryModel}
                onChange={(e) => setLlmQueryModel(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => handleLlmConfigSave(true)}
              disabled={llmSaving}
            >
              Clear
            </Button>
            <Button onClick={() => handleLlmConfigSave(false)} disabled={llmSaving}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete confirmation dialog */}
      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Organisation</DialogTitle>
          </DialogHeader>
          <p className="text-sm py-2">
            Are you sure you want to delete <strong>{selectedOrg?.name}</strong>? This cannot be
            undone.
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteOpen(false)} disabled={deleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={deleting}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
