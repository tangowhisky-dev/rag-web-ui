'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { api, ApiError } from '@/lib/api';
import { fetchTokenClaims } from '@/lib/auth';
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
  level: number;
  user_count: number;
  hierarchy_name: string;
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
  const [isSuperAdmin, setIsSuperAdmin] = useState(false);
  const [currentOrgId, setCurrentOrgId] = useState<number | null>(null);

  // Create dialog
  const [createOpen, setCreateOpen] = useState(false);
  const [newOrgName, setNewOrgName] = useState('');
  const [newOrgParentId, setNewOrgParentId] = useState<string>('');
  const [creating, setCreating] = useState(false);

  // Edit dialog
  const [editOpen, setEditOpen] = useState(false);
  const [selectedOrg, setSelectedOrg] = useState<Org | null>(null);
  const [editName, setEditName] = useState('');
  const [editParentId, setEditParentId] = useState<string>('');
  const [editing, setEditing] = useState(false);

  // Search
  const [search, setSearch] = useState('');

  // Delete dialog
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const fetchIngestionStatuses = useCallback(async (orgList: Org[]) => {
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
  }, []);

  const fetchOrgs = useCallback(async () => {
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
  }, [toast, fetchIngestionStatuses]);

  useEffect(() => {
    fetchTokenClaims().then((claims) => {
      setIsSuperAdmin(claims?.role === 'super_admin');
      setCurrentOrgId(claims?.org_id ?? null);
    });
    let cancelled = false;
    (async () => {
      if (!cancelled) await fetchOrgs();
    })();
    return () => { cancelled = true; };
  }, [router, fetchOrgs]);

  async function handleCreate() {
    if (!newOrgName.trim()) return;
    setCreating(true);
    try {
      await api.post('/api/admin/orgs', {
        name: newOrgName.trim(),
        parent_id: parseInt(newOrgParentId, 10),
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

  function openEdit(org: Org) {
    setSelectedOrg(org);
    setEditName(org.name);
    // For edit: parent_id can be any org except itself
    // If self-referencing root, default to first available org
    const validParent = org.parent_id !== org.id ? String(org.parent_id) : '';
    setEditParentId(validParent || (orgs.length > 0 ? String(orgs[0].id) : ''));
    setEditOpen(true);
  }

  async function handleEdit() {
    if (!selectedOrg || !editName.trim() || !editParentId) return;
    setEditing(true);
    try {
      await api.patch(`/api/admin/orgs/${selectedOrg.id}`, {
        name: editName.trim(),
        parent_id: parseInt(editParentId, 10),
      });
      toast({ title: 'Organisation updated' });
      setEditOpen(false);
      await fetchOrgs();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to update organisation',
        variant: 'destructive',
      });
    } finally {
      setEditing(false);
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

  // Filter by search (case-insensitive on name)
  const filteredOrgs = orgs.filter((o) =>
    o.name.toLowerCase().includes(search.toLowerCase())
  );

  // Check if candidateOrg is a descendant of org (or org itself)
  const isDescendantOf = (candidateOrg: Org, ancestorOrg: Org): boolean => {
    if (!candidateOrg.path || !ancestorOrg.path) return false;
    // ancestor's path must be a prefix of candidate's path
    return candidateOrg.path.startsWith(ancestorOrg.path + '/') || candidateOrg.id === ancestorOrg.id;
  };

  // For edit dialog: exclude self and all descendants from parent options
  const availableParents = selectedOrg
    ? orgs.filter((o) => !isDescendantOf(o, selectedOrg))
    : orgs;

  // Whether the current user can edit/delete this org.
  // super_admin: yes for any org.
  // org admin: yes only for strict descendants of their own org.
  const canManageOrg = (org: Org): boolean => {
    if (isSuperAdmin) return true;
    if (currentOrgId === null) return false;
    const myOrg = orgs.find((o) => o.id === currentOrgId);
    if (!myOrg || !myOrg.path || !org.path) return false;
    return org.path.startsWith(myOrg.path + '/') && org.id !== currentOrgId;
  };

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 pt-16 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Organisations</h1>
          <p className="text-muted-foreground">Manage your organisations and LLM configurations</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="relative">
            <input
              type="text"
              placeholder="Search organisations…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-64 rounded-md border border-input bg-background pl-9 pr-3 py-2 text-sm"
            />
            <svg className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <Button onClick={() => {
            setNewOrgParentId(orgs.length > 0 ? String(orgs[0].id) : '');
            setCreateOpen(true);
          }}>+ New Organization</Button>
        </div>
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>ID</TableHead>
              <TableHead>Name</TableHead>
              <TableHead>Users</TableHead>
              <TableHead>Ingestion Status</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {orgs.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="text-center text-muted-foreground">
                  No organisations yet.
                </TableCell>
              </TableRow>
            ) : (
              filteredOrgs.map((org) => (
                <TableRow key={org.id} title={org.hierarchy_name || org.name}>
                  <TableCell>{org.id}</TableCell>
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-1">
                      {org.level > 0 && (
                        <span className="text-muted-foreground text-xs select-none">
                          {'—'.repeat(org.level)}
                        </span>
                      )}
                      <span>{org.name}</span>
                    </div>
                  </TableCell>
                  <TableCell>{org.user_count}</TableCell>
                  <TableCell>
                    <IngestionBadge status={ingestionStatuses[org.id]} />
                  </TableCell>
                  <TableCell className="space-x-2">
                    {canManageOrg(org) && (
                      <Button variant="outline" size="sm" onClick={() => openEdit(org)} title="Change name and parent organization">
                        Edit
                      </Button>
                    )}
                    <Link href={`/dashboard/admin/orgs/${org.id}/settings`}>
                      <Button variant="outline" size="sm" title="Full organisation settings (retrieval, agentic, memory, etc.)">
                        Settings
                      </Button>
                    </Link>
                    {canManageOrg(org) && org.parent_id !== null && (
                      <Button variant="destructive" size="sm" onClick={() => openDelete(org)} title="Permanently delete this organization">
                        Delete
                      </Button>
                    )}
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
            <div className="space-y-1">
              <label className="text-sm font-medium">Organisation name</label>
              <Input
                placeholder="Organisation name"
                value={newOrgName}
                onChange={(e) => setNewOrgName(e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">Parent organization</label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={newOrgParentId}
                onChange={(e) => setNewOrgParentId(e.target.value)}
              >
                {orgs.map((o) => (
                  <option key={o.id} value={String(o.id)}>
                    {o.name}
                  </option>
                ))}
              </select>
            </div>
            {orgs.length === 0 && (
              <p className="text-xs text-muted-foreground">No organisations available — cannot create child org.</p>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)} disabled={creating}>
              Cancel
            </Button>
            <Button onClick={handleCreate} disabled={creating || !newOrgName.trim() || !newOrgParentId}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit dialog */}
      <Dialog open={editOpen} onOpenChange={setEditOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Organisation — {selectedOrg?.name}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div className="space-y-1">
              <label className="text-sm font-medium">Organisation name</label>
              <Input
                placeholder="Organisation name"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">Parent organization</label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={editParentId}
                onChange={(e) => setEditParentId(e.target.value)}
              >
                {availableParents.map((o) => (
                  <option key={o.id} value={String(o.id)}>
                    {o.name}
                  </option>
                ))}
              </select>
            </div>
            {availableParents.length === 0 && (
              <p className="text-xs text-muted-foreground">No organisations available — cannot edit.</p>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditOpen(false)} disabled={editing}>
              Cancel
            </Button>
            <Button onClick={handleEdit} disabled={editing || !editName.trim() || !editParentId}>
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
