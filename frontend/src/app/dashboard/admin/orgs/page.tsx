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

export default function AdminOrgsPage() {
  const router = useRouter();
  const { toast } = useToast();

  const [orgs, setOrgs] = useState<Org[]>([]);
  const [loading, setLoading] = useState(true);

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

  useEffect(() => {
    if (!isAdmin()) {
      router.replace('/dashboard');
      return;
    }
    fetchOrgs();
  }, [router]);

  async function fetchOrgs() {
    try {
      const data = await api.get('/api/admin/orgs');
      setOrgs(data);
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
              orgs.map((org) => (
                <TableRow key={org.id}>
                  <TableCell>{org.id}</TableCell>
                  <TableCell className="font-medium">{org.name}</TableCell>
                  <TableCell>{parentName(org.parent_id)}</TableCell>
                  <TableCell className="text-xs text-muted-foreground">{org.path}</TableCell>
                  <TableCell className="space-x-2">
                    <Button variant="outline" size="sm" onClick={() => openRename(org)}>
                      Rename
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
