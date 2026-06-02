'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { api, ApiError } from '@/lib/api';
import { isAdmin } from '@/lib/auth';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
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

interface User {
  id: number;
  username: string;
  email: string;
  role: string;
  org_id: number | null;
  is_active: boolean;
}

interface Org {
  id: number;
  name: string;
}

type RoleOption = 'user' | 'admin' | 'super_admin';

const ROLE_OPTIONS: RoleOption[] = ['user', 'admin', 'super_admin'];

function roleBadgeVariant(role: string): 'default' | 'secondary' | 'destructive' {
  if (role === 'super_admin') return 'destructive';
  if (role === 'admin') return 'secondary';
  return 'default';
}

export default function AdminUsersPage() {
  const router = useRouter();
  const { toast } = useToast();

  const [users, setUsers] = useState<User[]>([]);
  const [orgs, setOrgs] = useState<Org[]>([]);
  const [loading, setLoading] = useState(true);

  // Create dialog
  const [createOpen, setCreateOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createForm, setCreateForm] = useState({
    username: '',
    email: '',
    password: '',
    role: 'user' as RoleOption,
    org_id: '',
  });

  // Edit dialog
  const [editOpen, setEditOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [editForm, setEditForm] = useState({
    role: 'user' as RoleOption,
    org_id: '',
    is_active: true,
  });

  useEffect(() => {
    if (!isAdmin()) {
      router.replace('/dashboard');
      return;
    }
    fetchAll();
  }, [router]);

  async function fetchAll() {
    try {
      const [usersData, orgsData] = await Promise.all([
        api.get('/api/admin/users'),
        api.get('/api/admin/orgs'),
      ]);
      setUsers(usersData);
      setOrgs(orgsData);
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to load data',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  }

  async function handleCreate() {
    setCreating(true);
    try {
      await api.post('/api/admin/users', {
        username: createForm.username.trim(),
        email: createForm.email.trim(),
        password: createForm.password,
        role: createForm.role,
        org_id: createForm.org_id ? parseInt(createForm.org_id, 10) : null,
      });
      toast({ title: 'User created' });
      setCreateOpen(false);
      setCreateForm({ username: '', email: '', password: '', role: 'user', org_id: '' });
      await fetchAll();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to create user',
        variant: 'destructive',
      });
    } finally {
      setCreating(false);
    }
  }

  function openEdit(user: User) {
    setSelectedUser(user);
    setEditForm({
      role: user.role as RoleOption,
      org_id: user.org_id ? String(user.org_id) : '',
      is_active: user.is_active,
    });
    setEditOpen(true);
  }

  async function handleEdit() {
    if (!selectedUser) return;
    setEditing(true);
    try {
      await api.patch(`/api/admin/users/${selectedUser.id}`, {
        role: editForm.role,
        org_id: editForm.org_id ? parseInt(editForm.org_id, 10) : null,
        is_active: editForm.is_active,
      });
      toast({ title: 'User updated' });
      setEditOpen(false);
      await fetchAll();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to update user',
        variant: 'destructive',
      });
    } finally {
      setEditing(false);
    }
  }

  async function handleDeactivate(user: User) {
    try {
      await api.delete(`/api/admin/users/${user.id}`);
      toast({ title: `User ${user.username} deactivated` });
      await fetchAll();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to deactivate user',
        variant: 'destructive',
      });
    }
  }

  const orgName = (orgId: number | null) =>
    orgId ? (orgs.find((o) => o.id === orgId)?.name ?? String(orgId)) : '—';

  const createValid =
    createForm.username.trim() && createForm.email.trim() && createForm.password.length >= 1;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Users</h1>
        <Button onClick={() => setCreateOpen(true)}>+ New User</Button>
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>ID</TableHead>
              <TableHead>Username</TableHead>
              <TableHead>Email</TableHead>
              <TableHead>Role</TableHead>
              <TableHead>Org</TableHead>
              <TableHead>Active</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {users.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="text-center text-muted-foreground">
                  No users yet.
                </TableCell>
              </TableRow>
            ) : (
              users.map((user) => (
                <TableRow key={user.id}>
                  <TableCell>{user.id}</TableCell>
                  <TableCell className="font-medium">{user.username}</TableCell>
                  <TableCell>{user.email}</TableCell>
                  <TableCell>
                    <Badge variant={roleBadgeVariant(user.role)}>{user.role}</Badge>
                  </TableCell>
                  <TableCell>{orgName(user.org_id)}</TableCell>
                  <TableCell>{user.is_active ? 'Yes' : 'No'}</TableCell>
                  <TableCell className="space-x-2">
                    <Button variant="outline" size="sm" onClick={() => openEdit(user)}>
                      Edit
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      disabled={!user.is_active}
                      onClick={() => handleDeactivate(user)}
                    >
                      Deactivate
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
            <DialogTitle>New User</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <Input
              placeholder="Username"
              value={createForm.username}
              onChange={(e) => setCreateForm((f) => ({ ...f, username: e.target.value }))}
            />
            <Input
              placeholder="Email"
              type="email"
              value={createForm.email}
              onChange={(e) => setCreateForm((f) => ({ ...f, email: e.target.value }))}
            />
            <Input
              placeholder="Password"
              type="password"
              value={createForm.password}
              onChange={(e) => setCreateForm((f) => ({ ...f, password: e.target.value }))}
            />
            <select
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={createForm.role}
              onChange={(e) =>
                setCreateForm((f) => ({ ...f, role: e.target.value as RoleOption }))
              }
            >
              {ROLE_OPTIONS.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
            <select
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={createForm.org_id}
              onChange={(e) => setCreateForm((f) => ({ ...f, org_id: e.target.value }))}
            >
              <option value="">No organisation</option>
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
            <Button onClick={handleCreate} disabled={creating || !createValid}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit dialog */}
      <Dialog open={editOpen} onOpenChange={setEditOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit User — {selectedUser?.username}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <select
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={editForm.role}
              onChange={(e) => setEditForm((f) => ({ ...f, role: e.target.value as RoleOption }))}
            >
              {ROLE_OPTIONS.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
            <select
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={editForm.org_id}
              onChange={(e) => setEditForm((f) => ({ ...f, org_id: e.target.value }))}
            >
              <option value="">No organisation</option>
              {orgs.map((o) => (
                <option key={o.id} value={String(o.id)}>
                  {o.name}
                </option>
              ))}
            </select>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={editForm.is_active}
                onChange={(e) => setEditForm((f) => ({ ...f, is_active: e.target.checked }))}
                className="h-4 w-4"
              />
              Active
            </label>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditOpen(false)} disabled={editing}>
              Cancel
            </Button>
            <Button onClick={handleEdit} disabled={editing}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
