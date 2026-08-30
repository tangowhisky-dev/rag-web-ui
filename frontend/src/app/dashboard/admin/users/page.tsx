'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { validatePasswordStrength } from '@/lib/auth';
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

interface EditForm {
  role: RoleOption;
  org_id: string;
  is_active: boolean;
}

interface PasswordForm {
  new_password: string;
  confirm_password: string;
}

type SetState<T> = (value: T | ((prev: T) => T)) => void;

const ALL_ROLE_OPTIONS: RoleOption[] = ['user', 'admin', 'super_admin'];

const adminRoleOptions: RoleOption[] = ['user'];

function roleBadgeVariant(role: string): 'default' | 'secondary' | 'destructive' {
  if (role === 'super_admin') return 'destructive';
  if (role === 'admin') return 'secondary';
  return 'default';
}

function canChangePassword(user: User, currentRole: string): boolean {
  return currentRole === 'super_admin' || user.role === 'user';
}

function canEditUser(user: User, currentRole: string): boolean {
  return currentRole === 'super_admin' || user.role === 'user';
}

function canDeleteUser(user: User, currentRole: string): boolean {
  return currentRole === 'super_admin' && user.role !== 'super_admin';
}

function validateChangePassword(passwords: PasswordForm): string | null {
  const passwordError = validatePasswordStrength(passwords.new_password);
  if (passwordError) return passwordError;
  if (passwords.new_password !== passwords.confirm_password) return 'Passwords do not match';
  return null;
}

function getOrgName(orgs: Org[], orgId: number | null): string {
  return orgId ? (orgs.find((o) => o.id === orgId)?.name ?? String(orgId)) : '—';
}

function filterUsers(users: User[], search: string, orgs: Org[]): User[] {
  if (!search.trim()) return users;
  const q = search.toLowerCase();
  return users.filter(
    (u) =>
      u.username.toLowerCase().includes(q) ||
      u.email.toLowerCase().includes(q) ||
      getOrgName(orgs, u.org_id).toLowerCase().includes(q),
  );
}

function buildEditForm(user: User, currentRole: string, orgs: Org[]): EditForm {
  const isSuperAdmin = currentRole === 'super_admin';
  const orgId = user.org_id || (orgs.length > 0 ? orgs[0].id : null);
  return {
    role: isSuperAdmin ? (user.role as RoleOption) : 'user',
    org_id: orgId ? String(orgId) : '',
    is_active: user.is_active,
  };
}

function isCreateValid(form: { username: string; password: string; org_id: string }): boolean {
  return !!(form.username.trim() && form.password.length >= 1 && !!form.org_id);
}

function isEditValid(selectedUser: User | null, editForm: EditForm, orgs: Org[]): boolean {
  return !!selectedUser && !!editForm.org_id && orgs.length > 0;
}

interface UserTableProps {
  loading: boolean;
  users: User[];
  search: string;
  orgs: Org[];
  currentRole: string;
  onEdit: (user: User) => void;
  onDelete: (user: User) => void;
  onChangePassword: (user: User) => void;
}

function UserTable({
  loading,
  users,
  search,
  orgs,
  currentRole,
  onEdit,
  onDelete,
  onChangePassword,
}: UserTableProps) {
  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading…</p>;
  }
  const filtered = filterUsers(users, search, orgs);
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
          <TableHead>Username</TableHead>
          <TableHead>Email</TableHead>
          <TableHead>Role</TableHead>
          <TableHead>Org</TableHead>
          <TableHead>Status</TableHead>
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
        ) : filtered.length === 0 ? (
          <TableRow>
            <TableCell colSpan={7} className="text-center text-muted-foreground">
              No users match &quot;{search}&quot;.
            </TableCell>
          </TableRow>
        ) : (
          filtered.map((user) => (
            <TableRow key={user.id}>
              <TableCell>{user.id}</TableCell>
              <TableCell className="font-medium">{user.username}</TableCell>
              <TableCell>{user.email}</TableCell>
              <TableCell>
                <Badge variant={roleBadgeVariant(user.role)}>{user.role}</Badge>
              </TableCell>
              <TableCell>{getOrgName(orgs, user.org_id)}</TableCell>
              <TableCell>
                {user.is_active ? (
                  <Badge variant="default" className="bg-green-100 text-green-700 hover:bg-green-100">
                    Active
                  </Badge>
                ) : (
                  <Badge variant="secondary" className="bg-red-100 text-red-700 hover:bg-red-100">
                    Deactivated
                  </Badge>
                )}
              </TableCell>
              <TableCell className="space-x-1">
                {canChangePassword(user, currentRole) && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onChangePassword(user)}
                    title="Change password"
                  >
                    Password
                  </Button>
                )}
                {canEditUser(user, currentRole) && (
                  <Button variant="outline" size="sm" onClick={() => onEdit(user)}>
                    Edit
                  </Button>
                )}
                {canDeleteUser(user, currentRole) && (
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={() => onDelete(user)}
                  >
                    Delete
                  </Button>
                )}
              </TableCell>
            </TableRow>
          ))
        )}
      </TableBody>
    </Table>
  );
}

interface UserDialogsProps {
  editOpen: boolean;
  setEditOpen: (open: boolean) => void;
  editing: boolean;
  selectedUser: User | null;
  editForm: EditForm;
  setEditForm: SetState<EditForm>;
  availableRoles: RoleOption[];
  orgs: Org[];
  currentRole: string;
  editValid: boolean;
  onEdit: () => void;
  passwordOpen: boolean;
  setPasswordOpen: (open: boolean) => void;
  changingPassword: boolean;
  userForPassword: User | null;
  passwordForm: PasswordForm;
  setPasswordForm: SetState<PasswordForm>;
  onChangePassword: () => void;
  deleteOpen: boolean;
  setDeleteOpen: (open: boolean) => void;
  deleting: boolean;
  userToDelete: User | null;
  onDelete: () => void;
}

function UserDialogs({
  editOpen,
  setEditOpen,
  editing,
  selectedUser,
  editForm,
  setEditForm,
  availableRoles,
  orgs,
  currentRole,
  editValid,
  onEdit,
  passwordOpen,
  setPasswordOpen,
  changingPassword,
  userForPassword,
  passwordForm,
  setPasswordForm,
  onChangePassword,
  deleteOpen,
  setDeleteOpen,
  deleting,
  userToDelete,
  onDelete,
}: UserDialogsProps) {
  return (
    <>
      <Dialog open={editOpen} onOpenChange={setEditOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit User — {selectedUser?.username}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div className="space-y-1">
              <label className="text-sm font-medium">Role</label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={editForm.role}
                onChange={(e) => setEditForm((f) => ({ ...f, role: e.target.value as RoleOption }))}
              >
                {selectedUser && currentRole !== 'super_admin' && selectedUser.role !== 'user' ? (
                  <option key={selectedUser.role} value={selectedUser.role} disabled>
                    {selectedUser.role} (cannot change — requires super admin)
                  </option>
                ) : null}
                {availableRoles.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">Organisation *</label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={editForm.org_id}
                onChange={(e) => setEditForm((f) => ({ ...f, org_id: e.target.value }))}
              >
                {orgs.map((o) => (
                  <option key={o.id} value={String(o.id)}>
                    {o.name}
                  </option>
                ))}
              </select>
              {orgs.length === 0 && (
                <p className="text-xs text-muted-foreground">No organisations available — cannot edit user.</p>
              )}
            </div>
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="is-active"
                checked={editForm.is_active}
                onChange={(e) => setEditForm((f) => ({ ...f, is_active: e.target.checked }))}
                className="h-4 w-4"
              />
              <label htmlFor="is-active" className="text-sm font-medium">Active</label>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditOpen(false)} disabled={editing}>
              Cancel
            </Button>
            <Button onClick={onEdit} disabled={editing || !editValid}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={passwordOpen} onOpenChange={setPasswordOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Change Password — {userForPassword?.username}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div className="space-y-1">
              <label className="text-sm font-medium">New Password *</label>
              <Input
                type="password"
                placeholder="Minimum 1 character"
                value={passwordForm.new_password}
                onChange={(e) => setPasswordForm({ ...passwordForm, new_password: e.target.value })}
              />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">Confirm Password *</label>
              <Input
                type="password"
                placeholder="Re-enter new password"
                value={passwordForm.confirm_password}
                onChange={(e) => setPasswordForm({ ...passwordForm, confirm_password: e.target.value })}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPasswordOpen(false)} disabled={changingPassword}>
              Cancel
            </Button>
            <Button
              onClick={onChangePassword}
              disabled={changingPassword || !passwordForm.new_password || passwordForm.new_password !== passwordForm.confirm_password}
            >
              {changingPassword ? 'Changing…' : 'Change Password'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="text-destructive">Permanently Delete User</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <p className="text-sm">
              Are you sure you want to permanently delete <strong>{userToDelete?.username}</strong>? This action:
            </p>
            <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
              <li>Removes the user permanently (cannot be undone)</li>
              <li>Deletes all knowledge bases owned by this user</li>
              <li>Deletes all chats owned by this user</li>
              <li>Deletes all messages, files, and chunks associated with the above</li>
            </ul>
          </div>
          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={() => setDeleteOpen(false)} disabled={deleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={onDelete} disabled={deleting}>
              {deleting ? 'Deleting…' : 'Delete Permanently'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default function AdminUsersPage() {
  const router = useRouter();
  const { toast } = useToast();

  const [users, setUsers] = useState<User[]>([]);
  const [orgs, setOrgs] = useState<Org[]>([]);
  const [loading, setLoading] = useState(true);
  const [availableRoles, setAvailableRoles] = useState<RoleOption[]>([]);
  const [currentRole, setCurrentRole] = useState<string>('');
  const [search, setSearch] = useState('');

  const [createOpen, setCreateOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createForm, setCreateForm] = useState({
    username: '',
    email: '',
    password: '',
    role: 'user' as RoleOption,
    org_id: '',
  });

  const [editOpen, setEditOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [editForm, setEditForm] = useState<EditForm>({
    role: 'user',
    org_id: '',
    is_active: true,
  });

  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [userToDelete, setUserToDelete] = useState<User | null>(null);

  const [passwordOpen, setPasswordOpen] = useState(false);
  const [changingPassword, setChangingPassword] = useState(false);
  const [userForPassword, setUserForPassword] = useState<User | null>(null);
  const [passwordForm, setPasswordForm] = useState<PasswordForm>({
    new_password: '',
    confirm_password: '',
  });

  const fetchAll = useCallback(async () => {
    try {
      const [usersData, orgsData, currentUser] = await Promise.all([
        api.get('/api/admin/users'),
        api.get('/api/admin/orgs'),
        api.get('/api/auth/test-token'),
      ]);
      setUsers(usersData);
      setOrgs(orgsData);
      const role = (currentUser as { role?: string }).role ?? '';
      setCurrentRole(role);
      setAvailableRoles(role === 'super_admin' ? ALL_ROLE_OPTIONS : adminRoleOptions);
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to load data',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!cancelled) await fetchAll();
    })();
    return () => { cancelled = true; };
  }, [router, fetchAll]);

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
    setEditForm(buildEditForm(user, currentRole, orgs));
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

  function openDeleteConfirm(user: User) {
    setUserToDelete(user);
    setDeleteOpen(true);
  }

  async function handleDelete() {
    if (!userToDelete) return;
    setDeleting(true);
    try {
      await api.delete(`/api/admin/users/${userToDelete.id}`);
      toast({ title: `User ${userToDelete.username} permanently deleted` });
      setDeleteOpen(false);
      setUserToDelete(null);
      await fetchAll();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to delete user',
        variant: 'destructive',
      });
    } finally {
      setDeleting(false);
    }
  }

  function openChangePassword(user: User) {
    setUserForPassword(user);
    setPasswordForm({ new_password: '', confirm_password: '' });
    setPasswordOpen(true);
  }

  async function handleChangePassword() {
    if (!userForPassword) return;
    const error = validateChangePassword(passwordForm);
    if (error) {
      toast({
        title: 'Error',
        description: error,
        variant: 'destructive',
      });
      return;
    }
    setChangingPassword(true);
    try {
      await api.post(`/api/admin/users/${userForPassword.id}/change-password`, {
        new_password: passwordForm.new_password,
      });
      toast({ title: 'Password changed successfully' });
      setPasswordOpen(false);
      setUserForPassword(null);
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to change password',
        variant: 'destructive',
      });
    } finally {
      setChangingPassword(false);
    }
  }

  const createValid = isCreateValid(createForm);
  const editValid = isEditValid(selectedUser, editForm, orgs);

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 pt-16 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Users</h1>
          <p className="text-muted-foreground">Manage users and their roles</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="relative">
            <input
              type="text"
              placeholder="Search users…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-64 rounded-md border border-input bg-background pl-9 pr-3 py-2 text-sm"
            />
            <svg className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <Button onClick={() => setCreateOpen(true)}>+ New User</Button>
        </div>
      </div>

      <UserTable
        loading={loading}
        users={users}
        search={search}
        orgs={orgs}
        currentRole={currentRole}
        onEdit={openEdit}
        onDelete={openDeleteConfirm}
        onChangePassword={openChangePassword}
      />

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>New User</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div className="space-y-1">
              <label className="text-sm font-medium">Username *</label>
              <Input
                placeholder="e.g. john.doe"
                value={createForm.username}
                onChange={(e) => setCreateForm((f) => ({ ...f, username: e.target.value }))}
              />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">Email (optional)</label>
              <Input
                placeholder="e.g. john@example.com"
                type="email"
                value={createForm.email}
                onChange={(e) => setCreateForm((f) => ({ ...f, email: e.target.value }))}
              />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">Password *</label>
              <Input
                placeholder="Minimum 1 character"
                type="password"
                value={createForm.password}
                onChange={(e) => setCreateForm((f) => ({ ...f, password: e.target.value }))}
              />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">Role</label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={createForm.role}
                onChange={(e) =>
                  setCreateForm((f) => ({ ...f, role: e.target.value as RoleOption }))
                }
              >
                {availableRoles.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">Organisation *</label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={createForm.org_id}
                onChange={(e) => setCreateForm((f) => ({ ...f, org_id: e.target.value }))}
              >
                <option value="">Select organisation</option>
                {orgs.map((o) => (
                  <option key={o.id} value={String(o.id)}>
                    {o.name}
                  </option>
                ))}
              </select>
              {orgs.length === 0 && (
                <p className="text-xs text-muted-foreground">No organisations available — create one first.</p>
              )}
            </div>
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

      <UserDialogs
        editOpen={editOpen}
        setEditOpen={setEditOpen}
        editing={editing}
        selectedUser={selectedUser}
        editForm={editForm}
        setEditForm={setEditForm}
        availableRoles={availableRoles}
        orgs={orgs}
        currentRole={currentRole}
        editValid={editValid}
        onEdit={handleEdit}
        passwordOpen={passwordOpen}
        setPasswordOpen={setPasswordOpen}
        changingPassword={changingPassword}
        userForPassword={userForPassword}
        passwordForm={passwordForm}
        setPasswordForm={setPasswordForm}
        onChangePassword={handleChangePassword}
        deleteOpen={deleteOpen}
        setDeleteOpen={setDeleteOpen}
        deleting={deleting}
        userToDelete={userToDelete}
        onDelete={handleDelete}
      />
    </div>
  );
}
