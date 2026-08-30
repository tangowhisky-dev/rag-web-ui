'use client';

import { useState, useEffect, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { api, ApiError } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from '@/components/ui/dialog';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { useToast } from '@/components/ui/use-toast';
import { LoadingDots } from '@/components/ui/loading-dots';
import { ArrowLeft, Trash2, Search } from 'lucide-react';

interface AbbreviationList {
  id: number;
  name: string;
  description: string | null;
  org_id: number | null;
  org_name: string | null;
  is_enabled: boolean;
  row_count: number;
  created_at: string;
  updated_at: string;
}

interface Abbreviation {
  id: number;
  list_id: number;
  abbreviation: string;
  expanded_form: string;
  category: string | null;
}

interface PaginatedResult {
  items: Abbreviation[];
  total: number;
  page: number;
  size: number;
}

export default function AbbreviationListDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { toast } = useToast();
  const listId = params?.listId as string;

  const [list, setList] = useState<AbbreviationList | null>(null);
  const [items, setItems] = useState<Abbreviation[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [editOpen, setEditOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);

  // Edit form
  const [editName, setEditName] = useState('');
  const [editDesc, setEditDesc] = useState('');
  const [editEnabled, setEditEnabled] = useState(true);
  const [saving, setSaving] = useState(false);

  const pageSize = 50;

  const loadList = useCallback(async () => {
    try {
      const data = await api.get(`/api/admin/abbreviation-lists/${listId}`);
      setList(data);
      setEditName(data.name);
      setEditDesc(data.description || '');
      setEditEnabled(data.is_enabled);
    } catch (err) {
      toast({ title: 'Error', description: 'Failed to load list', variant: 'destructive' });
    }
  }, [listId, toast]);

  const loadItems = useCallback(async () => {
    try {
      const params = new URLSearchParams({
        page: String(page),
        size: String(pageSize),
      });
      if (search) params.set('search', search);
      const data: PaginatedResult = await api.get(`/api/admin/abbreviation-lists/${listId}/abbreviations?${params}`);
      setItems(data.items);
      setTotal(data.total);
    } catch (err) {
      toast({ title: 'Error', description: 'Failed to load abbreviations', variant: 'destructive' });
    }
  }, [listId, page, search, toast]);

  useEffect(() => {
    (async () => {
      setLoading(true);
      await Promise.all([loadList(), loadItems()]);
      setLoading(false);
    })();
  }, [loadList, loadItems]);

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (page !== 1) {
        setPage(1);
      } else {
        loadItems();
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [search, page, loadItems]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!cancelled) await loadItems();
    })();
    return () => { cancelled = true; };
  }, [page, loadItems]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.put(`/api/admin/abbreviation-lists/${listId}`, {
        name: editName.trim(),
        description: editDesc.trim() || null,
        is_enabled: editEnabled,
      });
      toast({ title: 'Saved', description: 'List updated successfully' });
      setEditOpen(false);
      loadList();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to save';
      toast({ title: 'Error', description: msg, variant: 'destructive' });
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    try {
      await api.delete(`/api/admin/abbreviation-lists/${listId}`);
      toast({ title: 'Deleted', description: 'List deleted successfully' });
      router.push('/dashboard/admin/abbreviations');
    } catch (err) {
      toast({ title: 'Error', description: 'Failed to delete list', variant: 'destructive' });
    } finally {
      setDeleteOpen(false);
    }
  };

  const totalPages = Math.ceil(total / pageSize);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <LoadingDots />
      </div>
    );
  }

  if (!list) {
    return (
      <div className="p-6">
        <p className="text-muted-foreground">List not found.</p>
        <Link href="/dashboard/admin/abbreviations" className="text-sm text-blue-600 hover:underline mt-2 inline-block">
          Back to lists
        </Link>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <Link href="/dashboard/admin/abbreviations" className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1">
            <ArrowLeft className="w-3 h-3" />
            Back to lists
          </Link>
          <h1 className="text-2xl font-semibold">{list.name}</h1>
          {list.description && <p className="text-sm text-muted-foreground">{list.description}</p>}
          <div className="flex items-center gap-2 pt-1">
            {list.org_id === null ? (
              <Badge>Universal</Badge>
            ) : (
              <Badge variant="outline">{list.org_name || `Org ${list.org_id}`}</Badge>
            )}
            <Badge variant={list.is_enabled ? 'default' : 'secondary'}>
              {list.is_enabled ? 'Enabled' : 'Disabled'}
            </Badge>
            <span className="text-sm text-muted-foreground tabular-nums">
              {total.toLocaleString()} abbreviations
            </span>
          </div>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => setEditOpen(true)}>
            Edit
          </Button>
          <Button variant="outline" onClick={() => setDeleteOpen(true)} className="text-destructive hover:text-destructive">
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Search */}
      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <Input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search abbreviation or expanded form..."
          className="pl-9"
        />
      </div>

      {/* Abbreviations Table */}
      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-32">Abbreviation</TableHead>
              <TableHead>Expanded Form</TableHead>
              <TableHead className="w-48">Category</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.length === 0 ? (
              <TableRow>
                <TableCell colSpan={3} className="text-center text-muted-foreground py-8">
                  {search ? 'No matches found' : 'No abbreviations in this list'}
                </TableCell>
              </TableRow>
            ) : (
              items.map((abbr) => (
                <TableRow key={abbr.id}>
                  <TableCell className="font-mono font-medium">{abbr.abbreviation}</TableCell>
                  <TableCell>{abbr.expanded_form}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">{abbr.category || '—'}</TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, total)} of {total.toLocaleString()}
          </p>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage(p => p - 1)}
            >
              Previous
            </Button>
            <span className="text-sm py-1.5 tabular-nums">
              {page} / {totalPages}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= totalPages}
              onClick={() => setPage(p => p + 1)}
            >
              Next
            </Button>
          </div>
        </div>
      )}

      {/* Edit Dialog */}
      <Dialog open={editOpen} onOpenChange={setEditOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit List</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Name</Label>
              <Input value={editName} onChange={(e) => setEditName(e.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Description</Label>
              <Input value={editDesc} onChange={(e) => setEditDesc(e.target.value)} />
            </div>
            <div className="flex items-center justify-between">
              <Label>Enabled</Label>
              <Switch checked={editEnabled} onCheckedChange={setEditEnabled} />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditOpen(false)} disabled={saving}>Cancel</Button>
            <Button onClick={handleSave} disabled={saving}>
              {saving ? 'Saving...' : 'Save'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <ConfirmDialog
        open={deleteOpen}
        title="Delete Abbreviation List"
        description={`Are you sure you want to delete "${list.name}"? This will remove all ${total.toLocaleString()} abbreviation rows. This cannot be undone.`}
        confirmText="Delete"
        destructive
        onConfirm={handleDelete}
        onCancel={() => setDeleteOpen(false)}
      />
    </div>
  );
}
