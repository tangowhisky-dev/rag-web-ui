'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { api, ApiError } from '@/lib/api';
import { fetchTokenClaims } from '@/lib/auth';
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
import { Upload, Trash2 } from 'lucide-react';

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

export default function AbbreviationsPage() {
  const router = useRouter();
  const { toast } = useToast();
  const [lists, setLists] = useState<AbbreviationList[]>([]);
  const [loading, setLoading] = useState(true);
  const [userRole, setUserRole] = useState<string | undefined>();
  const [uploadOpen, setUploadOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<AbbreviationList | null>(null);
  const [togglingId, setTogglingId] = useState<number | null>(null);

  // Upload form state
  const [fileName, setFileName] = useState('');
  const [listName, setListName] = useState('');
  const [listDesc, setListDesc] = useState('');
  const [scope, setScope] = useState<'universal' | 'org'>('org');
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadLists = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.get('/api/admin/abbreviation-lists');
      setLists(data);
    } catch (err) {
      toast({ title: 'Error', description: 'Failed to load abbreviation lists', variant: 'destructive' });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const claims = await fetchTokenClaims();
      if (cancelled) return;
      setUserRole(claims?.role);
      await loadLists();
    })();
    return () => { cancelled = true; };
  }, [loadLists]);

  const handleUpload = async () => {
    if (!fileInputRef.current?.files?.[0]) {
      toast({ title: 'Error', description: 'Please select a CSV file', variant: 'destructive' });
      return;
    }
    if (!listName.trim()) {
      toast({ title: 'Error', description: 'Please enter a list name', variant: 'destructive' });
      return;
    }
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', fileInputRef.current.files[0]);
      const params = new URLSearchParams({
        name: listName.trim(),
        description: listDesc.trim(),
        scope,
      });
      const resp = await api.post(`/api/admin/abbreviation-lists/upload?${params}`, formData);
      toast({ title: 'Uploaded', description: `${resp.name}: ${resp.row_count} rows` });
      setUploadOpen(false);
      setListName('');
      setListDesc('');
      setFileName('');
      setScope('org');
      loadLists();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Upload failed';
      toast({ title: 'Error', description: msg, variant: 'destructive' });
    } finally {
      setUploading(false);
    }
  };

  const handleToggle = async (lst: AbbreviationList) => {
    setTogglingId(lst.id);
    try {
      await api.put(`/api/admin/abbreviation-lists/${lst.id}`, { is_enabled: !lst.is_enabled });
      setLists(prev => prev.map(l => l.id === lst.id ? { ...l, is_enabled: !l.is_enabled } : l));
      toast({ title: lst.is_enabled ? 'Disabled' : 'Enabled', description: lst.name });
    } catch (err) {
      toast({ title: 'Error', description: 'Failed to toggle list', variant: 'destructive' });
    } finally {
      setTogglingId(null);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await api.delete(`/api/admin/abbreviation-lists/${deleteTarget.id}`);
      toast({ title: 'Deleted', description: deleteTarget.name });
      setLists(prev => prev.filter(l => l.id !== deleteTarget.id));
    } catch (err) {
      toast({ title: 'Error', description: 'Failed to delete list', variant: 'destructive' });
    } finally {
      setDeleteTarget(null);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    setFileName(file ? file.name : '');
    if (file && !listName) {
      setListName(file.name.replace(/\.csv$/i, ''));
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <LoadingDots />
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Abbreviation Lists</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Manage abbreviation CSV files for expansion during ingestion and retrieval
          </p>
        </div>
        <Button onClick={() => setUploadOpen(true)}>
          <Upload className="w-4 h-4 mr-2" />
          Upload CSV
        </Button>
      </div>

      {lists.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          No abbreviation lists uploaded yet. Click &quot;Upload CSV&quot; to add one.
        </div>
      ) : (
        <div className="border rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Scope</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Rows</TableHead>
                <TableHead>Created</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {lists.map((lst) => (
                <TableRow
                  key={lst.id}
                  className="cursor-pointer hover:bg-muted/50"
                  onClick={() => router.push(`/dashboard/admin/abbreviations/${lst.id}`)}
                >
                  <TableCell className="font-medium">
                    {lst.name}
                    {lst.description && (
                      <div className="text-xs text-muted-foreground mt-0.5">{lst.description}</div>
                    )}
                  </TableCell>
                  <TableCell>
                    {lst.org_id === null ? (
                      <Badge>Universal</Badge>
                    ) : (
                      <Badge variant="outline">{lst.org_name || `Org ${lst.org_id}`}</Badge>
                    )}
                  </TableCell>
                  <TableCell>
                    <Switch
                      checked={lst.is_enabled}
                      onCheckedChange={() => handleToggle(lst)}
                      disabled={togglingId === lst.id}
                      onClick={(e) => e.stopPropagation()}
                    />
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{lst.row_count.toLocaleString()}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {new Date(lst.created_at).toLocaleDateString()}
                  </TableCell>
                  <TableCell className="text-right" onClick={(e) => e.stopPropagation()}>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setDeleteTarget(lst)}
                      className="text-destructive hover:text-destructive"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {/* Upload Dialog */}
      <Dialog open={uploadOpen} onOpenChange={setUploadOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Upload Abbreviation CSV</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>CSV File</Label>
              <Input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileChange}
              />
              {fileName && <p className="text-xs text-muted-foreground">Selected: {fileName}</p>}
            </div>
            <div className="space-y-2">
              <Label>List Name</Label>
              <Input
                value={listName}
                onChange={(e) => setListName(e.target.value)}
                placeholder="e.g. Military Abbreviations"
              />
            </div>
            <div className="space-y-2">
              <Label>Description (optional)</Label>
              <Input
                value={listDesc}
                onChange={(e) => setListDesc(e.target.value)}
                placeholder="Brief description of this list"
              />
            </div>
            {userRole === 'super_admin' && (
              <div className="space-y-2">
                <Label>Scope</Label>
                <select
                  className="w-full border rounded-md p-2 bg-background"
                  value={scope}
                  onChange={(e) => setScope(e.target.value as 'universal' | 'org')}
                >
                  <option value="org">My Organisation</option>
                  <option value="universal">Universal (all organisations)</option>
                </select>
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              CSV format: abbreviation, expanded_form, category
            </p>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setUploadOpen(false)} disabled={uploading}>
              Cancel
            </Button>
            <Button onClick={handleUpload} disabled={uploading}>
              {uploading ? 'Uploading...' : 'Upload'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete Abbreviation List"
        description={`Are you sure you want to delete "${deleteTarget?.name}"? This will remove all ${deleteTarget?.row_count.toLocaleString()} abbreviation rows. This cannot be undone.`}
        confirmText="Delete"
        destructive
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
