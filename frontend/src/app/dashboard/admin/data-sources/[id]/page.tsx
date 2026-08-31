'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { api, ApiError } from '@/lib/api';

import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { useToast } from '@/components/ui/use-toast';
import { LoadingDots } from '@/components/ui/loading-dots';
import {
  ArrowLeft,
  Folder,
  FileText,
  FileCode,
  FileSpreadsheet,
  FileImage,
  File as FileIcon,
  ChevronRight,
  Loader2,
  AlertTriangle,
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BrowseItem {
  type: 'folder' | 'file';
  name: string;
  path: string;
  absolute_path?: string;
  size?: number;
  content_type?: string;
  modified_at?: string | null;
  document_id?: number | null;
  is_selected?: boolean;
  status?: string;
  chunk_count?: number;
  graph_status?: string | null;
  conversion_status?: string | null;
  title?: string | null;
  error_message?: string | null;
  file_count?: number;
  ingested_count?: number;
  selected_count?: number;
}

interface BrowseStats {
  total_documents: number;
  selected: number;
  unselected: number;
  ingested: number;
  completed: number;
  failed: number;
  processing: number;
  pending: number;
}

interface BrowseResponse {
  datastore_id: number;
  datastore_name: string;
  folder_path: string;
  current_path: string;
  breadcrumbs: { name: string; path: string }[];
  items: BrowseItem[];
  total: number;
  total_files: number;
  total_folders: number;
  page: number;
  page_size: number;
  stats: BrowseStats;
  error?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getFileIcon(name: string) {
  const ext = name.split('.').pop()?.toLowerCase() || '';
  if (['pdf'].includes(ext)) return <FileText className="h-4 w-4 text-red-500" />;
  if (['docx', 'doc', 'odt', 'rtf', 'txt', 'md'].includes(ext)) return <FileText className="h-4 w-4 text-blue-500" />;
  if (['xlsx', 'xls', 'csv', 'ods'].includes(ext)) return <FileSpreadsheet className="h-4 w-4 text-green-600" />;
  if (['pptx', 'ppt', 'odp'].includes(ext)) return <FileText className="h-4 w-4 text-orange-500" />;
  if (['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'].includes(ext)) return <FileImage className="h-4 w-4 text-purple-500" />;
  if (['html', 'htm', 'xml', 'json'].includes(ext)) return <FileCode className="h-4 w-4 text-cyan-600" />;
  return <FileIcon className="h-4 w-4 text-muted-foreground" />;
}

function formatSize(bytes: number): string {
  if (bytes === 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let size = bytes;
  while (size >= 1024 && i < units.length - 1) {
    size /= 1024;
    i++;
  }
  return `${size.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

type VisualState = 'excluded' | 'ingested' | 'pending' | 'dirty-unselect' | 'dirty-select';

const VISUAL_STATE_TABLE: Record<string, VisualState> = {
  'true-true-true': 'ingested',
  'true-true-false': 'pending',
  'true-false-true': 'dirty-unselect',
  'true-false-false': 'excluded',
  'false-true-true': 'dirty-select',
  'false-true-false': 'dirty-select',
  'false-false-true': 'excluded',
  'false-false-false': 'excluded',
};

function getVisualState(item: BrowseItem, dirtyMap: Map<string, boolean>): VisualState {
  if (item.type === 'folder') return 'excluded';
  const original = item.is_selected ?? false;
  const current = dirtyMap.get(item.path) ?? original;
  const hasChunks = (item.chunk_count ?? 0) > 0;
  return VISUAL_STATE_TABLE[`${original}-${current}-${hasChunks}`];
}

function statusBadge(status: string | undefined) {
  if (!status || status === 'not_ingested') return <span className="text-xs text-muted-foreground">—</span>;
  if (status === 'completed') return <Badge variant="secondary" className="text-xs">✓ Done</Badge>;
  if (status === 'failed') return <Badge variant="destructive" className="text-xs">✗ Failed</Badge>;
  if (status === 'processing') return <Badge variant="default" className="text-xs"><Loader2 className="h-3 w-3 animate-spin mr-1" />Processing</Badge>;
  if (status === 'pending') return <Badge variant="outline" className="text-xs">Pending</Badge>;
  return <span className="text-xs text-muted-foreground">{status}</span>;
}

function getStateFromServerCounts(fileCount: number, selectedCount: number): boolean | 'indeterminate' {
  if (fileCount === 0) return false;
  if (selectedCount === 0) return false;
  if (selectedCount >= fileCount) return true;
  return 'indeterminate';
}

function getStateFromDirtyEntries(
  dirtyMap: Map<string, boolean>,
  folderPath: string,
  fileCount: number,
  selectedCount: number,
): boolean | 'indeterminate' {
  const folderPrefix = folderPath + '/';
  let dirtyCount = 0;
  let checkedCount = 0;
  for (const [p, v] of dirtyMap) {
    if (p.startsWith(folderPrefix) || p === folderPath) {
      dirtyCount++;
      if (v) checkedCount++;
    }
  }
  if (dirtyCount === 0) return getStateFromServerCounts(fileCount, selectedCount);
  if (checkedCount === 0) return false;
  if (checkedCount === dirtyCount) return true;
  return 'indeterminate';
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function FolderRow({
  item,
  folderState,
  folderLoading: loading,
  onToggle,
  onNavigate,
}: {
  item: BrowseItem;
  folderState: boolean | 'indeterminate';
  folderLoading: string | null;
  onToggle: (item: BrowseItem) => void;
  onNavigate: (path: string) => void;
}) {
  const selectedCount = item.selected_count ?? 0;
  const fileCount = item.file_count ?? 0;
  return (
    <TableRow
      className="cursor-pointer hover:bg-muted/50"
      onClick={() => onNavigate(item.path)}
    >
      <TableCell onClick={(e) => e.stopPropagation()}>
        <Checkbox
          checked={folderState}
          disabled={loading === item.path}
          onCheckedChange={() => onToggle(item)}
        />
      </TableCell>
      <TableCell>
        <div className="flex items-center gap-2">
          <Folder className="h-4 w-4 text-muted-foreground" />
          <span className="font-medium">{item.name}</span>
          <span className="text-xs text-muted-foreground">
            {item.file_count} files · {item.ingested_count} ingested
            {selectedCount > 0 && selectedCount < fileCount && (
              <span className="ml-1 text-amber-600 dark:text-amber-400">· {item.selected_count} selected</span>
            )}
          </span>
          {loading === item.path && (
            <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
          )}
        </div>
      </TableCell>
      <TableCell />
      <TableCell />
      <TableCell />
      <TableCell />
    </TableRow>
  );
}

function FileRow({
  item,
  visualState: vs,
  checkState,
  onToggle,
  onEdit,
}: {
  item: BrowseItem;
  visualState: VisualState;
  checkState: boolean | 'indeterminate';
  onToggle: (item: BrowseItem) => void;
  onEdit: (documentId: number) => void;
}) {
  return (
    <TableRow
      className={vs === 'dirty-unselect' ? 'bg-red-50 dark:bg-red-950/20' : ''}
      onClick={() => onToggle(item)}
    >
      <TableCell onClick={(e) => e.stopPropagation()}>
        <Checkbox
          checked={checkState}
          onCheckedChange={() => onToggle(item)}
        />
      </TableCell>
      <TableCell>
        <div className="flex items-center gap-2">
          {getFileIcon(item.name)}
          <div className="min-w-0">
            <div className={
              vs === 'excluded' ? 'text-muted-foreground italic text-sm truncate' :
              vs === 'dirty-unselect' ? 'text-red-600 dark:text-red-400 text-sm truncate' :
              vs === 'dirty-select' ? 'text-blue-600 dark:text-blue-400 font-medium text-sm truncate' :
              vs === 'pending' ? 'font-medium text-sm truncate' :
              'text-sm truncate'
            }>
              {item.title || item.name}
            </div>
            {item.title && item.title !== item.name && (
              <div className="text-xs text-muted-foreground truncate">{item.name}</div>
            )}
          </div>
        </div>
      </TableCell>
      <TableCell className="text-sm text-muted-foreground tabular-nums">
        {formatSize(item.size || 0)}
      </TableCell>
      <TableCell>{statusBadge(item.status)}</TableCell>
      <TableCell className="text-sm text-muted-foreground tabular-nums">
        {item.chunk_count || 0}
      </TableCell>
      <TableCell className="text-xs text-muted-foreground">
        {item.modified_at ? new Date(item.modified_at).toLocaleDateString() : '—'}
      </TableCell>
      <TableCell onClick={(e) => e.stopPropagation()}>
        {item.document_id && item.conversion_status === 'completed' && (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 px-2 text-xs"
            onClick={() => onEdit(item.document_id!)}
          >
            Edit
          </Button>
        )}
      </TableCell>
    </TableRow>
  );
}

function DirtyStateBar({
  toSelect,
  toUnselect,
  saving,
  onDiscard,
  onSave,
}: {
  toSelect: string[];
  toUnselect: string[];
  saving: boolean;
  onDiscard: () => void;
  onSave: () => void;
}) {
  return (
    <div className="fixed bottom-0 left-0 right-0 border-t bg-card shadow-lg z-50">
      <div className="container mx-auto max-w-7xl flex items-center justify-between p-4">
        <div className="flex items-center gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-500" />
          <div className="text-sm">
            <span className="font-medium">{toSelect.length + toUnselect.length} unsaved change(s)</span>
            {toUnselect.length > 0 && (
              <span className="text-red-600 dark:text-red-400 ml-2">
                {toUnselect.length} item(s) will be DELETED from Qdrant/MySQL/Neo4j
              </span>
            )}
            {toSelect.length > 0 && (
              <span className="text-blue-600 dark:text-blue-400 ml-2">
                {toSelect.length} item(s) will be marked for ingestion
              </span>
            )}
          </div>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={onDiscard}>
            Discard
          </Button>
          <Button size="sm" disabled={saving} onClick={onSave}>
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Save Changes'}
          </Button>
        </div>
      </div>
    </div>
  );
}

function Pagination({
  page,
  pageSize,
  totalFiles,
  onPrev,
  onNext,
}: {
  page: number;
  pageSize: number;
  totalFiles: number;
  onPrev: () => void;
  onNext: () => void;
}) {
  return (
    <div className="flex items-center justify-between p-3 border-t text-sm">
      <span className="text-muted-foreground">
        {page * pageSize + 1}–{Math.min((page + 1) * pageSize, totalFiles)} of {totalFiles}
      </span>
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={page === 0}
          onClick={onPrev}
        >
          Previous
        </Button>
        <span className="flex items-center px-2 text-muted-foreground">
          {page + 1} / {Math.ceil(totalFiles / pageSize)}
        </span>
        <Button
          variant="outline"
          size="sm"
          disabled={(page + 1) * pageSize >= totalFiles}
          onClick={onNext}
        >
          Next
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function DatastoreBrowsePage() {
  const params = useParams();
  const router = useRouter();
  const { toast } = useToast();
  const datastoreId = (params?.id ?? '') as string;

  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<BrowseResponse | null>(null);
  const [currentPath, setCurrentPath] = useState('');
  const [sort, setSort] = useState('name');
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);
  const [saving, setSaving] = useState(false);
  const [folderLoading, setFolderLoading] = useState<string | null>(null);

  // Dirty state: path -> selected (true/false)
  const [dirtyMap, setDirtyMap] = useState<Map<string, boolean>>(new Map());
  // Ref mirror of dirtyMap so async functions always read the latest value
  const dirtyMapRef = useRef(dirtyMap);
  // Original states from server (for computing dirty diff)
  const [originalMap, setOriginalMap] = useState<Map<string, boolean>>(new Map());
  const originalMapRef = useRef(originalMap);
  // Folder-level dirty state: folder path -> selected (true/false)
  const [dirtyFolders, setDirtyFolders] = useState<Map<string, boolean>>(new Map());
  const dirtyFoldersRef = useRef(dirtyFolders);
  // Track which folder paths have been expanded into dirtyMap
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const expandedFoldersRef = useRef(expandedFolders);
  // Map relative path -> absolute path for all files seen (browse + folder expansion)
  // Used by resolvePath to send absolute paths to save-selection.
  const [pathToAbsolute, setPathToAbsolute] = useState<Map<string, string>>(new Map());
  const pathToAbsoluteRef = useRef(pathToAbsolute);

  // Sync refs whenever state changes
  useEffect(() => { dirtyMapRef.current = dirtyMap; }, [dirtyMap]);
  useEffect(() => { originalMapRef.current = originalMap; }, [originalMap]);
  useEffect(() => { dirtyFoldersRef.current = dirtyFolders; }, [dirtyFolders]);
  useEffect(() => { expandedFoldersRef.current = expandedFolders; }, [expandedFolders]);
  useEffect(() => { pathToAbsoluteRef.current = pathToAbsolute; }, [pathToAbsolute]);

  // Confirm dialog
  const [confirmOpen, setConfirmOpen] = useState(false);

  const pageSize = 100;

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        path: currentPath,
        sort,
        page: String(page),
        page_size: String(pageSize),
        search,
      });
      const resp = await api.get(`/api/admin/datastores/${datastoreId}/browse?${params.toString()}`) as BrowseResponse;
      setData(resp);

      // Update original map with server state
      const newOriginal = new Map(originalMap);
      const newPathMap = new Map(pathToAbsoluteRef.current);
      for (const item of resp.items) {
        if (item.type === 'file') {
          newOriginal.set(item.path, item.is_selected ?? false);
          if (item.absolute_path) {
            newPathMap.set(item.path, item.absolute_path);
          }
        }
      }
      setOriginalMap(newOriginal);
      setPathToAbsolute(newPathMap);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : 'Failed to load datastore contents';
      toast({ title: 'Error', description: msg, variant: 'destructive' });
    } finally {
      setLoading(false);
    }
  }, [datastoreId, currentPath, sort, page, search]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const debounce = setTimeout(fetchData, search ? 300 : 0);
    return () => clearTimeout(debounce);
  }, [fetchData, search]);

  // Compute dirty changes — file-level only.
  //
  // dirtyFolders is NOT used for counting — it's only used by the save
  // handler to collapse individual file paths back into folder paths
  // for efficiency.  This means:
  //   - Unselecting a 15-file folder shows "15 items" (the real file
  //     count), not "1 item" (the folder path).
  //   - Toggling a folder off then back on shows 0 changes (each file
  //     is back to its original state), not "1 item".
  const dirtyChanges = useMemo(() => {
    const toSelect: string[] = [];
    const toUnselect: string[] = [];

    for (const [path, selected] of dirtyMap) {
      const original = originalMap.get(path);
      if (original === undefined) continue;
      if (selected !== original) {
        if (selected) toSelect.push(path);
        else toUnselect.push(path);
      }
    }

    return { toSelect, toUnselect };
  }, [dirtyMap, originalMap]);

  const hasChanges = dirtyChanges.toSelect.length > 0 || dirtyChanges.toUnselect.length > 0;

  // Get folder checkbox state: checked / unchecked / indeterminate
  const getFolderCheckState = useCallback((item: BrowseItem): boolean | 'indeterminate' => {
    if (dirtyFolders.has(item.path)) {
      return dirtyFolders.get(item.path) ?? false;
    }
    if (expandedFolders.has(item.path)) {
      return getStateFromDirtyEntries(dirtyMap, item.path, item.file_count ?? 0, item.selected_count ?? 0);
    }
    return getStateFromServerCounts(item.file_count ?? 0, item.selected_count ?? 0);
  }, [dirtyFolders, expandedFolders, dirtyMap]);

  // Get current checkbox state for a file
  const getCheckState = (item: BrowseItem): boolean | 'indeterminate' => {
    if (item.type !== 'file') return false;
    const original = item.is_selected ?? false;
    return dirtyMap.get(item.path) ?? original;
  };

  // Toggle a file's selection
  const toggleFile = useCallback((item: BrowseItem) => {
    if (item.type !== 'file' || !item.absolute_path) return;
    const original = item.is_selected ?? false;
    const current = dirtyMap.get(item.path) ?? original;
    const newMap = new Map(dirtyMap);
    newMap.set(item.path, !current);
    setDirtyMap(newMap);
  }, [dirtyMap, setDirtyMap]);

  // Core folder toggle logic — sets folder to a specific target state
  const toggleFolderTo = useCallback(async (item: BrowseItem, targetChecked: boolean) => {
    if (item.type !== 'folder') return;

    setFolderLoading(item.path);
    try {
      const resp = await api.get(
        `/api/admin/datastores/${datastoreId}/folder-files?path=${encodeURIComponent(item.path)}`
      ) as { files: { path: string; absolute_path: string; is_selected: boolean }[] };

      const newMap = new Map(dirtyMapRef.current);
      const newOriginal = new Map(originalMapRef.current);
      const newPathMap = new Map(pathToAbsoluteRef.current);
      for (const f of resp.files) {
        newMap.set(f.path, targetChecked);
        newPathMap.set(f.path, f.absolute_path);
        if (!newOriginal.has(f.path)) {
          newOriginal.set(f.path, f.is_selected);
        }
      }
      setDirtyMap(newMap);
      setOriginalMap(newOriginal);
      setPathToAbsolute(newPathMap);
      setExpandedFolders(prev => new Set(prev).add(item.path));

      // Track folder-level intent so save sends the folder path
      const newFolders = new Map(dirtyFoldersRef.current);
      newFolders.set(item.path, targetChecked);
      setDirtyFolders(newFolders);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : 'Failed to list folder files';
      toast({ title: 'Error', description: msg, variant: 'destructive' });
    } finally {
      setFolderLoading(null);
    }
  }, [datastoreId, toast, setFolderLoading, setDirtyMap, setOriginalMap, setPathToAbsolute, setExpandedFolders, setDirtyFolders]);

  // Toggle a folder's selection — fetches all files under the folder
  // and adds them to dirtyMap so navigating into the folder shows state.
  const toggleFolder = useCallback(async (item: BrowseItem) => {
    if (item.type !== 'folder') return;
    const currentState = getFolderCheckState(item);
    const targetChecked = currentState !== true;
    await toggleFolderTo(item, targetChecked);
  }, [getFolderCheckState, toggleFolderTo]);

  // Toggle all items on current page (files + folders)
  const toggleAllFiles = useCallback(async (checked: boolean) => {
    if (!data) return;
    // Start from the latest dirtyMap (via ref, not stale closure)
    const newMap = new Map(dirtyMapRef.current);
    const newOriginal = new Map(originalMapRef.current);
    const newFolders = new Map(dirtyFoldersRef.current);
    const newExpanded = new Set(expandedFoldersRef.current);

    // Toggle files on current page immediately
    const newPathMap = new Map(pathToAbsoluteRef.current);
    for (const item of data.items) {
      if (item.type === 'file' && item.absolute_path) {
        newMap.set(item.path, checked);
        newPathMap.set(item.path, item.absolute_path);
      }
    }

    // Toggle folders — fetch files for each and accumulate into newMap
    const folders = data.items.filter(i => i.type === 'folder');
    for (const folder of folders) {
      try {
        const resp = await api.get(
          `/api/admin/datastores/${datastoreId}/folder-files?path=${encodeURIComponent(folder.path)}`
        ) as { files: { path: string; absolute_path: string; is_selected: boolean }[] };

        for (const f of resp.files) {
          newMap.set(f.path, checked);
          newPathMap.set(f.path, f.absolute_path);
          if (!newOriginal.has(f.path)) {
            newOriginal.set(f.path, f.is_selected);
          }
        }
        newExpanded.add(folder.path);
        newFolders.set(folder.path, checked);
      } catch (e) {
        // Skip folder on error — don't abort the whole operation
      }
    }

    setDirtyMap(newMap);
    setOriginalMap(newOriginal);
    setDirtyFolders(newFolders);
    setExpandedFolders(newExpanded);
    setPathToAbsolute(newPathMap);
  }, [data, datastoreId, setDirtyMap, setOriginalMap, setDirtyFolders, setExpandedFolders, setPathToAbsolute]);

  // All-items-on-page checkbox state (files + folders)
  const pageCheckboxState = (): boolean | 'indeterminate' => {
    if (!data) return false;
    const items = data.items;
    if (items.length === 0) return false;
    let checkedCount = 0;
    for (const item of items) {
      if (item.type === 'file') {
        if (getCheckState(item) === true) checkedCount++;
      } else {
        if (getFolderCheckState(item) === true) checkedCount++;
      }
    }
    if (checkedCount === 0) return false;
    if (checkedCount === items.length) return true;
    return 'indeterminate';
  };

  // Navigate into a folder
  const navigateTo = useCallback((path: string) => {
    setCurrentPath(path);
    setPage(0);
    setSearch('');
  }, [setCurrentPath, setPage, setSearch]);

  // Save changes
  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      // Resolve a file path to its absolute path for the backend.
      // We send individual file paths (not collapsed folder paths)
      // because the backend's expand_folder_paths expects absolute
      // paths and we only have relative folder paths in dirtyFolders.
      // The file count is typically small enough that sending individual
      // paths is fine, and it avoids the path resolution mismatch.
      const resolvePath = (p: string): string => {
        const abs = pathToAbsoluteRef.current.get(p);
        if (abs) return abs;
        const item = data?.items.find(i => i.path === p);
        if (item?.absolute_path) return item.absolute_path;
        return p;
      };

      const body = {
        select: dirtyChanges.toSelect.map(resolvePath),
        unselect: dirtyChanges.toUnselect.map(resolvePath),
      };
      await api.post(`/api/admin/datastores/${datastoreId}/save-selection`, body);
      toast({
        title: 'Changes saved',
        description: `${dirtyChanges.toUnselect.length} item(s) unselected, ${dirtyChanges.toSelect.length} item(s) selected.`,
      });
      setDirtyMap(new Map());
      setDirtyFolders(new Map());
      setExpandedFolders(new Set());
      setConfirmOpen(false);
      await fetchData();
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : 'Failed to save changes';
      toast({ title: 'Error', description: msg, variant: 'destructive' });
    } finally {
      setSaving(false);
    }
  }, [data, dirtyChanges, fetchData, datastoreId, toast, setSaving, setDirtyMap, setDirtyFolders, setExpandedFolders, setConfirmOpen]);

  // Discard changes
  const discardChanges = useCallback(() => {
    setDirtyMap(new Map());
    setDirtyFolders(new Map());
    setExpandedFolders(new Set());
  }, [setDirtyMap, setDirtyFolders, setExpandedFolders]);

  // Render
  return (
    <div className="container mx-auto p-6 max-w-7xl">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <Button variant="ghost" size="sm" onClick={() => router.push('/dashboard/admin/data-sources')}>
          <ArrowLeft className="h-4 w-4 mr-1" />
          Data Sources
        </Button>
      </div>

      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">
            {data?.datastore_name || 'Datastore'}
          </h1>
          {data && (
            <p className="text-sm text-muted-foreground mt-1">
              {data.folder_path} · {data.stats.total_documents} documents · {data.stats.ingested} ingested
            </p>
          )}
        </div>
      </div>

      {/* Stats bar */}
      {data && (
        <div className="flex flex-wrap gap-3 mb-4 text-sm">
          <span className="text-muted-foreground">
            <span className="font-medium text-foreground">{data.stats.ingested}</span> ingested
          </span>
          <span className="text-muted-foreground">·</span>
          <span className="text-muted-foreground">
            <span className="font-medium text-foreground">{data.stats.failed}</span> failed
          </span>
          <span className="text-muted-foreground">·</span>
          <span className="text-muted-foreground">
            <span className="font-medium text-foreground">{data.stats.pending}</span> pending
          </span>
          <span className="text-muted-foreground">·</span>
          <span className="text-muted-foreground">
            <span className="font-medium text-foreground">{data.stats.unselected}</span> excluded
          </span>
        </div>
      )}

      {/* Browser */}
      <div className="border rounded-lg bg-card">
        {/* Toolbar */}
        <div className="flex items-center gap-3 p-3 border-b">
          {/* Breadcrumbs */}
          <div className="flex items-center gap-1 text-sm flex-1 min-w-0 overflow-x-auto">
            {data?.breadcrumbs.map((bc, i) => (
              <div key={i} className="flex items-center gap-1 shrink-0">
                {i > 0 && <ChevronRight className="h-3 w-3 text-muted-foreground" />}
                <button
                  className="hover:underline text-muted-foreground hover:text-foreground"
                  onClick={() => navigateTo(bc.path)}
                >
                  {bc.name}
                </button>
              </div>
            ))}
          </div>

          {/* Search */}
          <Input
            placeholder="Search files..."
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(0); }}
            className="w-48 h-8"
          />

          {/* Sort */}
          <Select value={sort} onValueChange={setSort}>
            <SelectTrigger className="w-32 h-8">
              <SelectValue placeholder="Sort" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="name">Name</SelectItem>
              <SelectItem value="-name">Name (desc)</SelectItem>
              <SelectItem value="size">Size</SelectItem>
              <SelectItem value="-size">Size (desc)</SelectItem>
              <SelectItem value="modified">Modified</SelectItem>
              <SelectItem value="-modified">Modified (desc)</SelectItem>
              <SelectItem value="status">Status</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Table */}
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <LoadingDots />
          </div>
        ) : data && data.items.length > 0 ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-10">
                  <Checkbox
                    checked={pageCheckboxState()}
                    onCheckedChange={(v) => toggleAllFiles(v === true)}
                  />
                </TableHead>
                <TableHead>Name</TableHead>
                <TableHead className="w-24">Size</TableHead>
                <TableHead className="w-32">Status</TableHead>
                <TableHead className="w-20">Chunks</TableHead>
                <TableHead className="w-32">Modified</TableHead>
                <TableHead className="w-16"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.items.map((item) => {
                if (item.type === 'folder') {
                  return (
                    <FolderRow
                      key={`folder-${item.path}`}
                      item={item}
                      folderState={getFolderCheckState(item)}
                      folderLoading={folderLoading}
                      onToggle={toggleFolder}
                      onNavigate={navigateTo}
                    />
                  );
                }
                return (
                  <FileRow
                    key={`file-${item.path}`}
                    item={item}
                    visualState={getVisualState(item, dirtyMap)}
                    checkState={getCheckState(item)}
                    onToggle={toggleFile}
                    onEdit={(id) => router.push(`/dashboard/admin/data-sources/${datastoreId}/documents/${id}`)}
                  />
                );
              })}
            </TableBody>
          </Table>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <Folder className="h-8 w-8 mb-2 opacity-50" />
            <p className="text-sm">No files in this folder</p>
          </div>
        )}

        {/* Pagination */}
        {data && data.total_files > pageSize && (
          <Pagination
            page={page}
            pageSize={pageSize}
            totalFiles={data.total_files}
            onPrev={() => setPage(p => p - 1)}
            onNext={() => setPage(p => p + 1)}
          />
        )}
      </div>

      {/* Dirty state bar */}
      {hasChanges && (
        <DirtyStateBar
          toSelect={dirtyChanges.toSelect}
          toUnselect={dirtyChanges.toUnselect}
          saving={saving}
          onDiscard={discardChanges}
          onSave={() => setConfirmOpen(true)}
        />
      )}

      {/* Save confirmation */}
      <ConfirmDialog
        open={confirmOpen}
        title="Save selection changes?"
        description={
          dirtyChanges.toUnselect.length > 0
            ? `You are about to DELETE ingested data (vectors, chunks, graph nodes) for ${dirtyChanges.toUnselect.length} item(s) and mark ${dirtyChanges.toSelect.length} item(s) for ingestion. Files on disk will NOT be deleted. Folder selections will apply to all files within.`
            : `You are about to mark ${dirtyChanges.toSelect.length} item(s) for ingestion on next scan. Folder selections will apply to all files within.`
        }
        confirmText="Confirm"
        destructive={dirtyChanges.toUnselect.length > 0}
        onConfirm={handleSave}
        onCancel={() => setConfirmOpen(false)}
      />
    </div>
  );
}
