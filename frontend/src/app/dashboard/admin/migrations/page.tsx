'use client';

import { useState, useEffect, useCallback } from 'react';
import { api } from '@/lib/api';
import { fetchTokenClaims } from '@/lib/auth';

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
import { useToast } from '@/components/ui/use-toast';
import { LoadingDots } from '@/components/ui/loading-dots';

type Kind = 'datastores' | 'knowledge-bases';

interface Bm25Job {
  phase: string; // schema | backfill | titles | verify | done | error
  points_total: number;
  points_done: number;
  title_points: number;
  chunks_missing_bm25: number | null;
  verified: boolean;
  error: string | null;
}

interface Bm25Status {
  id: number;
  name: string;
  collection: string;
  exists: boolean;
  schema_ready?: boolean;
  needs_migration?: boolean;
  chunks_missing_bm25?: number | null;
  points_total?: number;
  job: Bm25Job | null;
}

interface MigrationGroups {
  datastores: Bm25Status[];
  knowledge_bases: Bm25Status[];
}

function StatusCell({
  st,
  kind,
  migrating,
  onMigrate,
}: {
  st: Bm25Status;
  kind: Kind;
  migrating: boolean;
  onMigrate: (kind: Kind, id: number) => void;
}) {
  if (!st.exists) {
    return <span className="text-xs text-muted-foreground">No collection</span>;
  }
  const job = st.job;
  const running = job !== null && job.phase !== 'done' && job.phase !== 'error';
  if (running && job) {
    const pct = job.points_total > 0
      ? Math.min((job.points_done / job.points_total) * 100, 100)
      : 0;
    return (
      <div className="space-y-1 min-w-[160px]">
        <div className="flex items-center gap-1.5">
          <LoadingDots size="sm" />
          <span className="text-xs text-blue-600 capitalize">{job.phase}...</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
          <div
            className="bg-blue-500 h-2 transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{job.points_done} / {job.points_total}</span>
          <span>{pct.toFixed(0)}%</span>
        </div>
      </div>
    );
  }
  if (job?.phase === 'error') {
    return (
      <div className="space-y-1 min-w-[160px]">
        <div className="text-[10px] text-red-500" title={job.error ?? undefined}>
          Migration failed{job.error ? `: ${job.error}` : ''}
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => onMigrate(kind, st.id)}
          disabled={migrating}
        >
          Retry
        </Button>
      </div>
    );
  }
  if (st.needs_migration || st.schema_ready === false) {
    return (
      <div className="flex items-center gap-2">
        <Badge variant="secondary" className="bg-amber-100 text-amber-800 text-[10px]">
          Needs migration
        </Badge>
        <Button
          variant="outline"
          size="sm"
          onClick={() => onMigrate(kind, st.id)}
          disabled={migrating}
          title="Backfill BM25 vectors on the Qdrant collection (in-place, no downtime)"
        >
          Migrate BM25
        </Button>
      </div>
    );
  }
  return (
    <Badge
      variant="secondary"
      className={`${job?.verified ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-700'} text-[10px]`}
      title={job ? `${job.title_points} title points` : undefined}
    >
      {job?.verified ? 'BM25 ✓' : 'BM25'}
    </Badge>
  );
}

function MigrationTable({
  title,
  kind,
  items,
  migrating,
  onMigrate,
}: {
  title: string;
  kind: Kind;
  items: Bm25Status[];
  migrating: Set<string>;
  onMigrate: (kind: Kind, id: number) => void;
}) {
  return (
    <div className="space-y-2">
      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Collection</TableHead>
            <TableHead>Points</TableHead>
            <TableHead>Missing BM25</TableHead>
            <TableHead>Status</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.length === 0 ? (
            <TableRow>
              <TableCell colSpan={5} className="text-center text-muted-foreground">
                None configured.
              </TableCell>
            </TableRow>
          ) : (
            items.map((st) => (
              <TableRow key={st.id}>
                <TableCell className="font-medium">{st.name}</TableCell>
                <TableCell className="text-xs text-muted-foreground font-mono">
                  {st.collection}
                </TableCell>
                <TableCell className="text-xs tabular-nums">
                  {st.exists ? st.points_total ?? '—' : '—'}
                </TableCell>
                <TableCell className="text-xs tabular-nums">
                  {st.exists ? st.chunks_missing_bm25 ?? '—' : '—'}
                </TableCell>
                <TableCell>
                  <StatusCell
                    st={st}
                    kind={kind}
                    migrating={migrating.has(`${kind}:${st.id}`)}
                    onMigrate={onMigrate}
                  />
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}

export default function MigrationsPage() {
  const { toast } = useToast();
  const [isSuperAdmin, setIsSuperAdmin] = useState(false);
  const [loading, setLoading] = useState(true);
  const [groups, setGroups] = useState<MigrationGroups>({ datastores: [], knowledge_bases: [] });
  const [migrating, setMigrating] = useState<Set<string>>(new Set());

  useEffect(() => {
    fetchTokenClaims().then((claims) => {
      setIsSuperAdmin(claims?.role === 'super_admin');
    });
  }, []);

  const fetchStatuses = useCallback(async () => {
    try {
      const data = await api.get('/api/admin/migrations/bm25') as MigrationGroups;
      setGroups(data);
    } catch {
      // Qdrant may be unavailable — leave stale state
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!isSuperAdmin) return;
    void Promise.resolve().then(fetchStatuses);
    const id = setInterval(fetchStatuses, 5000);
    return () => clearInterval(id);
  }, [isSuperAdmin, fetchStatuses]);

  const handleMigrate = async (kind: Kind, id: number) => {
    const key = `${kind}:${id}`;
    setMigrating((prev) => new Set(prev).add(key));
    try {
      await api.post(`/api/admin/migrations/bm25/${kind}/${id}`, {});
      await fetchStatuses();
    } catch (err) {
      toast({
        title: 'BM25 migration failed to start',
        description: (err as { message?: string }).message ?? 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setMigrating((prev) => {
        const next = new Set(prev);
        next.delete(key);
        return next;
      });
    }
  };

  if (!isSuperAdmin) {
    return null;
  }

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 pt-16 space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Migrations</h1>
        <p className="text-muted-foreground">
          Qdrant BM25 vector backfill for datastore and knowledge base collections.
          Runs in place — no downtime, safe to re-run.
        </p>
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : (
        <>
          <MigrationTable
            title="Data Stores"
            kind="datastores"
            items={groups.datastores}
            migrating={migrating}
            onMigrate={handleMigrate}
          />
          <MigrationTable
            title="Knowledge Bases"
            kind="knowledge-bases"
            items={groups.knowledge_bases}
            migrating={migrating}
            onMigrate={handleMigrate}
          />
        </>
      )}
    </div>
  );
}
