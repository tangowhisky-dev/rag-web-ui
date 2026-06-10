'use client';

import Link from 'next/link';
import { Upload, Link2 } from 'lucide-react';

export function Step2Options() {
  return (
    <div className="flex flex-col gap-3">
      <Link
        href="/dashboard/knowledge"
        className="relative flex flex-col items-center justify-center rounded-2xl border bg-card text-card-foreground p-4 hover:shadow-md hover:border-foreground/30 transition-all"
      >
        <span className="absolute top-2 left-2 flex h-7 w-7 items-center justify-center rounded-full bg-muted text-xs font-bold">
          2
        </span>
        <div className="rounded-full bg-muted p-3 mb-2">
          <Upload className="h-6 w-6 text-foreground" />
        </div>
        <h3 className="text-base font-medium mb-1">Upload Documents</h3>
        <p className="text-sm text-muted-foreground text-center">
          Add PDFs, docs, images, or CSVs to your knowledge bases
        </p>
      </Link>
      <Link
        href="/dashboard/admin/data-sources"
        className="relative flex flex-col items-center justify-center rounded-2xl border bg-card text-card-foreground p-4 hover:shadow-md hover:border-foreground/30 transition-all"
      >
        <span className="absolute top-2 left-2 flex h-7 w-7 items-center justify-center rounded-full bg-muted text-xs font-bold">
          2
        </span>
        <div className="rounded-full bg-muted p-3 mb-2">
          <Link2 className="h-6 w-6 text-foreground" />
        </div>
        <h3 className="text-base font-medium mb-1">Link Data Sources</h3>
        <p className="text-sm text-muted-foreground text-center">
          Connect SMB shares or cloud storage — auto-synced
        </p>
      </Link>
    </div>
  );
}
