"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { FileIcon, defaultStyles } from "react-file-icon";
import { ArrowRight, Plus, Settings, Trash2, Search } from "lucide-react";
import KnowledgeLayout from "@/components/layout/knowledge-layout";
import { useKnowledgeContext } from "@/contexts/knowledge-context";
import { ApiError } from "@/lib/api";
import { useToast } from "@/components/ui/use-toast";

interface KbDocument {
  id: number;
  file_name: string;
  created_at: string;
}

function KnowledgeBaseList() {
  const { kbList, deleteKb, refreshKbList } = useKnowledgeContext();
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    refreshKbList().finally(() => setLoading(false));
  }, [refreshKbList]);

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to delete this knowledge base?")) return;
    try {
      await deleteKb(id);
      toast({ title: "Success", description: "Knowledge base deleted successfully" });
    } catch (error) {
      if (error instanceof ApiError) {
        toast({ title: "Error", description: error.message, variant: "destructive" });
      }
    }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="space-y-8 p-6 pt-16">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Knowledge Bases</h2>
            <p className="text-muted-foreground">Manage your knowledge bases and documents</p>
          </div>
          <Link
            href="/dashboard/knowledge/new"
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            <Plus className="mr-2 h-4 w-4" />
            New Knowledge Base
          </Link>
        </div>

        <div className="grid gap-6">
          {kbList.map((kb) => (
            <div key={kb.id} className="rounded-lg border bg-card p-6 space-y-4">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-lg font-semibold">{kb.name}</h3>
                  <p className="text-sm text-muted-foreground">{kb.description || "No description"}</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    {kb.documents.length} documents • {new Date(kb.created_at).toLocaleDateString()}
                  </p>
                </div>
                <div className="flex space-x-2">
                  <Link
                    href={`/dashboard/knowledge/${kb.id}`}
                    className="inline-flex items-center justify-center rounded-md bg-secondary w-8 h-8"
                  >
                    <Settings className="h-4 w-4" />
                  </Link>
                  <Link
                    href={`/dashboard/test-retrieval/${kb.id}`}
                    className="inline-flex items-center justify-center rounded-md bg-secondary w-8 h-8"
                  >
                    <Search className="h-4 w-4" />
                  </Link>
                  <button
                    onClick={() => handleDelete(kb.id)}
                    className="inline-flex items-center justify-center rounded-md bg-destructive/10 hover:bg-destructive/20 w-8 h-8"
                  >
                    <Trash2 className="h-4 w-4 text-destructive" />
                  </button>
                </div>
              </div>

              {kb.documents.length > 0 && (
                <div className="border-t pt-4">
                  <h4 className="text-sm font-medium mb-2">Documents</h4>
                  <div className="flex flex-wrap gap-2 max-h-[400px] overflow-y-auto">
                    {kb.documents.slice(0, 9).map((doc) => (
                      <div
                        key={doc.id}
                        className="flex flex-col items-center gap-2 p-2 rounded-lg border bg-card hover:bg-accent/50 cursor-pointer transition-colors w-[150px] h-[150px] justify-center"
                      >
                        <div className="w-8 h-8 mb-2">
                          {(() => {
                            const ext = ((doc.file_name ?? "").split(".").pop() || "").toLowerCase();
                            const style = defaultStyles[ext as keyof typeof defaultStyles];
                            return style
                              ? <FileIcon extension={ext} {...style} />
                              : <FileIcon extension={ext} color="#E2E8F0" labelColor="#94A3B8" />;
                          })()}
                        </div>
                        <div className="text-sm font-medium text-center max-w-[100px]">
                          <div className="line-clamp-2 overflow-hidden text-ellipsis">{doc.file_name ?? ""}</div>
                        </div>
                        <span className="text-xs text-muted-foreground mt-1">
                          {doc.created_at ? new Date(doc.created_at).toLocaleDateString() : ""}
                        </span>
                      </div>
                    ))}
                    {kb.documents.length > 9 && (
                      <Link
                        href={`/dashboard/knowledge/${kb.id}`}
                        className="flex flex-col items-center p-2 rounded-lg border bg-card hover:bg-accent/50 cursor-pointer transition-colors w-[150px] h-[150px] justify-center"
                      >
                        <div className="w-8 h-8 mb-2 flex items-center justify-center">
                          <ArrowRight className="w-6 h-6" />
                        </div>
                        <span className="text-sm font-medium text-center">View All Documents</span>
                        <span className="text-xs text-muted-foreground mt-1">{kb.documents.length} total</span>
                      </Link>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {!loading && kbList.length === 0 && (
            <div className="text-center py-12">
              <p className="text-muted-foreground">No knowledge bases found. Create one to get started.</p>
            </div>
          )}

          {loading && (
            <div className="flex items-center justify-center py-12">
              <div className="space-y-4">
                <div className="w-8 h-8 border-4 border-primary/30 border-t-primary rounded-full animate-spin mx-auto"></div>
                <p className="text-muted-foreground animate-pulse">Loading knowledge bases...</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function KnowledgeBasePage() {
  return (
    <KnowledgeLayout>
      <KnowledgeBaseList />
    </KnowledgeLayout>
  );
}
