"use client";

import { useParams } from "next/navigation";
import { useState, useCallback, useEffect } from "react";
import { DocumentUploadSteps } from "@/components/knowledge-base/document-upload-steps";
import { DocumentList } from "@/components/knowledge-base/document-list";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { PlusIcon, LinkIcon, XIcon } from "lucide-react";
import KnowledgeLayout from "@/components/layout/knowledge-layout";
import { api, ApiError } from "@/lib/api";
import { useToast } from "@/components/ui/use-toast";

interface DataSource {
  id: number;
  name: string;
  folder_path: string;
}

interface AvailableDataSource extends DataSource {
  assigned_orgs: Array<{ id: number; name: string }>;
}

export default function KnowledgeBasePage() {
  const params = useParams();
  const knowledgeBaseId = parseInt(params?.id as string);
  const [refreshKey, setRefreshKey] = useState(0);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [kbName, setKbName] = useState<string>("Loading...");
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [linkDialogOpen, setLinkDialogOpen] = useState(false);
  const [availableSources, setAvailableSources] = useState<AvailableDataSource[]>([]);
  const [selectedSourceId, setSelectedSourceId] = useState<string>("");
  const [linking, setLinking] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    api.get(`/api/knowledge-base/${knowledgeBaseId}`)
      .then((data) => {
        setKbName(data.name || "Knowledge Base");
        setDataSources(data.data_sources || []);
        setLoading(false);
      })
      .catch(() => {
        setKbName("Knowledge Base");
        setLoading(false);
      });
  }, [knowledgeBaseId]);

  const fetchAvailableSources = async () => {
    try {
      const data = await api.get("/api/knowledge-base/available-datastores");
      setAvailableSources(data as AvailableDataSource[]);
    } catch (err) {
      console.error("Failed to fetch data stores:", err);
    }
  };

  const handleLinkSource = async () => {
    if (!selectedSourceId) {
      toast({
        title: "Error",
        description: "Please select a data store",
        variant: "destructive",
      });
      return;
    }

    setLinking(true);
    try {
      await api.post(`/api/knowledge-base/${knowledgeBaseId}/link-datastore`, {
        data_store_id: parseInt(selectedSourceId),
      });
      toast({
        title: "Success",
        description: "Data store linked to knowledge base",
      });
      setLinkDialogOpen(false);
      setSelectedSourceId("");
      // Refresh data stores
      const data = await api.get(`/api/knowledge-base/${knowledgeBaseId}`);
      setDataSources(data.data_sources || []);
    } catch (err) {
      toast({
        title: "Error",
        description: (err as ApiError).message ?? "Failed to link data store",
        variant: "destructive",
      });
    } finally {
      setLinking(false);
    }
  };

  const handleUnlinkSource = async (sourceId: number) => {
    try {
      await api.delete(`/api/knowledge-base/${knowledgeBaseId}/unlink-datastore/${sourceId}`);
      toast({
        title: "Success",
        description: "Data store unlinked from knowledge base",
      });
      // Refresh data stores
      const data = await api.get(`/api/knowledge-base/${knowledgeBaseId}`);
      setDataSources(data.data_sources || []);
    } catch (err) {
      toast({
        title: "Error",
        description: (err as ApiError).message ?? "Failed to unlink data store",
        variant: "destructive",
      });
    }
  };

  const handleUploadComplete = useCallback(() => {
    setRefreshKey((prev) => prev + 1);
    setDialogOpen(false);
  }, []);

  const handleDialogOpenChange = useCallback((open: boolean) => {
    setDialogOpen(open);
    if (!open) {
      setRefreshKey((prev) => prev + 1);
    }
  }, []);

  const handleLinkDialogOpenChange = useCallback((open: boolean) => {
    setLinkDialogOpen(open);
    if (open) {
      fetchAvailableSources();
    }
  }, []);

  return (
    <KnowledgeLayout pageTitle={kbName}>
      <div className="h-full overflow-y-auto">
        <div className="p-6 pt-16 space-y-6">
          <div className="flex justify-between items-center">
            <h1 className="text-3xl font-bold tracking-tight">{loading ? "Loading..." : kbName}</h1>
          </div>

          {loading ? (
            <div className="flex justify-center items-center h-64">
              <div className="text-muted-foreground">Loading knowledge base...</div>
            </div>
          ) : (
            <>
            <Dialog open={dialogOpen} onOpenChange={handleDialogOpenChange}>
              <DialogTrigger asChild>
                <Button>
                  <PlusIcon className="w-4 h-4 mr-2" />
                  Add Document
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-4xl">
                <DialogHeader>
                  <DialogTitle>Add Document</DialogTitle>
                  <DialogDescription>
                    Upload a document to your knowledge base. Supported formats:
                    PDF, DOCX, Markdown, and Text files.
                  </DialogDescription>
                </DialogHeader>
                <DocumentUploadSteps
                  knowledgeBaseId={knowledgeBaseId}
                  onComplete={handleUploadComplete}
                />
              </DialogContent>
            </Dialog>

            {/* Data Stores Section */}
            <div className="border rounded-lg p-4 bg-card">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-lg font-semibold">Linked Data Stores</h2>
                <Button variant="outline" size="sm" onClick={() => handleLinkDialogOpenChange(true)}>
                  <LinkIcon className="w-4 h-4 mr-2" />
                  Link Data Store
                </Button>
              </div>
              {dataSources.length === 0 ? (
                <p className="text-sm text-muted-foreground">No data stores linked. Link a data store to automatically ingest documents.</p>
              ) : (
                <div className="space-y-2">
                  {dataSources.map((ds) => (
                    <div key={ds.id} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                      <div className="flex-1">
                        <div className="font-medium flex items-center gap-2">
                          {ds.name}
                          <Badge variant="secondary" className="text-xs">
                            Auto-process
                          </Badge>
                        </div>
                        <div className="text-xs text-muted-foreground truncate">{ds.folder_path}</div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleUnlinkSource(ds.id)}
                        title="Unlink data store"
                      >
                        <XIcon className="w-4 h-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Link Data Store Dialog */}
            <Dialog open={linkDialogOpen} onOpenChange={handleLinkDialogOpenChange}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Link Data Store</DialogTitle>
                  <DialogDescription>
                    Select a data store to link to this knowledge base. Documents from the linked data store will be automatically available.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <Select value={selectedSourceId} onValueChange={setSelectedSourceId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a data store" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableSources.map((ds) => (
                        <SelectItem key={ds.id} value={String(ds.id)}>
                          <div className="flex items-center justify-between">
                            <span>{ds.name}</span>
                            <span className="text-xs text-muted-foreground">
                              {ds.folder_path}
                            </span>
                          </div>
                        </SelectItem>
                      ))}
                      {availableSources.length === 0 && (
                        <div className="p-4 text-center text-muted-foreground">
                          No data stores available. Ask an admin to configure data stores for your organization.
                        </div>
                      )}
                    </SelectContent>
                  </Select>
                  <Button
                    onClick={handleLinkSource}
                    disabled={!selectedSourceId || linking}
                    className="w-full"
                  >
                    {linking ? "Linking..." : "Link Data Store"}
                  </Button>
                </div>
              </DialogContent>
            </Dialog>

            <div>
              <DocumentList knowledgeBaseId={knowledgeBaseId} refreshKey={refreshKey} />
            </div>
            </>
          )}
        </div>
      </div>
    </KnowledgeLayout>
  );
}
