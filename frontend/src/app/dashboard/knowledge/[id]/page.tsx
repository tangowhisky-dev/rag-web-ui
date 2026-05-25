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
import { PlusIcon } from "lucide-react";
import KnowledgeLayout from "@/components/layout/knowledge-layout";
import { api } from "@/lib/api";

export default function KnowledgeBasePage() {
  const params = useParams();
  const knowledgeBaseId = parseInt(params.id as string);
  const [refreshKey, setRefreshKey] = useState(0);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [kbName, setKbName] = useState<string | undefined>();

  useEffect(() => {
    api.get(`/api/knowledge-base/${knowledgeBaseId}`)
      .then((data) => setKbName(data.name))
      .catch(() => {});
  }, [knowledgeBaseId]);

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

  return (
    <KnowledgeLayout pageTitle={kbName}>
      <div className="h-full overflow-y-auto">
        <div className="p-6 pt-16">
          <div className="flex justify-between items-center mb-8">
            <h1 className="text-3xl font-bold">{kbName || "Knowledge Base"}</h1>
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
          </div>
          <div className="mt-8">
            <DocumentList knowledgeBaseId={knowledgeBaseId} refreshKey={refreshKey} />
          </div>
        </div>
      </div>
    </KnowledgeLayout>
  );
}
