"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import KnowledgeLayout from "@/components/layout/knowledge-layout";
import { api, ApiError } from "@/lib/api";
import { useToast } from "@/components/ui/use-toast";

interface DataSource {
  id: number;
  name: string;
  folder_path: string;
  description: string | null;
  assigned_orgs?: Array<{ org_id: number; org_name: string }>;
}

export default function NewKnowledgeBasePage() {
  const router = useRouter();
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [availableSources, setAvailableSources] = useState<DataSource[]>([]);
  const [selectedSources, setSelectedSources] = useState<number[]>([]);
  const { toast } = useToast();

  useEffect(() => {
    // Fetch available data sources for the user's organization
    api.get("/api/knowledge-base/available-datastores")
      .then((data) => {
        setAvailableSources(data as DataSource[]);
      })
      .catch((err) => {
        console.error("Failed to fetch data sources:", err);
      });
  }, []);

  const handleSourceToggle = (sourceId: number) => {
    setSelectedSources((prev) =>
      prev.includes(sourceId)
        ? prev.filter((id) => id !== sourceId)
        : [...prev, sourceId]
    );
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError("");
    setIsSubmitting(true);

    try {
      const formData = new FormData(e.currentTarget);
      const name = formData.get("name") as string;
      const description = formData.get("description") as string;

      const data = await api.post("/api/knowledge-base", {
        name,
        description,
      });

      // Link selected data sources
      const linkFailures: string[] = [];
      for (const sourceId of selectedSources) {
        try {
          await api.post(`/api/knowledge-base/${data.id}/link-datastore`, {
            data_store_id: sourceId,
          });
        } catch (err) {
          const source = availableSources.find((s) => s.id === sourceId);
          const sourceName = source?.name ?? `ID ${sourceId}`;
          linkFailures.push(sourceName);
          console.error(`Failed to link data source ${sourceId}:`, err);
        }
      }

      if (linkFailures.length > 0) {
        toast({
          title: "Data source linking failed",
          description: `Knowledge base created, but ${linkFailures.length} data source(s) failed to link: ${linkFailures.join(", ")}. You can retry from the KB page.`,
          variant: "destructive",
        });
      }

      router.push(`/dashboard/knowledge/${data.id}`);
    } catch (error) {
      console.error("Failed to create knowledge base:", error);
      if (error instanceof ApiError) {
        setError(error.message);
        toast({
          title: "Error",
          description: error.message,
          variant: "destructive",
        });
      } else {
        setError("Failed to create knowledge base");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <KnowledgeLayout>
      <div className="h-full overflow-y-auto">
        <div className="max-w-2xl mx-auto space-y-8 p-6 pt-16">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">
              Create Knowledge Base
            </h2>
            <p className="text-muted-foreground">
              Create a new knowledge base to store your documents
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <label
                htmlFor="name"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Name
              </label>
              <input
                id="name"
                name="name"
                type="text"
                required
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                placeholder="Enter knowledge base name"
              />
            </div>

            <div className="space-y-2">
              <label
                htmlFor="description"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Description
              </label>
              <textarea
                id="description"
                name="description"
                className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                placeholder="Enter knowledge base description"
              />
            </div>

            {/* Data Sources Selection */}
            {availableSources.length > 0 && (
              <div className="space-y-3">
                <label className="text-sm font-medium leading-none">
                  Link Data Sources (optional)
                </label>
                <p className="text-xs text-muted-foreground">
                  Select data sources to automatically ingest documents into this knowledge base
                </p>
                <div className="space-y-2 border rounded-lg p-3 max-h-60 overflow-y-auto">
                  {availableSources.map((source) => (
                    <div
                      key={source.id}
                      className="flex items-start space-x-3 p-2 rounded-lg hover:bg-muted/50"
                    >
                      <input
                        type="checkbox"
                        id={`source-${source.id}`}
                        checked={selectedSources.includes(source.id)}
                        onChange={() => {
                          handleSourceToggle(source.id);
                        }}
                        className="mt-0.5 h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                      />
                      <div className="flex-1 space-y-1">
                        <label
                          htmlFor={`source-${source.id}`}
                          className="text-sm font-medium leading-none cursor-pointer"
                        >
                          {source.name}
                        </label>
                        <p className="text-xs text-muted-foreground">
                          {source.folder_path}
                        </p>
                        {source.description && (
                          <p className="text-xs text-muted-foreground">
                            {source.description}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {availableSources.length === 0 && (
              <div className="p-4 border rounded-lg bg-muted/50">
                <p className="text-sm text-muted-foreground">
                  No data sources available for your organization. Ask an admin to configure data sources.
                </p>
              </div>
            )}

            {error && <div className="text-sm text-red-500">{error}</div>}

            <div className="flex justify-end space-x-4">
              <button
                type="button"
                onClick={() => router.back()}
                className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-10 px-4 py-2"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSubmitting}
                className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2"
              >
                {isSubmitting ? "Creating..." : "Create"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </KnowledgeLayout>
  );
}
