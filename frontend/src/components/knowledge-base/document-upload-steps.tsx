"use client";

import { useState, useCallback, useEffect } from "react";
import { FileIcon, defaultStyles } from "react-file-icon";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/components/ui/use-toast";
import { Loader2, Upload, X, Settings } from "lucide-react";
import { cn } from "@/lib/utils";
import { api, ApiError } from "@/lib/api";
import { useDropzone } from "react-dropzone";

interface DocumentUploadStepsProps {
  knowledgeBaseId: number;
  onComplete?: () => void;
}

interface FileStatus {
  file: File;
  status:
    | "pending"
    | "uploading"
    | "uploaded"
    | "processing"
    | "completed"
    | "error";
  uploadId?: number;
  taskId?: number;     // task_id from the process response — key into taskStatuses
  documentId?: number;
  tempPath?: string;
  error?: string;
}

interface UploadResult {
  upload_id?: number;
  document_id?: number;
  file_name: string;
  status: "exists" | "pending";
  message?: string;
  skip_processing: boolean;
  temp_path?: string;
}

interface TaskResponse {
  tasks: Array<{
    upload_id: number;
    task_id: number;
  }>;
}

interface TaskStatus {
  document_id: number;
  status: "pending" | "processing" | "completed" | "failed";
  error_message?: string;
  progress?: number;         // 0-100
  progress_message?: string; // human-readable stage label
}

interface TaskStatusMap {
  [key: number]: TaskStatus;
}

interface TaskStatusResponse {
  [key: string]: TaskStatus;
}

export function DocumentUploadSteps({
  knowledgeBaseId,
  onComplete,
}: DocumentUploadStepsProps) {
  const [currentStep, setCurrentStep] = useState(1);
  const [files, setFiles] = useState<FileStatus[]>([]);
  const [taskStatuses, setTaskStatuses] = useState<{
    [key: number]: TaskStatus;
  }>({});
  const [isLoading, setIsLoading] = useState(false);
  // Per-file OCR toggle. Defaults to false for PDFs >5 MB (memory pressure),
  // true for everything else (images, scanned docs genuinely need it).
  const [ocrEnabled, setOcrEnabled] = useState<{ [uploadId: number]: boolean }>({});
  const [ocrAvailable, setOcrAvailable] = useState(true);
  // Per-file graph ingestion toggle. Defaults to true (if GRAPHRAG_ENABLED).
  const [graphEnabled, setGraphEnabled] = useState<{ [uploadId: number]: boolean }>({});
  const [graphAvailable, setGraphAvailable] = useState(false);
  const { toast } = useToast();

  // Check whether OCR is available (VISION_MODEL configured).
  useEffect(() => {
    api.get("/api/knowledge-base/ocr-availability").then((data: { ocr_available: boolean }) => {
      setOcrAvailable(data.ocr_available);
    }).catch(() => {
      // Non-fatal — assume available if the endpoint fails.
    });
  }, []);

  // Check whether graph ingestion is available (GRAPHRAG_ENABLED).
  useEffect(() => {
    api.get("/api/config").then((data: { graphrag_enabled?: boolean }) => {
      setGraphAvailable(!!data.graphrag_enabled);
    }).catch(() => {
      // Non-fatal — assume unavailable if the endpoint fails.
    });
  }, []);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles((prev) => [
      ...prev,
      ...acceptedFiles.map((file) => ({
        file,
        status: "pending" as const,
      })),
    ]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
      "application/msword": [".doc"],
      "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
      "application/vnd.ms-powerpoint": [".ppt"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
      "application/vnd.ms-excel": [".xls"],
      "text/plain": [".txt"],
      "text/markdown": [".md"],
      "text/html": [".html", ".htm"],
      "message/rfc822": [".mhtml", ".eml"],
      "text/csv": [".csv"],
      "application/json": [".json"],
      "application/xml": [".xml"],
      "application/vnd.ms-outlook": [".msg"],
      "application/epub+zip": [".epub"],
      "image/jpeg": [".jpg", ".jpeg"],
      "image/png": [".png"],
      "image/gif": [".gif"],
      "image/bmp": [".bmp"],
      "image/tiff": [".tiff"],
      "application/zip": [".zip"],
    },
  });

  const removeFile = (file: File) => {
    setFiles((prev) => prev.filter((f) => f.file !== file));
  };

  // Step 1: Upload files
  const handleFileUpload = async () => {
    const pendingFiles = files.filter((f) => f.status === "pending");
    if (pendingFiles.length === 0) return;

    setIsLoading(true);
    try {
      const formData = new FormData();
      pendingFiles.forEach((fileStatus) => {
        formData.append("files", fileStatus.file);
      });

      const data = (await api.post(
        `/api/knowledge-base/${knowledgeBaseId}/documents/upload`,
        formData,
        {
          headers: {},
        }
      )) as UploadResult[];

      // Update file statuses and initialise OCR + graph defaults
      const newOcrDefaults: { [id: number]: boolean } = {};
      const newGraphDefaults: { [id: number]: boolean } = {};
      setFiles((prev) =>
        prev.map((f) => {
          const uploadResult = data.find((d) => d.file_name === f.file.name);
          if (uploadResult) {
            if (uploadResult.status === "exists") {
              return { ...f, status: "completed", documentId: uploadResult.document_id, error: uploadResult.message };
            } else {
              // Default OCR off for PDFs > 5 MB — they usually have a text layer
              // and the vision model OOMs on long documents.
              const isPdf = f.file.name.toLowerCase().endsWith(".pdf");
              const isLarge = f.file.size > 5 * 1024 * 1024;
              const defaultOcr = !(isPdf && isLarge);
              if (uploadResult.upload_id != null) {
                newOcrDefaults[uploadResult.upload_id] = defaultOcr;
                newGraphDefaults[uploadResult.upload_id] = true;
              }
              return { ...f, status: "uploaded", uploadId: uploadResult.upload_id, tempPath: uploadResult.temp_path };
            }
          }
          return f;
        })
      );
      setOcrEnabled((prev) => ({ ...prev, ...newOcrDefaults }));
      setGraphEnabled((prev) => ({ ...prev, ...newGraphDefaults }));

      setCurrentStep(2);
      toast({
        title: "Upload successful",
        description: `${data.length} files uploaded successfully.`,
      });
    } catch (error) {
      toast({
        title: "Upload failed",
        description:
          error instanceof ApiError ? error.message : "Something went wrong",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Step 2: Process documents
  const handleProcess = async (uploadResults?: UploadResult[]) => {
    const resultsToProcess =
      uploadResults ||
      files
        .filter((f) => f.status === "uploaded")
        .map((f) => ({
          upload_id: f.uploadId!,
          file_name: f.file.name,
          status: "pending" as const,
          skip_processing: false,
          temp_path: f.tempPath!,
          enable_ocr: ocrEnabled[f.uploadId!] ?? true,
          enable_graph: graphEnabled[f.uploadId!] ?? true,
        }));

    if (resultsToProcess.length === 0) return;

    setIsLoading(true);
    try {
      const data = (await api.post(
        `/api/knowledge-base/${knowledgeBaseId}/documents/process`,
        resultsToProcess
      )) as TaskResponse;

      // Initialize task statuses
      const initialStatuses = data.tasks.reduce<TaskStatusMap>(
        (acc, task) => ({
          ...acc,
          [task.task_id]: {
            document_id: task.upload_id,
            status: "pending" as const,
          },
        }),
        {}
      );
      setTaskStatuses(initialStatuses);

      // Stamp taskId onto each file so the progress render can look up taskStatuses[file.taskId]
      setFiles((prev) =>
        prev.map((f) => {
          const t = data.tasks.find((t) => t.upload_id === f.uploadId);
          return t ? { ...f, taskId: t.task_id } : f;
        })
      );

      // Start polling for task status
      pollTaskStatus(data.tasks.map((t) => t.task_id));
    } catch (error) {
      setIsLoading(false);
      toast({
        title: "Processing failed",
        description:
          error instanceof ApiError ? error.message : "Something went wrong",
        variant: "destructive",
      });
    }
  };

  // Poll task status
  const pollTaskStatus = async (taskIds: number[]) => {
    let consecutiveErrors = 0;
    const MAX_ERRORS = 10;       // give up only after 10 consecutive failures
    const BASE_DELAY = 3000;     // 3s base — less hammering during heavy ingestion
    const ERROR_DELAY = 8000;    // back off on error so a busy backend gets room

    const poll = async () => {
      try {
        const response = (await api.get(
          `/api/knowledge-base/${knowledgeBaseId}/documents/tasks?task_ids=${taskIds.join(
            ","
          )}`
        )) as TaskStatusResponse;

        consecutiveErrors = 0;  // reset on success

        // Convert string keys to numbers
        const data = Object.entries(response).reduce<TaskStatusMap>(
          (acc, [key, value]) => ({
            ...acc,
            [parseInt(key)]: value,
          }),
          {}
        );

        setTaskStatuses(data);

        const allDone = Object.values(data).every(
          (task) => task.status === "completed" || task.status === "failed"
        );

        if (allDone) {
          setIsLoading(false);
          const hasErrors = Object.values(data).some(
            (task) => task.status === "failed"
          );
          if (!hasErrors) {
            toast({
              title: "Processing completed",
              description: "All documents have been processed successfully.",
            });
            onComplete?.();
          } else {
            toast({
              title: "Processing completed with errors",
              description: "Some documents failed to process.",
              variant: "destructive",
            });
          }
        } else {
          setTimeout(poll, BASE_DELAY);
        }
      } catch (error) {
        consecutiveErrors++;
        if (consecutiveErrors >= MAX_ERRORS) {
          // Backend appears genuinely dead — stop polling
          setIsLoading(false);
          toast({
            title: "Lost contact with server",
            description: "Processing may still be running. Refresh the page to check status.",
            variant: "destructive",
          });
          return;
        }
        // Network blip (ECONNRESET, timeout, etc.) — retry silently
        setTimeout(poll, ERROR_DELAY);
      }
    };

    poll();
  };

  const handleProcessClick = (e: React.MouseEvent) => {
    e.preventDefault();
    handleProcess();
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="mb-8">
        <div className="flex justify-between mb-2">
          {[
            { step: 1, icon: Upload, label: "Upload" },
            { step: 2, icon: Settings, label: "Process" },
          ].map(({ step, icon: Icon, label }, index, array) => (
            <div
              key={step}
              className="flex flex-col items-center space-y-2 flex-1"
            >
              <div
                className={cn(
                  "w-12 h-12 rounded-full flex items-center justify-center border-2 transition-colors",
                  currentStep === step
                    ? "bg-primary text-primary-foreground border-primary"
                    : currentStep > step
                    ? "bg-primary/20 border-primary/20"
                    : "bg-background border-input"
                )}
              >
                <Icon className="w-6 h-6" />
              </div>
              <span className="text-sm font-medium">
                {step}. {label}
              </span>
              {index < array.length - 1 && (
                <div
                  className={cn(
                    "h-0.5 w-full mt-2",
                    currentStep > step ? "bg-primary/20" : "bg-input"
                  )}
                />
              )}
            </div>
          ))}
        </div>
      </div>

      <Tabs value={String(currentStep)} className="w-full">
        <TabsContent value="1" className="mt-6">
          <Card className="p-6">
            <div className="space-y-4">
              <div
                {...getRootProps()}
                className={cn(
                  "border-2 border-dashed rounded-lg p-8 text-center transition-colors",
                  isDragActive
                    ? "border-primary bg-primary/5"
                    : "hover:border-primary/50"
                )}
              >
                <input {...getInputProps()} />
                <Upload className="w-12 h-12 mx-auto text-muted-foreground" />
                <p className="mt-2 text-sm font-medium">
                  Drop your files here or click to browse
                </p>
                <p className="text-xs text-muted-foreground">
                  PDF, Word, PowerPoint, Excel, HTML, CSV, JSON, XML, Markdown, TXT, EPUB, Images (OCR), ZIP, EML, MSG
                </p>
              </div>
              {files.length > 0 && (
                <div className="space-y-2 max-h-[300px] overflow-y-auto">
                  {files.map((fileStatus) => (
                    <div
                      key={fileStatus.file.name}
                      className="flex items-center justify-between p-4 rounded-lg border"
                    >
                      <div className="flex items-center space-x-4">
                        <div className="w-8 h-8">
                          <FileIcon
                            extension={fileStatus.file.name.split(".").pop()}
                            {...defaultStyles[
                              fileStatus.file.name
                                .split(".")
                                .pop() as keyof typeof defaultStyles
                            ]}
                          />
                        </div>
                        <div>
                          <p className="text-sm font-medium">
                            {fileStatus.file.name}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {(fileStatus.file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        {fileStatus.status === "uploaded" && (
                          <span className="text-green-500 text-sm">
                            Uploaded
                          </span>
                        )}
                        {fileStatus.status === "error" && (
                          <span className="text-red-500 text-sm">
                            {fileStatus.error}
                          </span>
                        )}
                        <button
                          onClick={() => removeFile(fileStatus.file)}
                          className="p-1 hover:bg-accent rounded-full"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <Button
                onClick={handleFileUpload}
                disabled={
                  !files.some((f) => f.status === "pending") || isLoading
                }
                className="w-full"
              >
                {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Upload Files
              </Button>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="2" className="mt-6">
          <Card className="p-6">
            <div className="space-y-4">
              <div className="max-h-[300px] overflow-y-auto space-y-2 rounded-lg border p-4">
                {files
                  .filter((f) => f.status === "uploaded")
                  .map((file) => {
                    const task = file.taskId != null
                      ? taskStatuses[file.taskId]
                      : Object.values(taskStatuses).find((t) => t.document_id === file.documentId);
                    return (
                      <div
                        key={file.uploadId}
                        className="p-4 border rounded-lg space-y-2"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            <div className="w-8 h-8">
                              <FileIcon
                                extension={file.file.name.split(".").pop()}
                                {...defaultStyles[
                                  file.file.name
                                    .split(".")
                                    .pop() as keyof typeof defaultStyles
                                ]}
                              />
                            </div>
                            <div>
                              <p className="text-sm font-medium">
                                {file.file.name}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                {(file.file.size / 1024 / 1024).toFixed(2)} MB
                              </p>
                              {task && (
                                <p className="text-xs text-muted-foreground">
                                  Status: {task.status || "pending"}
                                </p>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center gap-4">
                            <div className="flex items-center gap-2">
                              <Switch
                                id={`ocr-${file.uploadId}`}
                                checked={ocrAvailable && (ocrEnabled[file.uploadId!] ?? true)}
                                onCheckedChange={(v) =>
                                  setOcrEnabled((prev) => ({ ...prev, [file.uploadId!]: v }))
                                }
                                disabled={isLoading || !ocrAvailable}
                              />
                              <label
                                htmlFor={`ocr-${file.uploadId}`}
                                className="text-xs text-muted-foreground cursor-pointer select-none"
                                title={ocrAvailable ? undefined : "OCR is not available — no vision model configured"}
                              >
                                OCR{!ocrAvailable && " (unavailable)"}
                              </label>
                            </div>
                            <div className="flex items-center gap-2">
                              <Switch
                                id={`graph-${file.uploadId}`}
                                checked={graphAvailable && (graphEnabled[file.uploadId!] ?? true)}
                                onCheckedChange={(v) =>
                                  setGraphEnabled((prev) => ({ ...prev, [file.uploadId!]: v }))
                                }
                                disabled={isLoading || !graphAvailable}
                              />
                              <label
                                htmlFor={`graph-${file.uploadId}`}
                                className="text-xs text-muted-foreground cursor-pointer select-none"
                                title={graphAvailable ? undefined : "Graph ingestion is not enabled — set GRAPHRAG_ENABLED=true"}
                              >
                                Graph{!graphAvailable && " (unavailable)"}
                              </label>
                            </div>
                            {task?.status === "failed" && (
                              <p className="text-sm text-destructive">
                                {task.error_message}
                              </p>
                            )}
                          </div>
                        </div>
                        {task &&
                          (task.status === "pending" ||
                            task.status === "processing") && (
                            <div className="space-y-1">
                              <Progress
                                value={task.progress ?? (task.status === "processing" ? 10 : 5)}
                                className="w-full"
                              />
                              {task.progress_message && (
                                <p className="text-xs text-muted-foreground">
                                  {task.progress_message}
                                </p>
                              )}
                            </div>
                          )}
                      </div>
                    );
                  })}
              </div>

              <Button
                onClick={handleProcessClick}
                disabled={
                  isLoading ||
                  files.filter((f) => f.status === "uploaded").length === 0
                }
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Settings className="mr-2 h-4 w-4" />
                    Process
                  </>
                )}
              </Button>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
