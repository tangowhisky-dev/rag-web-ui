"use client";

import { useState } from "react";
import { X } from "lucide-react";
import { api } from "@/lib/api";

export interface ChatPatch {
  title?: string;
  pinned?: boolean;
  temperature?: number;
  model_name?: string;
}

export interface ChatSettingsData {
  id: number;
  title: string;
  temperature?: number;
  model_name?: string;
}

interface ChatSettingsProps {
  chat: ChatSettingsData;
  onClose: () => void;
  onUpdate: (patch: Partial<ChatPatch>) => void;
}

const MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"];

export default function ChatSettings({ chat, onClose, onUpdate }: ChatSettingsProps) {
  const [model, setModel] = useState(chat.model_name ?? "gpt-4o");
  const [temperature, setTemperature] = useState(chat.temperature ?? 0.7);
  const [saving, setSaving] = useState(false);

  const handleApply = async () => {
    setSaving(true);
    const patch: ChatPatch = {
      model_name: model,
      temperature,
    };
    try {
      await api.patch(`/api/chat/${chat.id}`, patch);
      onUpdate(patch);
      onClose();
    } catch (e) {
      console.error("Failed to update chat settings", e);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      data-testid="chat-settings-panel"
      className="absolute right-0 top-0 h-full w-72 bg-background border-l shadow-lg z-40 flex flex-col transition-transform duration-200"
    >
      <div className="flex items-center justify-between px-4 py-3 border-b shrink-0">
        <span className="font-semibold text-sm">Chat Settings</span>
        <button onClick={onClose} aria-label="Close settings" className="p-1 rounded hover:bg-muted">
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* Model selector */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Model</label>
          <select
            data-testid="model-select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full rounded border bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
          >
            {MODELS.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        {/* Temperature slider */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            Temperature: <span data-testid="temperature-value">{temperature.toFixed(1)}</span>
          </label>
          <input
            data-testid="temperature-slider"
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>

      </div>

      <div className="px-4 py-3 border-t shrink-0">
        <button
          data-testid="apply-settings"
          onClick={handleApply}
          disabled={saving}
          className="w-full rounded-lg bg-primary text-primary-foreground px-4 py-2 text-sm font-medium hover:bg-primary/90 disabled:opacity-50 transition-colors"
        >
          {saving ? "Saving…" : "Apply"}
        </button>
      </div>
    </div>
  );
}
