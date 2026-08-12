'use client';

import { useState, useRef, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/use-toast';
import { api } from '@/lib/api';
import { RefreshCw } from 'lucide-react';

interface ModelPickerProps {
  value: string | null;
  apiBase: string | null;
  apiKey: string | null;
  fetchUrl: string;
  onChange: (v: string | null) => void;
  placeholder?: string;
}

export function ModelPicker({
  value,
  apiBase,
  apiKey,
  fetchUrl,
  onChange,
  placeholder,
}: ModelPickerProps) {
  const { toast } = useToast();
  const [models, setModels] = useState<string[]>([]);
  const [fetching, setFetching] = useState(false);
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const canFetch = !!apiBase && !fetching;

  async function fetchModels() {
    if (!apiBase) return;
    setFetching(true);
    try {
      const params = new URLSearchParams({ api_base: apiBase });
      if (apiKey && !apiKey.startsWith('••••')) {
        params.set('api_key', apiKey);
      }
      const data = await api.get(`${fetchUrl}?${params.toString()}`) as { models: string[] };
      setModels(data.models || []);
      setOpen(true);
      if (data.models.length === 0) {
        toast({ title: 'No models returned', description: 'The endpoint returned an empty model list.' });
      }
    } catch (err) {
      toast({
        title: 'Failed to fetch models',
        description: (err as { message?: string }).message ?? 'Check the API base URL and key.',
        variant: 'destructive',
      });
    } finally {
      setFetching(false);
    }
  }

  // Close dropdown when clicking outside
  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open]);

  return (
    <div ref={containerRef} className="flex items-center gap-2">
      <div className="relative flex-1">
        <Input
          type="text"
          value={value ?? ''}
          onChange={(e) => {
            onChange(e.target.value || null);
            if (models.length > 0) setOpen(true);
          }}
          onFocus={() => { if (models.length > 0) setOpen(true); }}
          placeholder={placeholder ?? 'Enter or select model'}
        />
        {open && models.length > 0 && (
          <div className="absolute z-50 top-full left-0 right-0 mt-1 max-h-60 overflow-y-auto rounded-md border bg-popover shadow-md">
            {models.map((m) => (
              <button
                key={m}
                type="button"
                className="flex w-full items-center px-3 py-1.5 text-sm hover:bg-accent text-left truncate"
                onClick={() => {
                  onChange(m);
                  setOpen(false);
                }}
                title={m}
              >
                {m}
              </button>
            ))}
          </div>
        )}
      </div>
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => {
          if (models.length > 0 && !fetching) {
            setOpen(o => !o);
          } else {
            fetchModels();
          }
        }}
        disabled={!canFetch}
        title={apiBase ? 'Fetch available models from endpoint' : 'Enter API base URL first'}
      >
        <RefreshCw className={`h-3.5 w-3.5 mr-1 ${fetching ? 'animate-spin' : ''}`} />
        Fetch
      </Button>
    </div>
  );
}
