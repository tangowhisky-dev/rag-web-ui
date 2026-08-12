'use client';

import { useState } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/use-toast';
import { api } from '@/lib/api';
import { RefreshCw } from 'lucide-react';

interface ModelPickerProps {
  value: string | null;
  apiBase: string | null;
  apiKey: string | null;
  fetchUrl: string;       // e.g. '/api/admin/settings/models' or '/api/admin/orgs/{id}/settings/models'
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

  const canFetch = !!apiBase && !fetching;

  async function fetchModels() {
    if (!apiBase) return;
    setFetching(true);
    try {
      const params = new URLSearchParams({ api_base: apiBase });
      // Only send api_key if it's not masked (starts with ••••)
      if (apiKey && !apiKey.startsWith('••••')) {
        params.set('api_key', apiKey);
      }
      const data = await api.get(`${fetchUrl}?${params.toString()}`) as { models: string[] };
      setModels(data.models || []);
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

  const listId = `model-list-${fetchUrl.replace(/[^a-z0-9]/gi, '')}`;

  return (
    <div className="flex items-center gap-2">
      <Input
        type="text"
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value || null)}
        placeholder={placeholder ?? 'Enter or select model'}
        list={listId}
        className="flex-1"
      />
      <datalist id={listId}>
        {models.map((m) => (
          <option key={m} value={m} />
        ))}
      </datalist>
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={fetchModels}
        disabled={!canFetch}
        title={apiBase ? 'Fetch available models from endpoint' : 'Enter API base URL first'}
      >
        <RefreshCw className={`h-3.5 w-3.5 mr-1 ${fetching ? 'animate-spin' : ''}`} />
        Fetch
      </Button>
    </div>
  );
}
