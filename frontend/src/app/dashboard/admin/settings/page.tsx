'use client';

import { useState, useEffect, useMemo } from 'react';
import { api } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { useToast } from '@/components/ui/use-toast';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { AlertTriangle, Save, RotateCcw } from 'lucide-react';

interface SettingItem {
  key: string;
  value: any;
  value_type: string;
  category: string;
  label: string;
  scope: string;
  source: string;
  reload: string;
  requires_reindex: boolean;
  description: string;
  min: number | null;
  max: number | null;
  choices: string[] | null;
  secret: boolean;
  is_set: boolean;
}

interface SettingsResponse {
  settings: SettingItem[];
}

export default function SuperAdminSettingsPage() {
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [settings, setSettings] = useState<SettingItem[]>([]);
  const [dirtyKeys, setDirtyKeys] = useState<Set<string>>(new Set());
  const [confirmKey, setConfirmKey] = useState<string | null>(null);

  useEffect(() => { fetchSettings(); }, []);

  async function fetchSettings() {
    try {
      const data = await api.get('/api/admin/settings') as SettingsResponse;
      setSettings(data.settings);
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to load settings',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  }

  function updateValue(key: string, value: any) {
    setSettings(prev => prev.map(s => s.key === key ? { ...s, value } : s));
    setDirtyKeys(prev => new Set(prev).add(key));
  }

  async function saveAll() {
    if (dirtyKeys.size === 0) return;
    setSaving(true);
    const updates = Array.from(dirtyKeys).map(key => {
      const s = settings.find(s => s.key === key);
      return { key, value: s?.value };
    });
    try {
      const result = await api.put('/api/admin/settings', { settings: updates });
      const results = (result as { results: { key: string; status: string; detail?: string }[] }).results;
      const errors = results.filter(r => r.status === 'error');
      const ok = results.filter(r => r.status === 'ok');
      if (errors.length > 0) {
        toast({
          title: `${ok.length} saved, ${errors.length} failed`,
          description: errors.map(e => `${e.key}: ${e.detail}`).join(', '),
          variant: 'destructive',
        });
      } else {
        toast({ title: `Saved ${ok.length} setting${ok.length !== 1 ? 's' : ''}` });
      }
      setDirtyKeys(new Set());
      fetchSettings();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to save settings',
        variant: 'destructive',
      });
    } finally {
      setSaving(false);
    }
  }

  async function resetSetting(key: string) {
    try {
      await api.delete(`/api/admin/settings/${key}`);
      toast({ title: `${key} reset to default` });
      setConfirmKey(null);
      setDirtyKeys(prev => { const n = new Set(prev); n.delete(key); return n; });
      fetchSettings();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to reset setting',
        variant: 'destructive',
      });
    }
  }

  // Group by category
  const categories = useMemo(() => {
    const map = new Map<string, SettingItem[]>();
    for (const s of settings) {
      if (!map.has(s.category)) map.set(s.category, []);
      map.get(s.category)!.push(s);
    }
    return Array.from(map.entries());
  }, [settings]);

  if (loading) {
    return <div className="px-4 sm:px-6 lg:px-8 py-6 pt-16">Loading...</div>;
  }

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 pt-16 overflow-y-auto h-full">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-xl font-semibold">Application Settings</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Defaults for all organisations. Org admins can override the subset marked per-org.
            </p>
          </div>
          <div className="flex items-center gap-2">
            {dirtyKeys.size > 0 && (
              <span className="text-sm text-muted-foreground">
                {dirtyKeys.size} unsaved
              </span>
            )}
            <Button onClick={saveAll} disabled={saving || dirtyKeys.size === 0} size="sm">
              <Save className="h-3.5 w-3.5 mr-1" />
              {saving ? 'Saving...' : 'Save'}
            </Button>
          </div>
        </div>

        {categories.map(([category, items]) => (
          <div key={category} className="mb-8">
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide mb-3">
              {category}
            </h2>
            <div className="space-y-4 rounded-lg border bg-card p-4">
              {items.map(s => (
                <SettingField
                  key={s.key}
                  setting={s}
                  dirty={dirtyKeys.has(s.key)}
                  onChange={(v) => updateValue(s.key, v)}
                  onReset={() => setConfirmKey(s.key)}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      <ConfirmDialog
        open={confirmKey !== null}
        title="Reset to default?"
        description={`This will delete the stored value for ${confirmKey} and revert to the .env / config.py default.`}
        confirmText="Reset"
        onConfirm={() => confirmKey && resetSetting(confirmKey)}
        onCancel={() => setConfirmKey(null)}
      />
    </div>
  );
}

function SettingField({
  setting,
  dirty,
  onChange,
  onReset,
}: {
  setting: SettingItem;
  dirty: boolean;
  onChange: (v: any) => void;
  onReset: () => void;
}) {
  const showWarning = setting.requires_reindex || setting.reload === 'restart' || setting.reload === 'ingest';

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-3">
        <div className="flex-1 min-w-0">
          <Label className="text-sm font-medium flex items-center gap-1.5">
            {setting.label}
            {dirty && <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />}
            {setting.scope === 'org' && (
              <span className="text-[10px] text-muted-foreground border rounded px-1 py-0.5">
                org-overridable
              </span>
            )}
          </Label>
          {setting.description && (
            <p className="text-xs text-muted-foreground mt-0.5">{setting.description}</p>
          )}
          {showWarning && (
            <p className="text-xs text-amber-600 dark:text-amber-500 mt-0.5 flex items-center gap-1">
              <AlertTriangle className="h-3 w-3" />
              {setting.requires_reindex
                ? 'Requires re-indexing / re-ingestion to take full effect.'
                : `Takes effect on ${setting.reload}.`}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {setting.source === 'database' && (
            <button
              onClick={onReset}
              className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1"
              title="Reset to .env default"
            >
              <RotateCcw className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>
      <SettingInput setting={setting} onChange={onChange} />
    </div>
  );
}

function SettingInput({ setting, onChange }: { setting: SettingItem; onChange: (v: any) => void }) {
  const [showSecret, setShowSecret] = useState(false);
  const [secretEditing, setSecretEditing] = useState(false);

  // Secret fields: render password input with show/hide and edit/clear controls.
  if (setting.secret) {
    const isMasked = typeof setting.value === 'string' && setting.value.startsWith('••••');
    if (!secretEditing && isMasked) {
      return (
        <div className="flex items-center gap-2">
          <Input
            type="password"
            value={setting.value ?? ''}
            readOnly
            className="flex-1 font-mono"
          />
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => { setSecretEditing(true); onChange(''); }}
          >
            Replace
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setShowSecret(!showSecret)}
            title={showSecret ? 'Hide' : 'Show'}
          >
            {showSecret ? 'Hide' : 'Show'}
          </Button>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-2">
        <Input
          type={showSecret ? 'text' : 'password'}
          value={String(setting.value ?? '')}
          onChange={(e) => onChange(e.target.value || null)}
          placeholder={setting.is_set ? 'Enter new value' : 'Not set — enter value'}
          className="flex-1 font-mono"
          autoFocus={secretEditing}
        />
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => setShowSecret(!showSecret)}
          title={showSecret ? 'Hide' : 'Show'}
        >
          {showSecret ? 'Hide' : 'Show'}
        </Button>
        {secretEditing && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => { setSecretEditing(false); onChange(setting.value); }}
          >
            Cancel
          </Button>
        )}
      </div>
    );
  }

  if (setting.value_type === 'bool') {
    return (
      <Switch
        checked={!!setting.value}
        onCheckedChange={onChange}
      />
    );
  }

  if (setting.choices && setting.choices.length > 0) {
    return (
      <Select value={String(setting.value ?? '')} onValueChange={onChange}>
        <SelectTrigger className="w-full">
          <SelectValue placeholder="Select..." />
        </SelectTrigger>
        <SelectContent>
          {setting.choices.map(c => (
            <SelectItem key={c} value={c}>{c}</SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  }

  if (setting.value_type === 'text') {
    return (
      <textarea
        value={String(setting.value ?? '')}
        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => onChange(e.target.value)}
        rows={4}
        className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs"
      />
    );
  }

  if (setting.value_type === 'json') {
    const strValue = setting.value
      ? (typeof setting.value === 'string' ? setting.value : JSON.stringify(setting.value, null, 2))
      : '';
    return (
      <textarea
        value={strValue}
        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => {
          try { onChange(JSON.parse(e.target.value)); }
          catch { onChange(e.target.value); }
        }}
        rows={6}
        className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs"
      />
    );
  }

  return (
    <Input
      type={setting.value_type === 'int' || setting.value_type === 'float' ? 'number' : 'text'}
      value={setting.value ?? ''}
      onChange={(e) => {
        if (setting.value_type === 'int') onChange(e.target.value === '' ? null : parseInt(e.target.value, 10));
        else if (setting.value_type === 'float') onChange(e.target.value === '' ? null : parseFloat(e.target.value));
        else onChange(e.target.value || null);
      }}
      min={setting.min ?? undefined}
      max={setting.max ?? undefined}
      placeholder={setting.source === 'install_default' ? '(default)' : ''}
    />
  );
}
