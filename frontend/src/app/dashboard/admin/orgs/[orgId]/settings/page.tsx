'use client';

import { useState, useEffect, useMemo } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
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
import { ArrowLeft, Save, RotateCcw, Layers } from 'lucide-react';

interface OrgSettingItem {
  key: string;
  value: any;
  value_type: string;
  category: string;
  label: string;
  scope: string;
  overridden: boolean;
  app_default: any;
  effective: any;
  reload: string;
  requires_reindex: boolean;
  description: string;
  min: number | null;
  max: number | null;
  choices: string[] | null;
  secret: boolean;
  is_set: boolean;
}

interface OrgSettingsResponse {
  settings: OrgSettingItem[];
}

export default function OrgSettingsPage() {
  const params = useParams();
  const orgId = Number(params.orgId);
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [settings, setSettings] = useState<OrgSettingItem[]>([]);
  const [dirtyKeys, setDirtyKeys] = useState<Set<string>>(new Set());
  const [confirmKey, setConfirmKey] = useState<string | null>(null);
  const [confirmClearAll, setConfirmClearAll] = useState(false);

  useEffect(() => { fetchSettings(); }, [orgId]);

  async function fetchSettings() {
    try {
      const data = await api.get(`/api/admin/orgs/${orgId}/settings`) as OrgSettingsResponse;
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
    setSettings(prev => prev.map(s => s.key === key ? { ...s, value, overridden: true } : s));
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
      const result = await api.put(`/api/admin/orgs/${orgId}/settings`, { settings: updates });
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
      await api.delete(`/api/admin/orgs/${orgId}/settings/${key}`);
      toast({ title: `${key} reset to app default` });
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

  async function clearAllOverrides() {
    try {
      await api.delete(`/api/admin/orgs/${orgId}/settings`);
      toast({ title: 'All overrides cleared' });
      setConfirmClearAll(false);
      setDirtyKeys(new Set());
      fetchSettings();
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to clear overrides',
        variant: 'destructive',
      });
    }
  }

  const categories = useMemo(() => {
    const map = new Map<string, OrgSettingItem[]>();
    for (const s of settings) {
      if (!map.has(s.category)) map.set(s.category, []);
      map.get(s.category)!.push(s);
    }
    return Array.from(map.entries());
  }, [settings]);

  const overrideCount = settings.filter(s => s.overridden).length;

  if (loading) {
    return <div className="px-4 sm:px-6 lg:px-8 py-6 pt-16">Loading...</div>;
  }

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 pt-16 overflow-y-auto h-full">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-center gap-2 mb-4">
          <Link href="/dashboard/admin/orgs" className="text-muted-foreground hover:text-foreground">
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <h1 className="text-xl font-semibold">Organisation Settings</h1>
        </div>
        <p className="text-sm text-muted-foreground mb-4">
          Override application defaults for this organisation. Leave fields unchanged to inherit.
          {overrideCount > 0 && (
            <span className="ml-2 text-amber-600 dark:text-amber-500">
              {overrideCount} override{overrideCount !== 1 ? 's' : ''} active
            </span>
          )}
        </p>

        <div className="flex items-center justify-between mb-6">
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
          {overrideCount > 0 && (
            <Button
              onClick={() => setConfirmClearAll(true)}
              variant="outline"
              size="sm"
            >
              <Layers className="h-3.5 w-3.5 mr-1" />
              Clear all overrides
            </Button>
          )}
        </div>

        {categories.map(([category, items]) => (
          <div key={category} className="mb-8">
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide mb-3">
              {category}
            </h2>
            <div className="space-y-4 rounded-lg border bg-card p-4">
              {items.map(s => (
                <OrgSettingField
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
        title="Reset to app default?"
        description={`This will remove the override for ${confirmKey} and inherit the application-level default.`}
        confirmText="Reset"
        onConfirm={() => confirmKey && resetSetting(confirmKey)}
        onCancel={() => setConfirmKey(null)}
      />

      <ConfirmDialog
        open={confirmClearAll}
        title="Clear all overrides?"
        description="This will remove all organisation-level overrides and revert every setting to the application default. This cannot be undone."
        confirmText="Clear all"
        destructive
        onConfirm={clearAllOverrides}
        onCancel={() => setConfirmClearAll(false)}
      />
    </div>
  );
}

function OrgSettingField({
  setting,
  dirty,
  onChange,
  onReset,
}: {
  setting: OrgSettingItem;
  dirty: boolean;
  onChange: (v: any) => void;
  onReset: () => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-3">
        <div className="flex-1 min-w-0">
          <Label className="text-sm font-medium flex items-center gap-1.5">
            {setting.label}
            {dirty && <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />}
            {setting.overridden && !dirty && (
              <span className="text-[10px] text-amber-600 dark:text-amber-500 border border-amber-300 dark:border-amber-700 rounded px-1 py-0.5">
                overridden
              </span>
            )}
          </Label>
          {setting.description && (
            <p className="text-xs text-muted-foreground mt-0.5">{setting.description}</p>
          )}
          {!setting.overridden && (
            <p className="text-xs text-muted-foreground mt-0.5">
              App default: <code>{String(setting.app_default ?? '(unset)')}</code>
            </p>
          )}
        </div>
        {setting.overridden && (
          <button
            onClick={onReset}
            className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 shrink-0"
            title="Inherit app default"
          >
            <RotateCcw className="h-3 w-3" />
            Inherit
          </button>
        )}
      </div>
      <OrgSettingInput setting={setting} onChange={onChange} />
    </div>
  );
}

function OrgSettingInput({ setting, onChange }: { setting: OrgSettingItem; onChange: (v: any) => void }) {
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
          placeholder={setting.is_set ? 'Enter new value' : 'Not set — inherits app default'}
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
      placeholder={setting.overridden ? '' : 'Inherits app default'}
    />
  );
}
