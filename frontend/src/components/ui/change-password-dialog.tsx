'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { useToast } from '@/components/ui/use-toast';
import { api } from '@/lib/api';
import { validatePasswordStrength } from '@/lib/auth';

interface ChangePasswordDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  username?: string;
}

export function ChangePasswordDialog({ open, onOpenChange, username }: ChangePasswordDialogProps) {
  const router = useRouter();
  const { toast } = useToast();
  const [changing, setChanging] = useState(false);
  const [form, setForm] = useState({
    new_password: '',
    confirm_password: '',
  });

  function resetForm() {
    setForm({ new_password: '', confirm_password: '' });
  }

  async function handleChangePassword() {
    const passwordError = validatePasswordStrength(form.new_password);
    if (passwordError) {
      toast({
        title: 'Error',
        description: passwordError,
        variant: 'destructive',
      });
      return;
    }
    if (form.new_password !== form.confirm_password) {
      toast({
        title: 'Error',
        description: 'Passwords do not match',
        variant: 'destructive',
      });
      return;
    }

    setChanging(true);
    try {
      await api.post('/api/auth/change-password', {
        new_password: form.new_password,
      });
      toast({ title: 'Password changed successfully. Logging you out...' });
      resetForm();
      onOpenChange(false);
      // Clear auth and redirect to login
      localStorage.removeItem('token');
      document.cookie = 'token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';
      router.push('/login');
    } catch (err) {
      toast({
        title: 'Error',
        description: (err as { message?: string }).message ?? 'Failed to change password',
        variant: 'destructive',
      });
    } finally {
      setChanging(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Change Password</DialogTitle>
        </DialogHeader>
        <div className="space-y-3 py-2">
          <div className="space-y-1">
            <Label htmlFor="new-password">New Password *</Label>
            <Input
              id="new-password"
              type="password"
              placeholder="Min 8 chars, 1 letter + 1 number"
              value={form.new_password}
              onChange={(e) => setForm({ ...form, new_password: e.target.value })}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="confirm-password">Confirm Password *</Label>
            <Input
              id="confirm-password"
              type="password"
              placeholder="Re-enter new password"
              value={form.confirm_password}
              onChange={(e) => setForm({ ...form, confirm_password: e.target.value })}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => { resetForm(); onOpenChange(false); }} disabled={changing}>
            Cancel
          </Button>
          <Button
            onClick={handleChangePassword}
            disabled={changing || !!validatePasswordStrength(form.new_password) || form.new_password !== form.confirm_password}
          >
            {changing ? 'Changing...' : 'Change Password'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
