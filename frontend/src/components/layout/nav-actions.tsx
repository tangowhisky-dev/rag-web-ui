"use client";

import { useState } from "react";
import { LogOut, Search } from "lucide-react";
import Link from "next/link";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { ChangePasswordDialog } from "@/components/ui/change-password-dialog";
import { UserName } from "./user-name";
import { useLogout } from "@/lib/hooks";

interface NavActionsProps {
  showPasswordButton?: boolean;
}

/**
 * Shared right-side nav actions (logout button, optional password dialog, theme toggle, user name).
 * Eliminates duplicate implementations across layout files.
 */
export function NavActions({ showPasswordButton = true }: NavActionsProps) {
  const logout = useLogout();
  const [passwordDialogOpen, setPasswordDialogOpen] = useState(false);

  return (
    <>
      <Link
        href="/dashboard/search"
        className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm text-muted-foreground hover:text-primary hover:bg-primary/10 transition-colors"
        title="Search knowledge bases"
      >
        <Search className="h-4 w-4" />
        <span className="hidden sm:inline">Search</span>
      </Link>
      {showPasswordButton && (
        <button
          onClick={() => setPasswordDialogOpen(true)}
          className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm text-muted-foreground hover:text-primary hover:bg-primary/10 transition-colors"
        >
          <span className="hidden sm:inline">Change Password</span>
          <span className="sm:hidden">Password</span>
        </button>
      )}
      <ThemeToggle />
      <button
        onClick={logout}
        className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
      >
        <LogOut className="h-4 w-4" />
        <span className="hidden sm:inline">Sign out</span>
      </button>
      <UserName />
      {showPasswordButton && (
        <ChangePasswordDialog
          open={passwordDialogOpen}
          onOpenChange={setPasswordDialogOpen}
        />
      )}
    </>
  );
}
