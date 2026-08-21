"use client";

import { useEffect, useRef } from "react";
import { api } from "@/lib/api";

const HEARTBEAT_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

/**
 * Calls the backend heartbeat endpoint every 5 minutes to keep the
 * session alive while the tab is open. The backend decides whether to
 * renew the token based on whether the user is making real API calls
 * and whether background work is running.
 *
 * If the token expires (user idle, no background work), the heartbeat
 * gets a 401 and the existing handleAuthRedirect in api.ts logs out
 * and redirects to login.
 */
export function useSessionHeartbeat() {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const beat = async () => {
      try {
        await api.get("/api/auth/heartbeat");
      } catch {
        // 401 is handled by handleAuthRedirect in api.ts (logout + redirect).
        // Any other error is non-fatal — the next heartbeat will retry.
      }
    };

    intervalRef.current = setInterval(beat, HEARTBEAT_INTERVAL_MS);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, []);
}
