"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";

export function useHydrated() {
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => setHydrated(true), []);
  return hydrated;
}

export function useLogout() {
  const router = useRouter();
  return useCallback(async () => {
    await api.post("/api/auth/logout");
    router.push("/");
  }, [router]);
}
