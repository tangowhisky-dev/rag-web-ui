"use client";

import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";

export function useLogout() {
  const router = useRouter();
  return useCallback(() => {
    localStorage.removeItem("token");
    document.cookie = "token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    router.push("/");
  }, [router]);
}
