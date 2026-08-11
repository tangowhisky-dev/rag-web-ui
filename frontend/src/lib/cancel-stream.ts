/**
 * Cancel an in-progress chat stream by calling the server-side cancel endpoint.
 *
 * Best-effort: returns true on success, false on any failure so the caller
 * can proceed with client-side cleanup regardless.
 */
import { api } from "@/lib/api";

export async function cancelStream(chatId: string): Promise<boolean> {
  try {
    await api.postRaw(`/api/chat/${chatId}/cancel`);
    return true;
  } catch {
    return false;
  }
}
