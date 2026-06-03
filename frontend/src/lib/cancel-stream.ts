/**
 * Cancel an in-progress chat stream by calling the server-side cancel endpoint.
 *
 * Best-effort: returns true on success, false on any failure so the caller
 * can proceed with client-side cleanup regardless.
 */
export async function cancelStream(chatId: string): Promise<boolean> {
  try {
    const token = typeof window !== "undefined" ? localStorage.getItem("token") || "" : "";
    const res = await fetch(`/api/chat/${chatId}/cancel`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    return res.ok;
  } catch {
    return false;
  }
}
