"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { useChatContext } from "@/contexts/chat-context";

function ChatIndexInner() {
  const router = useRouter();
  const { chatList } = useChatContext();
  const navigatedRef = useRef(false);

  // Redirect as soon as chatList is populated. No artificial timeout —
  // if the list is cached (from sessionStorage or module cache), this
  // fires immediately. If the API call is in-flight, it fires when the
  // data arrives. The fallback to /new is handled by the empty-state
  // check below, not by a timer.
  useEffect(() => {
    if (navigatedRef.current) return;
    if (chatList.length > 0) {
      navigatedRef.current = true;
      router.replace(`/dashboard/chat/${chatList[0].id}`);
    }
  }, [chatList, router]);

  // If the API call completes and returns an empty list, redirect to /new.
  // We detect this by checking that chatList is still empty after a
  // reasonable delay — but only if we haven't already navigated.
  // The ChatProvider's useEffect runs on mount, so by the time this
  // effect's second invocation fires (after the API resolves), chatList
  // will either have data or be confirmed empty.
  useEffect(() => {
    if (navigatedRef.current) return;
    // Only set the fallback timer if we don't have cached data at all.
    // If chatList is empty AND the module cache is empty, the API call
    // is in-flight. Give it a generous window before falling back.
    const timer = setTimeout(() => {
      if (!navigatedRef.current) {
        navigatedRef.current = true;
        router.replace("/dashboard/chat/new");
      }
    }, 500);
    return () => clearTimeout(timer);
  }, [router]);

  return (
    <div className="flex items-center justify-center h-full">
      <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
    </div>
  );
}

export default function ChatPage() {
  return <ChatIndexInner />;
}
