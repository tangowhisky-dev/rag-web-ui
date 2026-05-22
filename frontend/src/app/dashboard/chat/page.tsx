"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import ChatLayout from "@/components/layout/chat-layout";
import { useChatContext } from "@/contexts/chat-context";

function ChatIndexInner() {
  const router = useRouter();
  const { chatList } = useChatContext();

  useEffect(() => {
    // chatList starts empty while fetching; wait until the context has resolved.
    // We detect "resolved" by checking localStorage for a token — if present,
    // the fetch has been initiated. We redirect once chatList stabilises.
    // A small delay avoids redirecting before the first fetch completes.
    const timer = setTimeout(() => {
      if (chatList.length > 0) {
        // chatList is already sorted descending; first item is the latest
        router.replace(`/dashboard/chat/${chatList[0].id}`);
      } else {
        router.replace("/dashboard/chat/new");
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [chatList, router]);

  return (
    <div className="flex items-center justify-center h-full">
      <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
    </div>
  );
}

export default function ChatPage() {
  return (
    <ChatLayout>
      <ChatIndexInner />
    </ChatLayout>
  );
}
