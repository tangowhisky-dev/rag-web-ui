"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { useChatContext } from "@/contexts/chat-context";

function ChatIndexInner() {
  const router = useRouter();
  const { chatList } = useChatContext();

  useEffect(() => {
    const timer = setTimeout(() => {
      if (chatList.length > 0) {
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
  return <ChatIndexInner />;
}
