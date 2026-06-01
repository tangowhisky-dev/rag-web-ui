import ChatLayout from "@/components/layout/chat-layout";

export default function ChatSegmentLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <ChatLayout>{children}</ChatLayout>;
}
