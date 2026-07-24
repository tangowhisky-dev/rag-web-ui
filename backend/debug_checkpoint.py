"""Dump checkpoint state for chat-58 to diagnose message duplication."""
import asyncio
import json
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

async def main():
    cp = AsyncRedisSaver(redis_url="redis://redis:6379/0")
    await cp.__aenter__()
    await cp.asetup()
    
    thread_id = "chat-58"
    latest = await cp.aget_tuple({"configurable": {"thread_id": thread_id}})
    
    if latest is None:
        print("No checkpoint found for chat-58")
        return
    
    print("=== CHECKPOINT STATE ===")
    print(f"Thread: {thread_id}")
    print(f"Parent checkpoint: {latest.parent_checkpoint}")
    print(f"Metadata: {latest.metadata}")
    
    values = latest.values
    messages = values.get("messages", [])
    print(f"\nTotal messages in state: {len(messages)}")
    
    for i, m in enumerate(messages):
        role = getattr(m, "role", m.get("role", "unknown"))
        content = str(getattr(m, "content", m.get("content", "")))
        msg_id = getattr(m, "id", "N/A")
        preview = content[:120]
        print(f"  [{i}] {role} | id={msg_id[:20]}... | {preview}")
    
    print("\n=== CHECKPOINT HISTORY ===")
    all_tuples = []
    async for t in cp.alist({"configurable": {"thread_id": thread_id}}):
        all_tuples.append(t)
    
    for idx, t in enumerate(all_tuples):
        msgs = t.values.get("messages", [])
        print(f"  Step {idx}: {len(msgs)} messages | parent={t.parent_checkpoint}")

asyncio.run(main())
