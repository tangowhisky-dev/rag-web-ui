"""Redis-backed LangGraph memory layer.

Provides a durable short-term memory (thread-level checkpointer) and a
semantic long-term memory (cross-thread store) backed by Redis Stack.

Usage:
    memory = await get_redis_memory()
    checkpointer = memory.checkpointer      # AsyncRedisSaver
    store = memory.store                    # AsyncRedisStore

The same Redis service is used for both responsibilities so the application
adds only one new infrastructure component.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.redis.aio import AsyncRedisStore
from openai import AsyncOpenAI

from app.core.config import settings


def _get_embedding_dim() -> int:
    """Resolve DENSE_EMBEDDING_DIM from app-level settings."""
    from app.services.settings_service import get_setting
    from app.db.session import SessionLocal
    _db = SessionLocal()
    try:
        return get_setting(_db, "DENSE_EMBEDDING_DIM", None) or 1024
    finally:
        _db.close()

logger = logging.getLogger(__name__)


def _run_sync(coro: Any) -> Any:
    """Run a coroutine synchronously, respecting nested async contexts.

    ``asyncio.run()`` cannot be called when an event loop is already running.
    This helper uses the existing loop if one exists, otherwise creates a new
    one via ``asyncio.run()``. A ``concurrent.futures.ThreadPoolExecutor`` is
    used to avoid ``nest_asyncio`` and to keep the call thread-safe.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return loop.run_until_complete(coro)


class _StringEmbeddings(Embeddings):
    """OpenAI-compatible embeddings that always sends raw strings.

    ``OpenAIEmbeddings`` tokenises input and sends token IDs by default, which
    many local OpenAI-compatible servers reject. This wrapper uses the official
    ``openai`` async client and sends plain strings instead.
    """

    def __init__(self, model: str, api_base: str, api_key: str) -> None:
        self.model = model
        self._client = AsyncOpenAI(base_url=api_base, api_key=api_key)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(input=list(texts), model=self.model)
        data = sorted(response.data, key=lambda d: d.index)
        return [list(d.embedding) for d in data]

    async def aembed_query(self, text: str) -> List[float]:
        response = await self._client.embeddings.create(input=[text], model=self.model)
        return list(response.data[0].embedding)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return _run_sync(self.aembed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        return _run_sync(self.aembed_query(text))


_redis_memory: Optional["RedisMemory"] = None
_init_lock = asyncio.Lock()


class RedisMemory:
    """Singleton-like holder for the Redis checkpointer and store.

    Initialisation is async because both LangGraph objects require
    ``.asetup()`` / ``.asetup()`` to create their internal schemas.
    """

    def __init__(self) -> None:
        self._uri = settings.REDIS_URL
        self._checkpointer: Optional[Any] = None
        self._store: Optional[Any] = None
        self._embeddings: Optional[Embeddings] = None
        self._using_redis = False

    async def setup(self) -> None:
        """Lazily create the checkpointer and store.

        ``AsyncRedisSaver`` and ``AsyncRedisStore`` expose ``from_conn_string``
        as async context managers. For a long-lived singleton we enter the
        context once and keep it open for the application lifetime.

        If Redis is disabled, unreachable, or fails to initialise, the service
        transparently falls back to in-memory implementations so the pipeline
        keeps working (without cross-process persistence).
        """
        if not settings.MEMORY_ENABLED:
            logger.info("[MEMORY] Memory persistence disabled; using in-memory fallback.")
            self._checkpointer = MemorySaver()
            self._store = InMemoryStore()
            return

        if self._checkpointer is None:
            logger.info("[MEMORY] Initialising Redis checkpointer | uri=%s", self._uri)
            try:
                cp = AsyncRedisSaver(redis_url=self._uri)
                self._checkpointer = await cp.__aenter__()
                await self._checkpointer.asetup()
                self._using_redis = True
                logger.info("[MEMORY] Redis checkpointer ready")
            except Exception as exc:
                logger.warning(
                    "[MEMORY] Redis checkpointer init failed (%s); using in-memory fallback.",
                    exc,
                )
                self._checkpointer = MemorySaver()

        if self._store is None:
            logger.info("[MEMORY] Initialising Redis store | uri=%s", self._uri)
            # Embeddings API key/base are super_admin-only (app scope).
            from app.services.settings_service import get_setting
            from app.db.session import SessionLocal
            _db = SessionLocal()
            try:
                embedding_model = get_setting(_db, "MEMORY_EMBEDDING_MODEL", None) or get_setting(_db, "DENSE_EMBEDDINGS_MODEL", None)
                api_key = get_setting(_db, "EMBEDDING_API_KEY", None) or get_setting(_db, "OPENAI_API_KEY", None)
                api_base = get_setting(_db, "EMBEDDING_API_BASE", None) or get_setting(_db, "OPENAI_API_BASE", None)
            finally:
                _db.close()
            if not api_key:
                api_key = "not-required"
            self._embeddings = _StringEmbeddings(
                model=embedding_model,
                api_base=api_base,
                api_key=api_key,
            )
            try:
                st = AsyncRedisStore(
                    redis_url=self._uri,
                    index={
                        "embed": self._embeddings,
                        "dims": _get_embedding_dim(),
                    },
                )
                self._store = await st.__aenter__()
                await self._store.setup()
                self._using_redis = True
                logger.info("[MEMORY] Redis semantic store ready | dims=%d", _get_embedding_dim())
            except Exception as exc:
                logger.warning(
                    "[MEMORY] Redis semantic store init failed (%s); using in-memory fallback.",
                    exc,
                )
                self._store = InMemoryStore()

    @property
    def checkpointer(self) -> AsyncRedisSaver:
        if self._checkpointer is None:
            raise RuntimeError("RedisMemory has not been initialised. Call setup() first.")
        return self._checkpointer

    @property
    def store(self) -> AsyncRedisStore:
        if self._store is None:
            raise RuntimeError("RedisMemory has not been initialised. Call setup() first.")
        return self._store

    async def save_turn(
        self,
        query: str,
        answer: str,
        user_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist a user/assistant turn to the long-term memory store."""
        if not settings.MEMORY_ENABLED:
            return
        if not user_id and not chat_id:
            return

        value = {
            "text": f"User: {query}\nAssistant: {answer}",
            "query": query,
            "answer": answer,
            "chat_id": chat_id,
            **(extra or {}),
        }

        namespaces: List[tuple[str, ...]] = []
        if user_id:
            namespaces.append((str(user_id), "memories"))
        if chat_id:
            namespaces.append((str(chat_id), "memories"))

        for ns in namespaces:
            try:
                await self.store.aput(ns, str(uuid.uuid4()), value)
                logger.debug("[MEMORY] saved turn to namespace=%s", ns)
            except Exception as exc:
                logger.warning("[MEMORY] failed to save turn to %s: %s", ns, exc)

    async def search_memory(
        self,
        query: str,
        user_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        limit: int = 5,
    ) -> List[dict[str, Any]]:
        """Search long-term memory for relevant past turns."""
        if not settings.MEMORY_ENABLED:
            return []
        if not user_id and not chat_id:
            return []

        namespaces: List[tuple[str, ...]] = []
        if user_id:
            namespaces.append((str(user_id), "memories"))
        if chat_id:
            namespaces.append((str(chat_id), "memories"))

        results: List[dict[str, Any]] = []
        seen_keys: set[str] = set()

        for ns in namespaces:
            try:
                items = await self.store.asearch(ns, query=query, limit=limit)
            except Exception as exc:
                logger.warning("[MEMORY] search failed for %s: %s", ns, exc)
                continue
            for item in items:
                key = str(item.key)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                value = item.value or {}
                text = value.get("text") or value.get("data") or ""
                if text:
                    results.append(
                        {
                            "page_content": text,
                            "metadata": {
                                "source": "memory",
                                "namespace": ns,
                                "key": key,
                            },
                        }
                    )

        return results[:limit]


def _cleanup_embeddings() -> _StringEmbeddings:
    """Build the same embedding wrapper used during normal setup."""
    from app.services.settings_service import get_setting
    from app.db.session import SessionLocal
    _db = SessionLocal()
    try:
        embedding_model = get_setting(_db, "MEMORY_EMBEDDING_MODEL", None) or get_setting(_db, "DENSE_EMBEDDINGS_MODEL", None)
        api_key = get_setting(_db, "EMBEDDING_API_KEY", None) or get_setting(_db, "OPENAI_API_KEY", None)
        api_base = get_setting(_db, "EMBEDDING_API_BASE", None) or get_setting(_db, "OPENAI_API_BASE", None)
    finally:
        _db.close()
    if not api_key:
        api_key = "not-required"
    return _StringEmbeddings(
        model=embedding_model,
        api_base=api_base,
        api_key=api_key,
    )


async def _cleanup_store(redis_url: str) -> Any:
    """Create and set up a fresh AsyncRedisStore for cleanup operations."""
    st = AsyncRedisStore(
        redis_url=redis_url,
        index={
            "embed": _cleanup_embeddings(),
            "dims": _get_embedding_dim(),
        },
    )
    st = await st.__aenter__()
    await st.setup()
    return st


async def _cleanup_checkpointer(redis_url: str) -> Any:
    """Create and set up a fresh AsyncRedisSaver for cleanup operations."""
    cp = AsyncRedisSaver(redis_url=redis_url)
    cp = await cp.__aenter__()
    await cp.asetup()
    return cp


async def _delete_store_namespace(
    store: Any, namespace: tuple[str, ...], chat_id: Optional[int] = None
) -> None:
    """Best-effort deletion of items under a store namespace.

    If ``chat_id`` is given, only delete entries tagged with that chat_id
    (used to purge a single chat's turns out of a user-scoped namespace
    without wiping the user's other chats). Otherwise delete everything.
    """
    try:
        items = await store.asearch(namespace, limit=10000)
    except Exception as exc:
        logger.warning("[MEMORY] failed to list namespace %s for deletion: %s", namespace, exc)
        return
    for item in items:
        if chat_id is not None and (item.value or {}).get("chat_id") != chat_id:
            continue
        try:
            await store.adelete(namespace, str(item.key))
        except Exception as exc:
            logger.warning("[MEMORY] failed to delete item %s/%s: %s", namespace, item.key, exc)


async def _cleanup_chat_redis(chat_id: int, user_id: Optional[int] = None) -> None:
    """Delete the checkpoint thread and long-term memory namespace for one chat.

    Also purges this chat's turns out of the user-scoped namespace (if
    ``user_id`` is given) so deleting a chat doesn't leave a "ghost" copy of
    its content recallable from the user's other/future chats.

    Uses fresh LangGraph Redis clients so this can be called from a sync
    FastAPI endpoint (which runs in a threadpool) without relying on the
    global singleton's event loop.
    """
    if not settings.MEMORY_ENABLED:
        return

    uri = settings.REDIS_URL
    thread_id = f"chat-{chat_id}"

    try:
        cp = await _cleanup_checkpointer(uri)
        try:
            await cp.adelete_thread(thread_id)
            logger.debug("[MEMORY] deleted checkpoint thread=%s", thread_id)
        finally:
            await cp.__aexit__(None, None, None)
    except Exception as exc:
        logger.warning("[MEMORY] failed to delete checkpoint thread=%s: %s", thread_id, exc)

    try:
        st = await _cleanup_store(uri)
        try:
            await _delete_store_namespace(st, (str(chat_id), "memories"))
            logger.debug("[MEMORY] deleted memory namespace for chat=%s", chat_id)
            if user_id:
                await _delete_store_namespace(st, (str(user_id), "memories"), chat_id=chat_id)
                logger.debug("[MEMORY] purged chat=%s entries from user=%s namespace", chat_id, user_id)
        finally:
            await st.__aexit__(None, None, None)
    except Exception as exc:
        logger.warning("[MEMORY] failed to delete memory namespace for chat=%s: %s", chat_id, exc)


async def _cleanup_user_redis(user_id: int, chat_ids: List[int]) -> None:
    """Delete a user's memory namespace plus every chat memory/checkpoint."""
    if not settings.MEMORY_ENABLED:
        return

    uri = settings.REDIS_URL

    try:
        st = await _cleanup_store(uri)
        try:
            await _delete_store_namespace(st, (str(user_id), "memories"))
            for chat_id in chat_ids:
                await _delete_store_namespace(st, (str(chat_id), "memories"))
        finally:
            await st.__aexit__(None, None, None)
    except Exception as exc:
        logger.warning("[MEMORY] failed to delete memory namespaces for user=%s: %s", user_id, exc)

    try:
        cp = await _cleanup_checkpointer(uri)
        try:
            for chat_id in chat_ids:
                await cp.adelete_thread(f"chat-{chat_id}")
        finally:
            await cp.__aexit__(None, None, None)
    except Exception as exc:
        logger.warning("[MEMORY] failed to delete checkpoint threads for user=%s: %s", user_id, exc)


def delete_chat_redis_sync(chat_id: int, user_id: Optional[int] = None) -> None:
    """Sync entrypoint for chat Redis cleanup from sync FastAPI endpoints."""
    if not settings.MEMORY_ENABLED:
        return
    try:
        asyncio.run(_cleanup_chat_redis(chat_id, user_id=user_id))
    except Exception as exc:
        logger.warning("[MEMORY] sync cleanup failed for chat=%s: %s", chat_id, exc)


def delete_user_redis_sync(user_id: int, chat_ids: List[int]) -> None:
    """Sync entrypoint for user Redis cleanup from sync FastAPI endpoints."""
    if not settings.MEMORY_ENABLED:
        return
    try:
        asyncio.run(_cleanup_user_redis(user_id, chat_ids))
    except Exception as exc:
        logger.warning("[MEMORY] sync cleanup failed for user=%s: %s", user_id, exc)


async def get_redis_memory() -> RedisMemory:
    """Return the global RedisMemory instance, initialising it on first call."""
    global _redis_memory
    if _redis_memory is None:
        async with _init_lock:
            if _redis_memory is None:
                _redis_memory = RedisMemory()
                await _redis_memory.setup()
    return _redis_memory
