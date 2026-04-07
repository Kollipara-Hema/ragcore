"""
Long-term memory — Redis-backed persistent store for cross-session facts.

Stores key → value pairs with an optional TTL so the agent can remember
user preferences, document summaries, or resolved entity lookups across
multiple conversations.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL = 60 * 60 * 24 * 7  # 1 week


class LongTermMemory:
    """
    Persistent key-value memory backed by Redis.

    Falls back to a local dict when Redis is unavailable so the agent
    continues to function in development environments without Redis.

    Usage:
        mem = LongTermMemory(prefix="user:alice")
        await mem.set("preferred_language", "English")
        lang = await mem.get("preferred_language")
    """

    def __init__(self, prefix: str = "agent_memory", ttl: int = DEFAULT_TTL):
        self._prefix = prefix
        self._ttl = ttl
        self._fallback: dict[str, Any] = {}
        self._redis = None

    async def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis
            from config.settings import settings
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
            await self._redis.ping()
            return self._redis
        except Exception as exc:
            logger.debug("LongTermMemory: Redis unavailable (%s) — using local dict", exc)
            return None

    def _key(self, name: str) -> str:
        return f"{self._prefix}:{name}"

    async def set(self, name: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value. Value is JSON-serialized."""
        serialized = json.dumps(value)
        redis = await self._get_redis()
        if redis:
            try:
                await redis.set(self._key(name), serialized, ex=ttl or self._ttl)
                return
            except Exception as exc:
                logger.warning("LongTermMemory.set failed in Redis: %s", exc)
        self._fallback[self._key(name)] = value

    async def get(self, name: str, default: Any = None) -> Any:
        """Retrieve a value by name."""
        redis = await self._get_redis()
        if redis:
            try:
                raw = await redis.get(self._key(name))
                if raw is not None:
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("LongTermMemory.get failed in Redis: %s", exc)
        return self._fallback.get(self._key(name), default)

    async def delete(self, name: str) -> None:
        """Remove a stored value."""
        redis = await self._get_redis()
        if redis:
            try:
                await redis.delete(self._key(name))
            except Exception as exc:
                logger.warning("LongTermMemory.delete failed in Redis: %s", exc)
        self._fallback.pop(self._key(name), None)

    async def exists(self, name: str) -> bool:
        """Return True if the key exists."""
        return (await self.get(name)) is not None
