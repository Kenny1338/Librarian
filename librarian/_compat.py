"""Compatibility shim for the Hermes MemoryProvider ABC.

When installed alongside hermes-agent the real ABC is used.
Standalone, a minimal replica is provided so the library has no hard
dependency on hermes-agent.
"""

from __future__ import annotations

try:
    from agent.memory_provider import MemoryProvider  # type: ignore[import-untyped]
except ImportError:
    from abc import ABC, abstractmethod
    from typing import Any, Dict, List

    class MemoryProvider(ABC):  # type: ignore[no-redef]
        @property
        @abstractmethod
        def name(self) -> str: ...

        @abstractmethod
        def is_available(self) -> bool: ...

        @abstractmethod
        def initialize(self, session_id: str, **kwargs) -> None: ...

        def system_prompt_block(self) -> str:
            return ""

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            return ""

        def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
            pass

        def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
            pass

        @abstractmethod
        def get_tool_schemas(self) -> List[Dict[str, Any]]: ...

        def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
            raise NotImplementedError

        def shutdown(self) -> None:
            pass

        def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
            pass

        def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
            pass

        def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
            return ""

        def get_config_schema(self) -> List[Dict[str, Any]]:
            return []

        def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
            pass
