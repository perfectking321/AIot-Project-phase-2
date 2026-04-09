"""
Perception OmniParser wrapper.

This keeps the architecture boundary under `perception/` while reusing the
existing OmniParser implementation.
"""

from agent.omniparser import OmniParser, ParsedScreen, UIElement, get_omniparser

__all__ = ["OmniParser", "ParsedScreen", "UIElement", "get_omniparser"]
