"""
VOXCODE Skill System
Pre-defined skills for common automation tasks.
"""

from agent.skills.base import Skill, SkillResult, SkillStatus, SkillRegistry, get_registry
from agent.skills.app_skills import OpenAppSkill, CloseAppSkill, FocusAppSkill
from agent.skills.messaging_skills import SendMessageSkill, OpenChatSkill
from agent.skills.browser_skills import NavigateToUrlSkill, SearchWebSkill
from agent.skills.input_skills import TypeTextSkill, ClickElementSkill, ScrollSkill

# DOM Browser Skills (Playwright CDP)
try:
    from agent.skills.dom_browser_skills import (
        DOMReadPageSkill,
        DOMFillSkill,
        DOMClickSkill,
        DOMExtractSkill,
        DOMSearchExtractSkill,
        DOMWaitSkill,
    )
    DOM_SKILLS_AVAILABLE = True
except ImportError:
    DOM_SKILLS_AVAILABLE = False

# System Skills (Windows Native API)
from agent.skills.system_skills import (
    SystemCommandSkill,
    ProcessManagerSkill,
    NetworkInfoSkill,
    BrightnessControlSkill,
    BluetoothControlSkill,
    SystemInfoSkill,
    WindowManagerSkill,
)

__all__ = [
    'Skill',
    'SkillResult',
    'SkillStatus',
    'SkillRegistry',
    'get_registry',
    # App skills
    'OpenAppSkill',
    'CloseAppSkill',
    'FocusAppSkill',
    # Messaging skills
    'SendMessageSkill',
    'OpenChatSkill',
    # Browser skills
    'NavigateToUrlSkill',
    'SearchWebSkill',
    # Input skills
    'TypeTextSkill',
    'ClickElementSkill',
    'ScrollSkill',
    # DOM Browser skills
    'DOMReadPageSkill',
    'DOMFillSkill',
    'DOMClickSkill',
    'DOMExtractSkill',
    'DOMSearchExtractSkill',
    'DOMWaitSkill',
    # System skills
    'SystemCommandSkill',
    'ProcessManagerSkill',
    'NetworkInfoSkill',
    'BrightnessControlSkill',
    'BluetoothControlSkill',
    'SystemInfoSkill',
    'WindowManagerSkill',
]
