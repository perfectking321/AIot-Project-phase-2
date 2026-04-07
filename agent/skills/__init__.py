"""
VOXCODE Skill System
Pre-defined skills for common automation tasks.
"""

from agent.skills.base import Skill, SkillResult, SkillStatus, SkillRegistry, get_registry
from agent.skills.app_skills import OpenAppSkill, CloseAppSkill, FocusAppSkill
from agent.skills.messaging_skills import SendMessageSkill, OpenChatSkill
from agent.skills.browser_skills import NavigateToUrlSkill, SearchWebSkill
from agent.skills.input_skills import TypeTextSkill, ClickElementSkill, ScrollSkill

__all__ = [
    'Skill',
    'SkillResult',
    'SkillStatus',
    'SkillRegistry',
    'get_registry',
    'OpenAppSkill',
    'CloseAppSkill',
    'FocusAppSkill',
    'SendMessageSkill',
    'OpenChatSkill',
    'NavigateToUrlSkill',
    'SearchWebSkill',
    'TypeTextSkill',
    'ClickElementSkill',
    'ScrollSkill',
]
