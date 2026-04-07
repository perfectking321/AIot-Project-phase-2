"""
VOXCODE Messaging Skills
Skills for sending messages in chat applications.
"""

import time
import logging
from typing import Optional

from agent.skills.base import Skill, SkillResult, SkillStatus

logger = logging.getLogger("voxcode.skills.messaging")


class OpenChatSkill(Skill):
    """Skill to open a chat with a specific contact."""

    name = "open_chat"
    description = "Open a chat with a specific contact in a messaging app"
    params = ["contact_name", "app_name"]
    preconditions = ["messaging_app_open"]
    postconditions = ["chat_open"]

    def execute(
        self,
        contact_name: str = None,
        app_name: str = "WhatsApp",
        **kwargs
    ) -> SkillResult:
        """
        Open a chat with a contact.

        Args:
            contact_name: Name of the contact
            app_name: Messaging app (WhatsApp, Telegram, etc.)

        Returns:
            SkillResult
        """
        if not contact_name:
            return SkillResult(
                status=SkillStatus.FAILED,
                message="contact_name is required"
            )

        tools = self._get_tools()

        # Strategy 1: Try to click on the contact name in chat list
        logger.info(f"Looking for contact: {contact_name}")

        result = tools.click_text(contact_name)
        if result.success:
            time.sleep(1.0)
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Opened chat with {contact_name}",
                data={"contact": contact_name, "app": app_name}
            )

        # Strategy 2: Use search
        logger.info(f"Contact not visible, trying search...")

        # Click search/find area (common patterns)
        search_labels = ["Search", "Search or start new chat", "Search chats", "Find"]
        for label in search_labels:
            result = tools.click_text(label)
            if result.success:
                time.sleep(0.5)
                break

        # Type contact name
        tools.type_text(contact_name)
        time.sleep(1.0)

        # Click on search result
        result = tools.click_text(contact_name)
        if result.success:
            time.sleep(0.5)
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Found and opened chat with {contact_name}",
                data={"contact": contact_name, "app": app_name, "method": "search"}
            )

        return SkillResult(
            status=SkillStatus.FAILED,
            message=f"Could not find contact: {contact_name}"
        )


class SendMessageSkill(Skill):
    """Skill to send a message in a chat."""

    name = "send_message"
    description = "Send a message in the current chat"
    params = ["message", "recipient"]
    preconditions = ["chat_open", "message_input_visible"]
    postconditions = ["message_sent"]

    def execute(
        self,
        message: str = None,
        recipient: str = None,
        **kwargs
    ) -> SkillResult:
        """
        Send a message.

        Args:
            message: The message to send
            recipient: (Optional) Open chat with this person first

        Returns:
            SkillResult
        """
        if not message:
            return SkillResult(
                status=SkillStatus.FAILED,
                message="message is required"
            )

        tools = self._get_tools()

        # If recipient specified, open chat first
        if recipient:
            open_chat = OpenChatSkill(tools=self._tools, perception=self._perception)
            result = open_chat.execute(contact_name=recipient)
            if not result.success:
                return result
            time.sleep(0.5)

        # Find and click message input
        input_labels = [
            "Type a message",
            "Message",
            "Write a message",
            "Type here",
            "Enter message"
        ]

        clicked = False
        for label in input_labels:
            result = tools.click_text(label)
            if result.success:
                clicked = True
                time.sleep(0.3)
                break

        if not clicked:
            # Try clicking at bottom of screen (common message input location)
            try:
                import pyautogui
                screen_width, screen_height = pyautogui.size()
                # Message input is usually at bottom center
                tools.click(screen_width // 2, screen_height - 100)
                time.sleep(0.3)
            except:
                pass

        # Type the message
        result = tools.type_text(message)
        if not result.success:
            return SkillResult(
                status=SkillStatus.FAILED,
                message=f"Failed to type message: {result.message}"
            )

        time.sleep(0.2)

        # Send with Enter key
        result = tools.press_key("enter")

        if result.success:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Sent message: {message[:30]}...",
                data={"message": message, "recipient": recipient}
            )

        # Fallback: try clicking send button
        send_labels = ["Send", "➤", "→"]
        for label in send_labels:
            result = tools.click_text(label)
            if result.success:
                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    message=f"Sent message (via button): {message[:30]}...",
                    data={"message": message, "recipient": recipient}
                )

        return SkillResult(
            status=SkillStatus.PARTIAL,
            message="Message typed but may not have been sent"
        )


class ReplyToMessageSkill(Skill):
    """Skill to reply to the last message."""

    name = "reply_message"
    description = "Reply to the last message in a chat"
    params = ["reply_text"]
    preconditions = ["chat_open"]
    postconditions = ["reply_sent"]

    def execute(self, reply_text: str = None, **kwargs) -> SkillResult:
        """
        Reply to the last message.

        Args:
            reply_text: The reply text

        Returns:
            SkillResult
        """
        # This is essentially the same as send_message for most apps
        send_skill = SendMessageSkill(tools=self._tools, perception=self._perception)
        return send_skill.execute(message=reply_text)
