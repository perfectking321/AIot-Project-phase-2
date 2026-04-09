"""
VOXCODE API Registry
Metadata registry used by the planner for API-aware decomposition.

Covers three categories:
  1. web   - remote services reachable via URL (browser / HTTP)
  2. system - local Windows OS controls (brightness, volume, power ...)
  3. app    - installed desktop applications launchable by name / path

Python libraries needed for system category:
  pip install screen-brightness-control pycaw pywin32 pywinauto wmi psutil
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

APIKind = Literal["web", "system", "app"]


@dataclass(frozen=True)
class APIInfo:
    """Metadata for one API / service / application."""

    api_id: str
    name: str
    kind: APIKind  # "web" | "system" | "app"
    base_url: str = ""  # web only
    endpoints: List[str] = field(default_factory=list)
    requires_auth: bool = False
    keywords: List[str] = field(default_factory=list)

    # system / app extras
    python_module: str = ""  # e.g. "screen_brightness_control"
    python_calls: List[str] = field(default_factory=list)  # e.g. ["set_brightness(50)"]
    shell_cmd: str = ""  # fallback PowerShell / cmd snippet
    exe_name: str = ""  # app executable, e.g. "notepad.exe"

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Compact dictionary used in planner prompts."""
        d: Dict[str, Any] = {
            "id": self.api_id,
            "name": self.name,
            "kind": self.kind,
            "requires_auth": self.requires_auth,
        }
        if self.keywords:
            d["keywords"] = self.keywords
        if self.base_url:
            d["base_url"] = self.base_url
        if self.endpoints:
            d["endpoints"] = self.endpoints
        if self.python_module:
            d["python_module"] = self.python_module
        if self.python_calls:
            d["python_calls"] = self.python_calls
        if self.shell_cmd:
            d["shell_cmd"] = self.shell_cmd
        if self.exe_name:
            d["exe_name"] = self.exe_name
        return d


# ---------------------------------------------------------------------------
# Default registry entries
# ---------------------------------------------------------------------------

_DEFAULT_APIS: List[APIInfo] = [
    # -- WEB -----------------------------------------------------------------
    APIInfo(
        api_id="google_search",
        name="Google Search",
        kind="web",
        base_url="https://google.com",
        endpoints=["search"],
        keywords=["google", "search", "query", "look up", "find online"],
    ),
    APIInfo(
        api_id="youtube",
        name="YouTube",
        kind="web",
        base_url="https://youtube.com",
        endpoints=["search", "watch", "subscribe", "play", "shorts", "channel"],
        keywords=["youtube", "yt", "video", "shorts", "channel", "watch", "stream"],
    ),
    APIInfo(
        api_id="gmail",
        name="Gmail",
        kind="web",
        base_url="https://mail.google.com",
        endpoints=["send", "read", "archive", "delete", "compose"],
        requires_auth=True,
        keywords=["gmail", "mail", "email", "inbox", "compose", "send email"],
    ),
    APIInfo(
        api_id="google_drive",
        name="Google Drive",
        kind="web",
        base_url="https://drive.google.com",
        endpoints=["upload", "download", "share", "open"],
        requires_auth=True,
        keywords=["drive", "google drive", "gdrive", "upload", "cloud storage", "file share"],
    ),
    APIInfo(
        api_id="google_docs",
        name="Google Docs",
        kind="web",
        base_url="https://docs.google.com",
        endpoints=["create", "open", "edit"],
        requires_auth=True,
        keywords=["google docs", "gdocs", "document", "write doc"],
    ),
    APIInfo(
        api_id="google_sheets",
        name="Google Sheets",
        kind="web",
        base_url="https://sheets.google.com",
        endpoints=["create", "open", "edit"],
        requires_auth=True,
        keywords=["google sheets", "spreadsheet", "gsheets", "excel online"],
    ),
    APIInfo(
        api_id="google_maps",
        name="Google Maps",
        kind="web",
        base_url="https://maps.google.com",
        endpoints=["search", "directions", "navigate"],
        keywords=["maps", "google maps", "directions", "navigate", "location", "route"],
    ),
    APIInfo(
        api_id="whatsapp",
        name="WhatsApp Web/Desktop",
        kind="web",
        base_url="https://web.whatsapp.com",
        endpoints=["chat", "send", "search"],
        requires_auth=True,
        keywords=["whatsapp", "wa", "chat", "message", "send message", "whatsapp web"],
    ),
    APIInfo(
        api_id="twitter",
        name="Twitter / X",
        kind="web",
        base_url="https://x.com",
        endpoints=["tweet", "search", "profile", "timeline"],
        requires_auth=True,
        keywords=["twitter", "tweet", "x.com", "post tweet", "timeline"],
    ),
    APIInfo(
        api_id="github",
        name="GitHub",
        kind="web",
        base_url="https://github.com",
        endpoints=["repo", "issues", "pull_request", "profile"],
        requires_auth=True,
        keywords=["github", "git", "repo", "repository", "pull request", "issue", "commit"],
    ),
    APIInfo(
        api_id="stackoverflow",
        name="Stack Overflow",
        kind="web",
        base_url="https://stackoverflow.com",
        endpoints=["search", "question"],
        keywords=["stackoverflow", "stack overflow", "so", "coding question", "error search"],
    ),
    APIInfo(
        api_id="wikipedia",
        name="Wikipedia",
        kind="web",
        base_url="https://wikipedia.org",
        endpoints=["search", "article"],
        keywords=["wikipedia", "wiki", "article", "encyclopedia"],
    ),
    APIInfo(
        api_id="reddit",
        name="Reddit",
        kind="web",
        base_url="https://reddit.com",
        endpoints=["search", "post", "subreddit"],
        keywords=["reddit", "subreddit", "post", "upvote"],
    ),
    APIInfo(
        api_id="spotify_web",
        name="Spotify Web",
        kind="web",
        base_url="https://open.spotify.com",
        endpoints=["play", "search", "playlist"],
        requires_auth=True,
        keywords=["spotify", "music", "song", "playlist", "artist", "album", "play music"],
    ),
    APIInfo(
        api_id="chatgpt",
        name="ChatGPT",
        kind="web",
        base_url="https://chat.openai.com",
        endpoints=["chat", "new_conversation"],
        requires_auth=True,
        keywords=["chatgpt", "openai", "gpt", "ai chat"],
    ),

    # -- SYSTEM: display ------------------------------------------------------
    APIInfo(
        api_id="sys_brightness",
        name="Screen Brightness Control",
        kind="system",
        python_module="screen_brightness_control",
        python_calls=[
            "screen_brightness_control.set_brightness(50)",
            "screen_brightness_control.get_brightness()",
            "screen_brightness_control.fade_brightness(75, interval=0.05)",
        ],
        shell_cmd=(
            "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods)"
            ".WmiSetBrightness(1, 50)"
        ),
        keywords=[
            "brightness",
            "screen brightness",
            "dim",
            "brighten",
            "display brightness",
            "increase brightness",
            "decrease brightness",
            "lower brightness",
            "raise brightness",
        ],
    ),
    APIInfo(
        api_id="sys_display",
        name="Display / Monitor Settings",
        kind="system",
        python_module="win32api",
        python_calls=[
            "win32api.ChangeDisplaySettings(devmode, 0)",
            "ctypes.windll.user32.SetSystemMetrics()",
        ],
        shell_cmd="Get-WmiObject -Class Win32_VideoController",
        keywords=[
            "resolution",
            "display settings",
            "monitor",
            "refresh rate",
            "screen resolution",
            "change resolution",
            "second screen",
            "extend display",
            "duplicate screen",
        ],
    ),

    # -- SYSTEM: audio --------------------------------------------------------
    APIInfo(
        api_id="sys_volume",
        name="System Volume Control",
        kind="system",
        python_module="pycaw",
        python_calls=[
            "AudioUtilities.GetSpeakers().EndpointVolume.SetMasterVolumeLevelScalar(0.5, None)",
            "AudioUtilities.GetSpeakers().EndpointVolume.SetMute(True, None)",
        ],
        shell_cmd=(
            "$wsh = New-Object -ComObject WScript.Shell; "
            "$wsh.SendKeys([char]175)"
        ),
        keywords=[
            "volume",
            "sound",
            "audio",
            "mute",
            "unmute",
            "volume up",
            "volume down",
            "increase volume",
            "decrease volume",
            "louder",
            "quieter",
            "silence",
        ],
    ),

    # -- SYSTEM: power management ---------------------------------------------
    APIInfo(
        api_id="sys_power",
        name="Power Management",
        kind="system",
        python_module="ctypes",
        python_calls=[
            "os.system('shutdown /s /t 60')",
            "os.system('shutdown /r /t 0')",
            "os.system('rundll32.exe powrprof.dll,SetSuspendState 0,1,0')",
            "ctypes.windll.PowrProf.SetSuspendState(0, 1, 0)",
        ],
        shell_cmd="Stop-Computer / Restart-Computer / SetSuspendState",
        keywords=[
            "shutdown",
            "restart",
            "reboot",
            "sleep",
            "hibernate",
            "power off",
            "turn off",
            "log off",
            "sign out",
            "lock screen",
            "lock pc",
        ],
    ),
    APIInfo(
        api_id="sys_lock",
        name="Lock Screen",
        kind="system",
        python_calls=["ctypes.windll.user32.LockWorkStation()"],
        shell_cmd="rundll32.exe user32.dll,LockWorkStation",
        keywords=["lock", "lock screen", "lock pc", "lock computer", "screen lock"],
    ),

    # -- SYSTEM: processes & tasks --------------------------------------------
    APIInfo(
        api_id="sys_processes",
        name="Process Manager",
        kind="system",
        python_module="psutil",
        python_calls=[
            "psutil.process_iter(['pid','name','cpu_percent'])",
            "psutil.Process(pid).kill()",
            "subprocess.Popen(['notepad.exe'])",
        ],
        shell_cmd="Get-Process | Where-Object {$_.CPU -gt 50}",
        keywords=[
            "process",
            "task",
            "task manager",
            "running apps",
            "kill process",
            "kill task",
            "cpu usage",
            "memory usage",
            "running programs",
            "end task",
        ],
    ),

    # -- SYSTEM: network ------------------------------------------------------
    APIInfo(
        api_id="sys_network",
        name="Network / WiFi Control",
        kind="system",
        python_module="subprocess",
        python_calls=[
            "subprocess.run(['netsh', 'wlan', 'show', 'profiles'])",
            "subprocess.run(['netsh', 'wlan', 'connect', 'name=MyWiFi'])",
            "subprocess.run(['netsh', 'interface', 'set', 'interface', 'Wi-Fi', 'enabled'])",
        ],
        shell_cmd="Get-NetAdapter; Enable-NetAdapter -Name 'Wi-Fi'; Disable-NetAdapter -Name 'Wi-Fi'",
        keywords=[
            "wifi",
            "wi-fi",
            "network",
            "internet",
            "connect wifi",
            "disconnect wifi",
            "airplane mode",
            "network settings",
            "ip address",
            "ethernet",
            "network adapter",
        ],
    ),

    # -- SYSTEM: clipboard & keyboard -----------------------------------------
    APIInfo(
        api_id="sys_clipboard",
        name="Clipboard",
        kind="system",
        python_module="pyperclip",
        python_calls=[
            "pyperclip.copy('text')",
            "pyperclip.paste()",
        ],
        shell_cmd="Get-Clipboard; Set-Clipboard -Value 'text'",
        keywords=["clipboard", "copy", "paste", "copy text", "paste text"],
    ),
    APIInfo(
        api_id="sys_keyboard",
        name="Keyboard / Hotkeys",
        kind="system",
        python_module="pyautogui",
        python_calls=[
            "pyautogui.hotkey('ctrl', 'c')",
            "pyautogui.press('volumeup')",
            "pyautogui.keyDown('alt'); pyautogui.press('tab'); pyautogui.keyUp('alt')",
        ],
        shell_cmd="$wsh.SendKeys('+{TAB}')",
        keywords=[
            "hotkey",
            "keyboard shortcut",
            "alt tab",
            "press key",
            "type text",
            "send keys",
            "keyboard input",
            "shortcut",
        ],
    ),

    # -- SYSTEM: file system ---------------------------------------------------
    APIInfo(
        api_id="sys_filesystem",
        name="File System",
        kind="system",
        python_module="pathlib",
        python_calls=[
            "pathlib.Path(path).mkdir(parents=True)",
            "shutil.copy(src, dst)",
            "os.remove(path)",
        ],
        shell_cmd="Copy-Item; Move-Item; Remove-Item; New-Item",
        keywords=[
            "file",
            "folder",
            "directory",
            "copy file",
            "move file",
            "delete file",
            "rename file",
            "create folder",
            "open folder",
            "file explorer",
        ],
    ),

    # -- SYSTEM: windows registry ---------------------------------------------
    APIInfo(
        api_id="sys_registry",
        name="Windows Registry",
        kind="system",
        python_module="winreg",
        python_calls=[
            "winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\\\\...')",
            "winreg.SetValueEx(key, 'name', 0, winreg.REG_SZ, 'value')",
            "winreg.QueryValueEx(key, 'name')",
        ],
        shell_cmd="Get-ItemProperty; Set-ItemProperty; New-ItemProperty",
        keywords=["registry", "regedit", "windows registry", "registry key", "reg edit"],
    ),

    # -- SYSTEM: system info ---------------------------------------------------
    APIInfo(
        api_id="sys_info",
        name="System Information",
        kind="system",
        python_module="psutil",
        python_calls=[
            "psutil.cpu_percent(interval=1)",
            "psutil.virtual_memory()",
            "psutil.disk_usage('/')",
            "psutil.battery()",
        ],
        shell_cmd="Get-ComputerInfo; systeminfo",
        keywords=[
            "cpu",
            "ram",
            "memory",
            "disk",
            "battery",
            "system info",
            "hardware info",
            "storage",
            "performance",
            "temperature",
            "system specs",
        ],
    ),

    # -- SYSTEM: notifications -------------------------------------------------
    APIInfo(
        api_id="sys_notification",
        name="Windows Notifications / Toast",
        kind="system",
        python_module="win10toast",
        python_calls=[
            "ToastNotifier().show_toast('Title', 'Message', duration=5)",
        ],
        shell_cmd=(
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.MessageBox]::Show('Hello')"
        ),
        keywords=[
            "notification",
            "toast",
            "alert",
            "popup",
            "notify",
            "reminder",
            "message box",
        ],
    ),

    # -- SYSTEM: window management --------------------------------------------
    APIInfo(
        api_id="sys_windows_mgr",
        name="Window Manager",
        kind="system",
        python_module="pygetwindow",
        python_calls=[
            "pygetwindow.getWindowsWithTitle('Chrome')[0].maximize()",
            "pygetwindow.getActiveWindow().minimize()",
            "pygetwindow.getAllTitles()",
        ],
        shell_cmd="(Get-Process notepad).MainWindowHandle",
        keywords=[
            "window",
            "maximize",
            "minimize",
            "close window",
            "switch window",
            "bring to front",
            "hide window",
            "restore window",
            "snap window",
        ],
    ),

    # -- SYSTEM: screenshots ---------------------------------------------------
    APIInfo(
        api_id="sys_screenshot",
        name="Screenshot / Screen Capture",
        kind="system",
        python_module="pyautogui",
        python_calls=[
            "pyautogui.screenshot('screen.png')",
            "PIL.ImageGrab.grab()",
        ],
        shell_cmd="Add-Type -AssemblyName System.Drawing",
        keywords=[
            "screenshot",
            "screen capture",
            "capture screen",
            "snip",
            "snipping tool",
            "print screen",
            "record screen",
        ],
    ),

    # -- APPS -----------------------------------------------------------------
    APIInfo(
        api_id="app_notepad",
        name="Notepad",
        kind="app",
        exe_name="notepad.exe",
        python_calls=["subprocess.Popen(['notepad.exe', 'file.txt'])"],
        keywords=["notepad", "text editor", "note", "open notepad"],
    ),
    APIInfo(
        api_id="app_calculator",
        name="Calculator",
        kind="app",
        exe_name="calc.exe",
        python_calls=["subprocess.Popen(['calc.exe'])"],
        keywords=["calculator", "calc", "calculate", "math"],
    ),
    APIInfo(
        api_id="app_file_explorer",
        name="File Explorer",
        kind="app",
        exe_name="explorer.exe",
        python_calls=["subprocess.Popen(['explorer.exe', 'C:\\\\Users'])"],
        keywords=["file explorer", "explorer", "my computer", "open folder", "browse files"],
    ),
    APIInfo(
        api_id="app_task_manager",
        name="Task Manager",
        kind="app",
        exe_name="taskmgr.exe",
        python_calls=["subprocess.Popen(['taskmgr.exe'])"],
        keywords=["task manager", "taskmgr", "processes", "performance monitor"],
    ),
    APIInfo(
        api_id="app_settings",
        name="Windows Settings",
        kind="app",
        exe_name="ms-settings:",
        python_calls=["os.system('start ms-settings:')"],
        shell_cmd="Start-Process ms-settings:",
        keywords=[
            "settings",
            "windows settings",
            "control panel",
            "system settings",
            "display settings",
            "sound settings",
            "network settings",
            "privacy settings",
        ],
    ),
    APIInfo(
        api_id="app_cmd",
        name="Command Prompt",
        kind="app",
        exe_name="cmd.exe",
        python_calls=["subprocess.Popen(['cmd.exe'])"],
        keywords=["cmd", "command prompt", "terminal", "command line", "console"],
    ),
    APIInfo(
        api_id="app_powershell",
        name="PowerShell",
        kind="app",
        exe_name="powershell.exe",
        python_calls=["subprocess.Popen(['powershell.exe'])"],
        keywords=["powershell", "ps", "shell", "powershell terminal"],
    ),
    APIInfo(
        api_id="app_chrome",
        name="Google Chrome",
        kind="app",
        exe_name="chrome.exe",
        python_calls=["subprocess.Popen(['chrome.exe', 'https://google.com'])"],
        keywords=["chrome", "google chrome", "browser", "open chrome", "web browser"],
    ),
    APIInfo(
        api_id="app_edge",
        name="Microsoft Edge",
        kind="app",
        exe_name="msedge.exe",
        python_calls=["subprocess.Popen(['msedge.exe'])"],
        keywords=["edge", "microsoft edge", "ie", "open edge"],
    ),
    APIInfo(
        api_id="app_vscode",
        name="Visual Studio Code",
        kind="app",
        exe_name="code.exe",
        python_calls=["subprocess.Popen(['code', '.'])"],
        keywords=["vscode", "vs code", "visual studio code", "code editor", "open vscode"],
    ),
    APIInfo(
        api_id="app_word",
        name="Microsoft Word",
        kind="app",
        exe_name="winword.exe",
        python_calls=["subprocess.Popen(['winword.exe'])"],
        keywords=["word", "microsoft word", "docx", "document editor", "open word"],
    ),
    APIInfo(
        api_id="app_excel",
        name="Microsoft Excel",
        kind="app",
        exe_name="excel.exe",
        python_calls=["subprocess.Popen(['excel.exe'])"],
        keywords=["excel", "microsoft excel", "spreadsheet", "xlsx", "open excel"],
    ),
    APIInfo(
        api_id="app_powerpoint",
        name="Microsoft PowerPoint",
        kind="app",
        exe_name="powerpnt.exe",
        python_calls=["subprocess.Popen(['powerpnt.exe'])"],
        keywords=["powerpoint", "ppt", "presentation", "slides", "open powerpoint"],
    ),
    APIInfo(
        api_id="app_outlook",
        name="Microsoft Outlook",
        kind="app",
        exe_name="outlook.exe",
        python_calls=["subprocess.Popen(['outlook.exe'])"],
        keywords=["outlook", "microsoft outlook", "email client", "calendar"],
    ),
    APIInfo(
        api_id="app_teams",
        name="Microsoft Teams",
        kind="app",
        exe_name="teams.exe",
        python_calls=["subprocess.Popen(['teams.exe'])"],
        keywords=["teams", "microsoft teams", "meeting", "video call", "ms teams"],
    ),
    APIInfo(
        api_id="app_zoom",
        name="Zoom",
        kind="app",
        exe_name="Zoom.exe",
        python_calls=["subprocess.Popen(['Zoom.exe'])"],
        keywords=["zoom", "zoom meeting", "video conference", "zoom call"],
    ),
    APIInfo(
        api_id="app_vlc",
        name="VLC Media Player",
        kind="app",
        exe_name="vlc.exe",
        python_calls=["subprocess.Popen(['vlc.exe', 'video.mp4'])"],
        keywords=["vlc", "media player", "play video", "play music", "video player"],
    ),
    APIInfo(
        api_id="app_spotify",
        name="Spotify Desktop",
        kind="app",
        exe_name="Spotify.exe",
        python_calls=["subprocess.Popen(['Spotify.exe'])"],
        keywords=["spotify", "spotify app", "music player", "open spotify"],
    ),
    APIInfo(
        api_id="app_paint",
        name="Microsoft Paint",
        kind="app",
        exe_name="mspaint.exe",
        python_calls=["subprocess.Popen(['mspaint.exe'])"],
        keywords=["paint", "ms paint", "draw", "image editor", "open paint"],
    ),
    APIInfo(
        api_id="app_snipping_tool",
        name="Snipping Tool",
        kind="app",
        exe_name="SnippingTool.exe",
        python_calls=["subprocess.Popen(['SnippingTool.exe'])"],
        keywords=["snipping tool", "snip", "screenshot tool", "snip and sketch"],
    ),
]


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------


class APIRegistry:
    """
    Finds APIs / system controls / apps relevant to a user command.

    This gives the planner context about expected URLs, capabilities,
    Python modules to use, and auth constraints before generating state
    transitions.
    """

    def __init__(self, apis: Optional[List[APIInfo]] = None):
        self._apis: List[APIInfo] = apis if apis is not None else list(_DEFAULT_APIS)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list_all(self) -> List[Dict[str, Any]]:
        """List all registered APIs in prompt-friendly format."""
        return [api.to_prompt_dict() for api in self._apis]

    def list_by_kind(self, kind: APIKind) -> List[Dict[str, Any]]:
        """List all APIs of a given kind: 'web', 'system', or 'app'."""
        return [api.to_prompt_dict() for api in self._apis if api.kind == kind]

    def find_relevant(self, command: str) -> List[Dict[str, Any]]:
        """
        Return APIs / system controls / apps relevant to the command
        using keyword overlap.
        """
        if not command:
            return []

        lowered = command.lower()
        tokens = set(re.findall(r"[a-z0-9]+", lowered))
        relevant: List[APIInfo] = []

        for api in self._apis:
            corpus = {
                api.api_id.lower(),
                api.name.lower(),
                *[k.lower() for k in api.keywords],
            }
            if any(keyword in lowered or keyword in tokens for keyword in corpus):
                relevant.append(api)

        return [api.to_prompt_dict() for api in relevant]

    def get(self, api_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single API by its ID."""
        for api in self._apis:
            if api.api_id == api_id:
                return api.to_prompt_dict()
        return None

    def register(self, api: APIInfo) -> None:
        """Add a custom API entry at runtime."""
        self._apis.append(api)
