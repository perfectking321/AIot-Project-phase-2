"""
VOXCODE System Skills — Windows Native API Layer
Root-level Windows system access via PowerShell, ctypes, and subprocess.

IMPORTANT: Full capabilities require VOXCODE to run as Administrator.
Non-admin mode still supports most operations except service control and registry write.

Skills provided:
- SystemCommandSkill     — run any PowerShell/cmd command, return output
- ProcessManagerSkill    — list, kill, start processes
- NetworkInfoSkill       — get IP, adapters, connections, DNS flush
- BrightnessControlSkill — set brightness to specific level, max, or min
- BluetoothControlSkill  — toggle Bluetooth radio on/off
- SystemInfoSkill        — CPU, RAM, disk usage stats
- WindowManagerSkill     — list windows, focus, minimize, maximize, close
"""

import os
import subprocess
import logging
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

from agent.skills.base import Skill, SkillResult, SkillStatus

logger = logging.getLogger("voxcode.skills.system")


# ── Elevation Detection ─────────────────────────────────────────────────────

def is_admin() -> bool:
    """Check if VOXCODE is running with Administrator privileges."""
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


IS_ADMIN = is_admin()
if not IS_ADMIN:
    logger.warning(
        "VOXCODE is NOT running as Administrator. "
        "Some system skills (services, registry write) will be limited. "
        "Right-click main.py → Run as administrator for full access."
    )


# ── PowerShell Executor ─────────────────────────────────────────────────────

def run_powershell(command: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute a PowerShell command and return structured output.
    """
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-Command", command
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False  # Explicit — never shell=True for elevated ops
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Command timed out", "returncode": -1, "success": False}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1, "success": False}


def run_cmd(command: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute a cmd.exe command."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout,
            shell=True, encoding="utf-8", errors="replace"
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "success": result.returncode == 0
        }
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1, "success": False}


# ── Skills ──────────────────────────────────────────────────────────────────

class SystemCommandSkill(Skill):
    """
    Run any PowerShell or cmd command and return output.
    This is the ROOT-LEVEL access skill — the agent's direct interface to Windows.
    """

    name = "system_command"
    description = "Run a PowerShell or cmd command and return the output"
    params = ["command"]
    preconditions = []
    postconditions = ["command_executed"]

    def execute(self, command: str = None, use_cmd: bool = False, **kwargs) -> SkillResult:
        if not command:
            return SkillResult(status=SkillStatus.FAILED, message="command is required")

        if use_cmd:
            result = run_cmd(command)
        else:
            result = run_powershell(command)

        if result["success"]:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Command executed successfully",
                data={"output": result["stdout"], "command": command}
            )
        else:
            return SkillResult(
                status=SkillStatus.FAILED,
                message=f"Command failed: {result['stderr'] or result['stdout']}",
                data=result
            )


class ProcessManagerSkill(Skill):
    """List, find, kill, and start processes."""

    name = "process_manager"
    description = "Manage Windows processes: list, kill by name/PID, or get resource usage"
    params = ["action"]
    preconditions = []
    postconditions = ["process_action_done"]

    def execute(
        self,
        action: str = "list",
        name: str = None,
        pid: int = None,
        sort_by: str = "cpu",
        top_n: int = 15,
        **kwargs
    ) -> SkillResult:

        if action == "list":
            sort_col = "CPU" if sort_by == "cpu" else ("WS" if sort_by == "memory" else "Name")
            ps_cmd = f"""
            Get-Process |
            Sort-Object {sort_col} -Descending |
            Select-Object -First {top_n} Name, Id,
                @{{N='CPU_s';E={{[math]::Round($_.CPU,1)}}}},
                @{{N='RAM_MB';E={{[math]::Round($_.WorkingSet64/1MB,1)}}}} |
            ConvertTo-Json
            """
            result = run_powershell(ps_cmd)
            if result["success"]:
                try:
                    data = json.loads(result["stdout"])
                    if isinstance(data, dict):
                        data = [data]
                except Exception:
                    data = result["stdout"]
                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    message=f"Top {top_n} processes by {sort_by}",
                    data={"processes": data}
                )

        elif action == "kill":
            if pid:
                ps_cmd = f"Stop-Process -Id {pid} -Force"
            elif name:
                ps_cmd = f"Stop-Process -Name '{name}' -Force"
            else:
                return SkillResult(status=SkillStatus.FAILED, message="Provide name or pid to kill")

            result = run_powershell(ps_cmd)
            return SkillResult(
                status=SkillStatus.SUCCESS if result["success"] else SkillStatus.FAILED,
                message=f"Killed {name or pid}" if result["success"] else result["stderr"]
            )

        elif action == "info":
            if pid:
                ps_cmd = f"Get-Process -Id {pid} | Select-Object Name, Id, CPU, WorkingSet64, Path | ConvertTo-Json"
            elif name:
                ps_cmd = f"Get-Process -Name '{name}' | Select-Object Name, Id, CPU, WorkingSet64, Path | ConvertTo-Json"
            else:
                return SkillResult(status=SkillStatus.FAILED, message="Provide name or pid for info")
            result = run_powershell(ps_cmd)
            return SkillResult(
                status=SkillStatus.SUCCESS if result["success"] else SkillStatus.FAILED,
                message=f"Process info for {name or pid}",
                data={"raw": result["stdout"]}
            )

        return SkillResult(status=SkillStatus.FAILED, message=f"Unknown action: {action}")


class NetworkInfoSkill(Skill):
    """Get network information and perform network operations."""

    name = "network_info"
    description = "Get IP addresses, network adapters, active connections, flush DNS"
    params = ["action"]
    preconditions = []
    postconditions = ["network_info_retrieved"]

    def execute(self, action: str = "ip", **kwargs) -> SkillResult:
        commands = {
            "ip": "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -ne '127.0.0.1' } | Select-Object InterfaceAlias, IPAddress, PrefixLength | Format-Table -AutoSize | Out-String",
            "adapters": "Get-NetAdapter | Select-Object Name, Status, LinkSpeed, MacAddress | Format-Table -AutoSize | Out-String",
            "connections": "netstat -an | Select-String 'ESTABLISHED|LISTENING' | Select-Object -First 20 | Out-String",
            "dns_flush": "Clear-DnsClientCache; Write-Output 'DNS cache flushed'",
            "wifi": "netsh wlan show interfaces",
            "ping": "Test-Connection -ComputerName 8.8.8.8 -Count 2 | Format-Table -AutoSize | Out-String",
        }
        ps_cmd = commands.get(action)
        if not ps_cmd:
            return SkillResult(status=SkillStatus.FAILED, message=f"Unknown action: {action}. Available: {', '.join(commands.keys())}")

        result = run_powershell(ps_cmd)
        if result["success"]:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Network {action}",
                data={"output": result["stdout"]}
            )
        else:
            return SkillResult(
                status=SkillStatus.FAILED,
                message=f"Network {action} failed: {result['stderr']}"
            )


class BrightnessControlSkill(Skill):
    """Set screen brightness to a specific level, max, or min."""

    name = "brightness_control"
    description = "Set screen brightness to max (100%), min (0%), or a specific percentage"
    params = ["level"]
    preconditions = []
    postconditions = ["brightness_set"]

    def execute(self, level: str = None, **kwargs) -> SkillResult:
        if not level:
            return SkillResult(status=SkillStatus.FAILED, message="level is required (e.g., 'max', 'min', '50')")

        level_str = str(level).strip().lower()

        # Determine numeric level
        if level_str in ("max", "maximum", "100", "full", "highest"):
            numeric_level = 100
        elif level_str in ("min", "minimum", "0", "lowest", "off"):
            numeric_level = 0
        else:
            try:
                numeric_level = max(0, min(100, int(level_str)))
            except ValueError:
                return SkillResult(status=SkillStatus.FAILED, message=f"Invalid brightness level: {level}")

        ps_cmd = (
            "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods)"
            f".WmiSetBrightness(1,{numeric_level})"
        )
        result = run_powershell(ps_cmd)

        if result["success"] or result["returncode"] == 0:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Brightness set to {numeric_level}%"
            )
        else:
            return SkillResult(
                status=SkillStatus.FAILED,
                message=f"Failed to set brightness: {result['stderr']}"
            )


class BluetoothControlSkill(Skill):
    """Toggle Bluetooth radio on or off."""

    name = "bluetooth_control"
    description = "Turn Bluetooth on or off"
    params = ["action"]
    preconditions = []
    postconditions = ["bluetooth_toggled"]

    def execute(self, action: str = "toggle", **kwargs) -> SkillResult:
        action_lower = action.strip().lower()

        if action_lower in ("off", "disable", "turn off", "false", "0"):
            # Disable Bluetooth radio
            ps_cmd = """
            $bt = Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue | Where-Object { $_.Status -eq 'OK' -and $_.FriendlyName -match 'Bluetooth' }
            if ($bt) {
                $bt | Disable-PnpDevice -Confirm:$false -ErrorAction SilentlyContinue
                Write-Output "Bluetooth disabled"
            } else {
                # Fallback: try radio management
                $radio = Get-PnpDevice | Where-Object { $_.Class -eq 'Bluetooth' -and $_.Status -eq 'OK' }
                if ($radio) { $radio | Disable-PnpDevice -Confirm:$false; Write-Output "Bluetooth disabled" }
                else { Write-Output "No active Bluetooth device found" }
            }
            """
        elif action_lower in ("on", "enable", "turn on", "true", "1"):
            # Enable Bluetooth radio
            ps_cmd = """
            $bt = Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue | Where-Object { $_.Status -eq 'Error' -or $_.Status -eq 'Disabled' }
            if ($bt) {
                $bt | Enable-PnpDevice -Confirm:$false -ErrorAction SilentlyContinue
                Write-Output "Bluetooth enabled"
            } else {
                $radio = Get-PnpDevice | Where-Object { $_.Class -eq 'Bluetooth' -and ($_.Status -eq 'Error' -or $_.Status -eq 'Disabled') }
                if ($radio) { $radio | Enable-PnpDevice -Confirm:$false; Write-Output "Bluetooth enabled" }
                else { Write-Output "Bluetooth is already on or no device found" }
            }
            """
        elif action_lower in ("status", "state", "check"):
            ps_cmd = """
            $bt = Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue
            if ($bt) { $bt | Select-Object FriendlyName, Status | Format-Table -AutoSize | Out-String }
            else { Write-Output "No Bluetooth devices found" }
            """
        else:
            return SkillResult(status=SkillStatus.FAILED, message=f"Unknown action: {action}. Use 'on', 'off', or 'status'")

        result = run_powershell(ps_cmd)

        if result["success"]:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Bluetooth: {result['stdout']}",
                data={"output": result["stdout"]}
            )
        else:
            # Bluetooth commands might need admin
            msg = result["stderr"] or result["stdout"]
            if "access" in msg.lower() or "denied" in msg.lower() or "administrator" in msg.lower():
                return SkillResult(
                    status=SkillStatus.FAILED,
                    message="Bluetooth control requires Administrator. Run VOXCODE as admin."
                )
            return SkillResult(
                status=SkillStatus.FAILED,
                message=f"Bluetooth {action} failed: {msg}"
            )


class SystemInfoSkill(Skill):
    """Get real-time system resource usage."""

    name = "system_info"
    description = "Get CPU, RAM, disk, and GPU usage statistics"
    params = []
    preconditions = []
    postconditions = ["system_info_retrieved"]

    def execute(self, **kwargs) -> SkillResult:
        ps_cmd = """
        $cpu = (Get-WmiObject Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        $os = Get-WmiObject Win32_OperatingSystem
        $ram_total = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
        $ram_free = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
        $ram_used = $ram_total - $ram_free
        $disk = Get-WmiObject Win32_LogicalDisk -Filter "DriveType=3" |
            Select-Object DeviceID,
                @{N='Total_GB';E={[math]::Round($_.Size/1GB,1)}},
                @{N='Free_GB';E={[math]::Round($_.FreeSpace/1GB,1)}}

        [PSCustomObject]@{
            CPU_Percent = $cpu
            RAM_Used_GB = $ram_used
            RAM_Total_GB = $ram_total
            RAM_Percent = [math]::Round(($ram_used / $ram_total) * 100, 1)
            Disks = $disk
        } | ConvertTo-Json -Depth 3
        """
        result = run_powershell(ps_cmd)
        if result["success"]:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message="System resource usage",
                data={"stats": result["stdout"]}
            )
        return SkillResult(status=SkillStatus.FAILED, message=f"System info failed: {result['stderr']}")


class WindowManagerSkill(Skill):
    """List and control application windows."""

    name = "window_manager"
    description = "List open windows, focus, minimize, maximize, or close them"
    params = ["action"]
    preconditions = []
    postconditions = ["window_action_done"]

    def execute(self, action: str = "list", window_title: str = None, **kwargs) -> SkillResult:
        if action == "list":
            ps_cmd = """
            Get-Process | Where-Object {$_.MainWindowTitle -ne ''} |
            Select-Object Name, Id, MainWindowTitle |
            Format-Table -AutoSize | Out-String
            """
            result = run_powershell(ps_cmd)
            return SkillResult(
                status=SkillStatus.SUCCESS if result["success"] else SkillStatus.FAILED,
                message="Open windows",
                data={"windows": result["stdout"]}
            )

        if not window_title:
            return SkillResult(status=SkillStatus.FAILED, message="window_title is required")

        try:
            import pygetwindow as gw
            windows = gw.getWindowsWithTitle(window_title)
            if not windows:
                return SkillResult(status=SkillStatus.FAILED, message=f"Window '{window_title}' not found")

            win = windows[0]
            if action == "focus":
                win.activate()
            elif action == "minimize":
                win.minimize()
            elif action == "maximize":
                win.maximize()
            elif action == "close":
                win.close()

            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Window '{window_title}' {action}d"
            )
        except ImportError:
            ps_cmd = f"""
            $wshell = New-Object -ComObject wscript.shell
            $wshell.AppActivate('{window_title}')
            """
            result = run_powershell(ps_cmd)
            return SkillResult(
                status=SkillStatus.SUCCESS if result["success"] else SkillStatus.FAILED,
                message=f"Window action: {action} on '{window_title}'"
            )
        except Exception as e:
            return SkillResult(status=SkillStatus.FAILED, message=str(e))
