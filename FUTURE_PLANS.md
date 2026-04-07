# VOXCODE Future Plans

## Plan 1: Terminal-Based File System Operations (Claude Code Style)

**Status:** Planned (To be implemented after Browser-Use integration)

### Goal
Instead of opening File Explorer and clicking around, VOXCODE should use terminal commands to interact with the file system - just like Claude Code does.

### User Story
User says: "What files are in my Projects folder?"
AI responds: Lists all files using `dir` or `ls` commands, displays them in the TUI.

User says: "Create a new Python file called app.py in Projects"
AI responds: Uses `echo` or terminal commands to create the file, confirms success.

### Implementation Approach

1. **Create `agent/skills/terminal_skills.py`**
   - `ListFilesSkill`: Run `dir` (Windows) to list directory contents
   - `CreateFileSkill`: Create files via terminal commands
   - `ReadFileSkill`: Read file contents via `type` command
   - `NavigateSkill`: Change working directory
   - `SearchFilesSkill`: Find files by pattern

2. **Add Terminal Session Manager**
   - Persistent terminal session (like Claude Code)
   - Working directory tracking
   - Command history

3. **TUI Display Enhancement**
   - Format terminal output nicely in the log
   - Syntax highlighting for code files
   - Tree view for directory listings

4. **Command Detection**
   - Detect file-related commands: "list", "show files", "what's in", "create file", "open folder"
   - Route to terminal skills instead of pyautogui

### Benefits
- Much faster than opening File Explorer
- More reliable (no UI element detection needed)
- Can handle complex operations (find all .py files, search in files)
- Mirrors how developers actually work

### Example Commands
```
"List files in C:\Projects\VOXCODE"
→ dir "C:\Projects\VOXCODE"
→ Display formatted output

"Find all Python files in my projects"
→ dir /s /b "C:\Projects\*.py"
→ Display list

"Create a new folder called 'backup'"
→ mkdir backup
→ Confirm creation

"Show me what's in config.py"
→ type config.py
→ Display with syntax highlighting
```

---

## Plan 2: Project Creation Assistant (Future)

**Status:** Concept

### Goal
Help users create new projects with proper structure, like a coding assistant.

### Features
- Template-based project creation
- Dependency installation
- Git initialization
- README generation

---

*Last updated: 2024*
