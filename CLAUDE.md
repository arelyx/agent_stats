# CLAUDE.md

This file provides guidance to coding agents when working with code in this repository.

## Overview

This repository contains `ai_usage_stats.py`, a Python script that collects and analyzes AI tool usage statistics from local log files of Claude Code, Codex CLI, and Gemini CLI. The script uses heuristics to parse various log formats and generates detailed statistics with per-tool breakdowns, execution times, and working directory tracking.

## Running the Script

```bash
# Basic usage with required student identifier
./ai_usage_stats.py --student "student@example.com"

# Filter statistics by directory/file regex pattern
./ai_usage_stats.py --student "student@example.com" --filter "cse247b"
./ai_usage_stats.py --student "student@example.com" --filter "project.*src"

# Specify custom CSV output path
./ai_usage_stats.py --student "student@example.com" --csv trace.csv

# Enable debug JSON output
./ai_usage_stats.py --student "student@example.com" --json debug.json

# Search custom log locations instead of defaults
./ai_usage_stats.py --student "student@example.com" --roots '~/.claude/projects/**/*.jsonl'
```

## Architecture

### Log Discovery & Classification

The script automatically discovers log files from default locations:
- **Claude Code**: `~/.claude/projects/**/*.jsonl`
- **Codex CLI**: `~/.codex/sessions/**/rollout-*.jsonl` (or `$CODEX_HOME`)
- **Gemini CLI**: `~/.gemini/tmp/**/session-*.json`
- **Cursor**: `~/.cursor/ai-tracking/ai-code-tracking.db` (SQLite)

Files are classified by path patterns and extensions. The `classify_and_parse()` function (line 274) routes each file to the appropriate parser.

### Parsing Strategy

The script uses format-tolerant heuristics rather than strict schemas:

1. **Message Detection**: Identifies user/assistant messages by checking common role/author fields
2. **Tool Call Extraction**: For Claude Code logs, extracts:
   - Tool names (Read, Write, Edit, Bash, etc.)
   - Execution times (by tracking tool_use_id from call to result)
   - Working directories (from file_path, path, directory parameters)
3. **Format-Specific Parsers**:
 - `parse_claude_jsonl()`: Detailed parser with timing and directory extraction
 - `parse_codex_jsonl()`: Basic counting parser
 - `parse_gemini_json()`: Basic counting parser with nested structure handling
 - `parse_cursor_sqlite()`: Parses Cursor's ai-code-tracking.db

### Data Model

**`ToolStats`**: Per-tool metrics
- `count`: Number of times the tool was called
- `execution_times`: List of execution times in seconds
- `avg_time()`: Helper to calculate average execution time

**`TraceEvent`**: A single time-stamped event for the CSV trace
- `timestamp`: ISO format timestamp
- `event_type`: "user_prompt", "assistant_response", "tool_call", or "tool_result"
- `coding_agent`: Which AI agent was used ("claude_code", "codex_cli", "gemini_cli", "cursor")
- `tool_name`: Name of the tool (for tool events)
- `execution_time`: Execution time in seconds (for tool_result events)
- `working_dir`: Working directory extracted from tool parameters
- `session_id`: Session identifier

**`SessionStats`**: Per-session aggregated metrics
- `tool`: Which coding agent ("claude_code", "codex_cli", "gemini_cli", "cursor")
- `session_id`: Unique session identifier
- `session_cwd`: Session-level working directory (from `cwd`/`workdir` fields)
- `prompts`: Count of user messages
- `assistant_msgs`: Count of assistant/model responses
- `tool_calls`: Total tool invocations
- `tool_stats`: Per-tool statistics dictionary
- `iterations_per_prompt`: List of iteration counts for each prompt
- `prompt_response_times`: List of response times for prompts
- `working_dirs`: List of tool-level working directories used
- `trace_events`: Time-ordered list of events

### Filtering

The `--filter` option allows you to focus analysis on specific projects using regex patterns. **Filtering works at the session level**, not per-tool:

- **Simple match**: `--filter "cse247b"` matches sessions where the project contains "cse247b"
- **Regex pattern**: `--filter "hagent.*core"` matches paths like `/hagent/core` or `/hagent/foo/core`
- **Path components**: `--filter "/src/"` matches only sessions with a `/src/` directory

#### How Session-Level Filtering Works

A session is included if **any** of these match the filter:
1. The session's working directory (`cwd` field in logs)
2. Any tool-level working directory used in the session

When a session matches:
- The **entire session** is included with all its statistics
- All prompts, assistant messages, and tool calls from that session are counted
- Works across all coding agents (Claude Code, Codex, Gemini, Cursor)

**Cursor exception**: Cursor's SQLite database aggregates all projects into one session. When a filter matches, the script applies **event-level** filtering: only trace events whose `working_dir` matches the filter are included. This ensures the summary and CSV contain only data for the specified project.

**Example**: With `--filter "hagent"`:
- Claude Code session working on `/Users/you/projs/hagent` → included
- Codex session with `cwd: /Users/you/projs/hagent` → included
- Session with no hagent references → excluded

This ensures you get complete statistics for all AI coding agents that worked on your filtered project.

### Output

The script generates:

1. **Console Summary** (Ruby-style format):
   ```
   === Claude Code Trace Analysis (N files) ===

   📊 AGGREGATE STATISTICS:
     • Total files analyzed: N
     • Total tools called: N
     • Tool use: X.XXX sec avg, N calls
       + Read tool: X.X sec avg, N calls
       + Write tool: X.X sec avg, N calls
       ...
     • Total user prompts: N
     • User prompts: X.XXX sec avg, N calls
     • Average Claude iterations per prompt: X.XX

   📊 BY CODING AGENT:
     • claude_code: N sessions, N prompts, N tool calls
       + Edit: X.X sec avg, N calls
       + Read: X.X sec avg, N calls
       + Bash: X.X sec avg, N calls
     • codex_cli: N sessions, N prompts, N tool calls
     • gemini_cli: N sessions, N prompts, N tool calls

   📂 WORKING DIRECTORIES (N unique):
     • /path/to/dir1
     • /path/to/dir2
     ...
   ```

2. **Time-based CSV trace** (`ai_usage_trace.csv`): Chronological event log with columns:
   - timestamp
   - event_type
   - coding_agent (claude_code, codex_cli, gemini_cli)
   - tool_name
   - execution_time
   - working_dir
   - session_id

3. **Optional debug JSON** (via `--json` flag): Complete statistics including execution time distributions, all working directories, and per-session breakdowns

The student identifier is hashed using SHA-256 to create a stable pseudonym without storing the original identifier.
