## Overview

This repository contains scripts that collect and analyze AI tool usage statistics from local log files of Claude Code, Codex CLI, Gemini CLI, and **Cursor**. The script uses heuristics to parse various log formats and generates detailed statistics with per-tool breakdowns, execution times, and working directory tracking.

### Cursor Support

Cursor stores usage data in `~/.cursor/ai-tracking/ai-code-tracking.db`. The script automatically discovers and parses this SQLite database. You get code generation (Write) counts and conversation stats. No setup required.

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

# Include Cursor logs explicitly (normally auto-discovered)
./ai_usage_stats.py --student "student@example.com" --roots '~/.cursor/ai-tracking/ai-code-tracking.db'
```

The `--filter` option allows you to focus analysis on specific projects (avoids leaking information that you may not want to share):

- `--filter "cse247b"` matches sessions where the project contains "cse247b"
- `--filter "hagent"` matches paths like `.../hagent/core` or `.../hagent/foo/core`

## Agent Notifications

The `hooks/` directory contains configuration files and scripts to set up visual/audio notifications for coding agent activity. See [hooks/README.md](hooks/README.md) for setup instructions on how to configure Claude Code to notify you when it's waiting for input or actively running.
