
Copy the toggle_attention.py to your path like ~/bin/

## Available Options

- `--waiting`: Sets warning indicator and announces "Agent waiting" (when agent needs user input)
- `--running`: Clears warning indicator (when agent is actively working)
- `--stop`: Sets warning indicator and announces "Agent done" (when agent completes its work)

## Claude Code Configuration

For claude code set the following in ~/.claude/setttings.json (or .claude/settings.json)

```json
{
  "hooks": {
    "Notification": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "toggle_attention.py --waiting"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "toggle_attention.py --running"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "toggle_attention.py --stop"
          }
        ]
      }
    ]
  }
}
```

## Codex

TODO: Modify the toggle_attention so that it works with both Claude and Codex (and Gemini)

Codex gets notified too (different API/method). E.g:
Adding to .codex/config.toml
```
notify = ["/path/to/notify.py"]
```

Then something like this:
```
#!/usr/bin/env python3
import json, subprocess, sys

def main() -> int:
    notification = json.loads(sys.argv[1])
    if notification.get("type") != "agent-turn-complete":
        return 0
    title = f"Codex: {notification.get('last-assistant-message', 'Turn Complete!')}"
    message = " ".join(notification.get("input-messages", []))
    subprocess.check_output([
        "terminal-notifier",
        "-title", title,
        "-message", message,
        "-group", "codex-" + notification.get("thread-id", ""),
        "-activate", "com.googlecode.iterm2",
    ])
    return 0

if __name__ == "__main__":
    sys.exit(main())
```
