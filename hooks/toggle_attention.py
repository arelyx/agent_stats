#!/usr/bin/env python3
"""
WezTerm attention indicator for Claude Code hooks.
Sets/clears visual warning and optionally announces when agent is waiting.
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path


def get_tab_id(pane_id):
    """Get the WezTerm tab ID for the current pane."""
    try:
        result = subprocess.run(
            ["wezterm", "cli", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
        panes = json.loads(result.stdout)

        for pane in panes:
            if pane.get("pane_id") == pane_id:
                return pane.get("tab_id")

        raise ValueError(f"Could not find tab for pane {pane_id}")

    except subprocess.CalledProcessError as e:
        print(f"Error running wezterm cli: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing wezterm output: {e}", file=sys.stderr)
        sys.exit(1)


def set_attention_state(tab_id, state):
    """
    Set the attention state for the WezTerm tab.

    Args:
        tab_id: WezTerm tab ID
        state: 1 for warning/waiting, 0 for clear/running
    """
    # Create state directory
    state_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "wezterm-attention"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Write state file
    state_file = state_dir / f"tab-{tab_id}"
    state_file.write_text(str(state))

    # Set pane user var "attention" (must be base64 encoded)
    b64_state = base64.b64encode(str(state).encode()).decode().strip()
    escape_sequence = f"\033]1337;SetUserVar=attention={b64_state}\a"

    # Write directly to /dev/tty to bypass output capture by Claude Code
    # This ensures the escape sequence reaches WezTerm even when stdout is captured
    try:
        with open("/dev/tty", "w") as tty:
            tty.write(escape_sequence)
            tty.flush()
    except Exception as e:
        # Fallback to stdout if /dev/tty is not available
        print(f"Warning: Could not write to /dev/tty: {e}", file=sys.stderr)
        sys.stdout.write(escape_sequence)
        sys.stdout.flush()


def announce_waiting():
    """Use text-to-speech to announce agent is waiting."""
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say("Agent waiting")
        engine.runAndWait()
    except ImportError:
        print("Warning: pyttsx3 not installed, skipping audio announcement", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not announce via TTS: {e}", file=sys.stderr)


def announce_done():
    """Use text-to-speech to announce agent is done."""
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say("Agent done")
        engine.runAndWait()
    except ImportError:
        print("Warning: pyttsx3 not installed, skipping audio announcement", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not announce via TTS: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Set WezTerm attention indicator for Claude Code agent status"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--waiting",
        action="store_true",
        help="Set warning indicator (agent waiting for user input)",
    )
    group.add_argument(
        "--running",
        action="store_true",
        help="Clear warning indicator (agent running)",
    )
    group.add_argument(
        "--stop",
        action="store_true",
        help="Set warning indicator and announce agent done",
    )

    args = parser.parse_args()

    # Check for WezTerm environment
    pane_id_str = os.environ.get("WEZTERM_PANE")
    if not pane_id_str:
        print("Error: Must be run inside a WezTerm pane (WEZTERM_PANE not set)", file=sys.stderr)
        sys.exit(2)

    try:
        pane_id = int(pane_id_str)
    except ValueError:
        print(f"Error: Invalid WEZTERM_PANE value: {pane_id_str}", file=sys.stderr)
        sys.exit(2)

    # Get tab ID
    tab_id = get_tab_id(pane_id)

    # Set state based on argument
    if args.waiting:
        set_attention_state(tab_id, 1)
        announce_waiting()
    elif args.stop:
        set_attention_state(tab_id, 1)
        announce_done()
    else:  # args.running
        set_attention_state(tab_id, 0)


if __name__ == "__main__":
    main()
