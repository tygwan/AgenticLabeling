#!/bin/bash
#
# Post-Tool-Use Tracker Hook
# Tracks file changes and tool usage for session awareness
#
# Events: PostToolUse (all tools)
#

TOOL_NAME="$1"
TOOL_INPUT="$2"
TOOL_OUTPUT="$3"
EXIT_CODE="${4:-0}"

# Configuration
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
SESSION_LOG="$PROJECT_ROOT/.claude/session.log"
CHANGES_LOG="$PROJECT_ROOT/.claude/changes.log"

# Colors
CYAN='\033[0;36m'
NC='\033[0m'

# Ensure log directory exists
mkdir -p "$(dirname "$SESSION_LOG")"

# Get timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Log tool usage
log_tool_usage() {
    local tool="$1"
    local input="$2"
    local status="$3"

    # Extract filename if present
    local filename=""
    if [[ "$input" =~ file_path[\":\s]+([^\",]+) ]]; then
        filename="${BASH_REMATCH[1]}"
    elif [[ "$input" =~ \"path\"[:\s]+\"([^\"]+)\" ]]; then
        filename="${BASH_REMATCH[1]}"
    fi

    # Log to session log (last 100 entries)
    echo "[$TIMESTAMP] $tool: ${filename:-${input:0:50}}" >> "$SESSION_LOG"

    # Keep only last 100 lines
    if [[ -f "$SESSION_LOG" ]]; then
        tail -n 100 "$SESSION_LOG" > "$SESSION_LOG.tmp" && mv "$SESSION_LOG.tmp" "$SESSION_LOG"
    fi
}

# Track file modifications
track_file_change() {
    local tool="$1"
    local input="$2"

    case "$tool" in
        "Write"|"Edit"|"MultiEdit")
            # Extract file path
            if [[ "$input" =~ file_path[\":\s]+\"?([^\",]+)\"? ]]; then
                local filepath="${BASH_REMATCH[1]}"
                echo "[$TIMESTAMP] MODIFIED: $filepath" >> "$CHANGES_LOG"
            fi
            ;;
        "Bash")
            # Track git operations
            if [[ "$input" == *"git add"* ]] || [[ "$input" == *"git commit"* ]]; then
                echo "[$TIMESTAMP] GIT: ${input:0:80}" >> "$CHANGES_LOG"
            fi
            ;;
    esac

    # Keep only last 50 entries in changes log
    if [[ -f "$CHANGES_LOG" ]]; then
        tail -n 50 "$CHANGES_LOG" > "$CHANGES_LOG.tmp" && mv "$CHANGES_LOG.tmp" "$CHANGES_LOG"
    fi
}

# Main logic
main() {
    # Log all tool usage
    log_tool_usage "$TOOL_NAME" "$TOOL_INPUT" "$EXIT_CODE"

    # Track file modifications
    track_file_change "$TOOL_NAME" "$TOOL_INPUT"
}

main
exit 0
