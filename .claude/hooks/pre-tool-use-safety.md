---
name: pre-tool-use-safety
event: PreToolUse
tools: ["Write", "Edit", "Bash"]
description: Safety check before file modifications or command execution
---

# Pre-Tool Safety Hook

## Purpose
Validates operations before execution to prevent accidental damage.

## Trigger Conditions
Activates before:
- `Write` tool (file creation/overwrite)
- `Edit` tool (file modification)
- `Bash` tool (command execution)

## Safety Checks

### File Operations (Write/Edit)

```bash
# Check 1: Protected paths
PROTECTED_PATHS=(
  ".git/"
  "node_modules/"
  ".env"
  "*.pem"
  "*.key"
  "package-lock.json"
  "yarn.lock"
)

for pattern in "${PROTECTED_PATHS[@]}"; do
  if [[ "$FILE_PATH" == *"$pattern"* ]]; then
    echo "⚠️ Warning: Modifying protected path: $FILE_PATH"
    echo "Confirm: Is this intentional? [y/N]"
  fi
done

# Check 2: Backup reminder for large changes
if [[ "$CHANGE_SIZE" -gt 100 ]]; then
  echo "📝 Large change detected (>100 lines)"
  echo "Consider: git stash or backup first"
fi
```

### Command Execution (Bash)

```bash
# Check 1: Dangerous commands
DANGEROUS_COMMANDS=(
  "rm -rf"
  "rm -r /"
  "dd if="
  "mkfs"
  "> /dev/"
  "chmod -R 777"
  "git push --force"
  "git reset --hard"
  "DROP TABLE"
  "DELETE FROM"
  "TRUNCATE"
)

for cmd in "${DANGEROUS_COMMANDS[@]}"; do
  if [[ "$COMMAND" == *"$cmd"* ]]; then
    echo "🚨 DANGEROUS COMMAND DETECTED: $cmd"
    echo "This operation is blocked by safety hook."
    echo "Override: Use --force flag if intentional"
    exit 1
  fi
done

# Check 2: Production indicators
if [[ "$COMMAND" == *"prod"* ]] || [[ "$COMMAND" == *"production"* ]]; then
  echo "⚠️ Production environment detected"
  echo "Double-check before proceeding"
fi
```

## Output Messages

### Warning (Proceed with Caution)
```
⚠️ Safety Check Warning
━━━━━━━━━━━━━━━━━━━━━━
Operation: {operation}
Target: {target}
Risk: {risk_level}

Proceeding... (use --safe-mode to block)
```

### Block (Dangerous Operation)
```
🚨 Operation Blocked
━━━━━━━━━━━━━━━━━━━━
Command: {command}
Reason: {reason}

To override: Add --force flag
To review: Check command and try again
```

## Configuration

```json
{
  "hooks": {
    "pre-tool-use-safety": {
      "enabled": true,
      "block_dangerous": true,
      "warn_protected": true,
      "protected_paths": [
        ".git/",
        ".env*",
        "*.key",
        "*.pem"
      ],
      "dangerous_commands": [
        "rm -rf",
        "git push --force"
      ]
    }
  }
}
```

## Bypass Options

1. **--force**: Override all safety checks (use with caution)
2. **--no-hooks**: Disable all hooks temporarily
3. **Config**: Modify `.claude/settings.json` to adjust rules
