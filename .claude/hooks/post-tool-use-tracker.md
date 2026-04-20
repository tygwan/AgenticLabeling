---
name: post-tool-use-tracker
event: PostToolUse
tools: ["Write", "Edit", "Bash"]
description: Track changes and suggest follow-up actions after tool execution
---

# Post-Tool Change Tracker Hook

## Purpose
Tracks changes made and suggests relevant follow-up actions.

## Trigger Conditions
Activates after:
- `Write` tool (file created/overwritten)
- `Edit` tool (file modified)
- `Bash` tool (command executed)

## Tracking Logic

### File Changes (Write/Edit)

```bash
# Track file type and suggest actions
FILE_EXT="${FILE_PATH##*.}"

case "$FILE_EXT" in
  ts|js|tsx|jsx)
    echo "📝 TypeScript/JavaScript file modified"
    echo "Suggestions:"
    echo "  • Run: npm run lint"
    echo "  • Run: npm test"
    ;;
  py)
    echo "📝 Python file modified"
    echo "Suggestions:"
    echo "  • Run: python -m pytest"
    echo "  • Run: black {file}"
    ;;
  cs)
    echo "📝 C# file modified"
    echo "Suggestions:"
    echo "  • Run: dotnet build"
    echo "  • Run: dotnet test"
    ;;
  md)
    echo "📝 Documentation modified"
    echo "Suggestions:"
    echo "  • Review for accuracy"
    echo "  • Check links"
    ;;
  json|yaml|yml)
    echo "📝 Config file modified"
    echo "Suggestions:"
    echo "  • Validate syntax"
    echo "  • Test configuration"
    ;;
esac

# Track change location
echo "📍 Changed: $FILE_PATH"
echo "   Lines: $LINES_CHANGED"
```

### Command Execution (Bash)

```bash
# Track git operations
if [[ "$COMMAND" == *"git commit"* ]]; then
  COMMIT_MSG=$(git log -1 --pretty=%B)
  echo "📝 Commit created: ${COMMIT_MSG:0:50}..."
  echo "Suggestions:"
  echo "  • Review: git show"
  echo "  • Push: git push"

  # Suggest doc update for feature commits
  if [[ "$COMMIT_MSG" == *"feat"* ]]; then
    echo "  • Update docs for new feature"
  fi
fi

# Track build operations
if [[ "$COMMAND" == *"build"* ]] || [[ "$COMMAND" == *"compile"* ]]; then
  if [[ "$EXIT_CODE" == "0" ]]; then
    echo "✅ Build successful"
    echo "Suggestions:"
    echo "  • Run tests"
    echo "  • Check output size"
  else
    echo "❌ Build failed"
    echo "Suggestions:"
    echo "  • Check error messages"
    echo "  • Review recent changes"
  fi
fi

# Track test operations
if [[ "$COMMAND" == *"test"* ]]; then
  if [[ "$EXIT_CODE" == "0" ]]; then
    echo "✅ Tests passed"
  else
    echo "❌ Tests failed"
    echo "Suggestions:"
    echo "  • Review failing tests"
    echo "  • Check test output"
  fi
fi
```

## Output Format

### Change Summary
```
📊 Change Tracker
━━━━━━━━━━━━━━━━━
Operation: Edit
File: src/utils.ts
Lines: +15, -3

Suggestions:
• Run lint: npm run lint
• Run tests: npm test
• Commit: /commit --type refactor
```

### Session Summary (on request)
```
📈 Session Changes
━━━━━━━━━━━━━━━━━━
Files Modified: 5
Files Created: 2
Commands Run: 12
Commits Made: 3

Changed Files:
• src/auth.ts (+45, -12)
• src/utils.ts (+8, -2)
• README.md (+20, -0)

Pending Actions:
• 2 files not committed
• Tests not run since last change
```

## Configuration

```json
{
  "hooks": {
    "post-tool-use-tracker": {
      "enabled": true,
      "track_changes": true,
      "suggest_actions": true,
      "show_summary": "on_request",
      "auto_lint_reminder": true,
      "auto_test_reminder": true
    }
  }
}
```

## Integration

### With /commit skill
```
After file changes → Suggest: /commit
After multiple changes → Suggest: /commit --scope {detected}
```

### With /test skill
```
After code change → Suggest: /test run
After new function → Suggest: /test generate
```

### With /review skill
```
After large changes → Suggest: /review --focus quality
After security-related files → Suggest: /review --focus security
```
