---
name: notification-handler
event: Notification
tools: []
description: Handle system notifications and provide contextual responses
---

# Notification Handler Hook

## Purpose
Process system notifications and provide helpful context-aware responses.

## Trigger Conditions
Activates on system notifications:
- Build completion/failure
- Test results
- Deployment status
- Error alerts
- Long-running task completion

## Notification Types

### Build Notifications

```bash
# Build success
if [[ "$NOTIFICATION_TYPE" == "build" ]] && [[ "$STATUS" == "success" ]]; then
  echo "✅ Build Complete"
  echo "━━━━━━━━━━━━━━━━"
  echo "Duration: $BUILD_TIME"
  echo "Output: $OUTPUT_PATH"
  echo ""
  echo "Next steps:"
  echo "  • Run tests: npm test"
  echo "  • Deploy: npm run deploy"
fi

# Build failure
if [[ "$NOTIFICATION_TYPE" == "build" ]] && [[ "$STATUS" == "failure" ]]; then
  echo "❌ Build Failed"
  echo "━━━━━━━━━━━━━━━"
  echo "Error: $ERROR_MESSAGE"
  echo ""
  echo "Troubleshooting:"
  echo "  • Check syntax errors"
  echo "  • Verify dependencies"
  echo "  • Review recent changes"
fi
```

### Test Notifications

```bash
# Test results
if [[ "$NOTIFICATION_TYPE" == "test" ]]; then
  echo "🧪 Test Results"
  echo "━━━━━━━━━━━━━━━"
  echo "Passed: $TESTS_PASSED"
  echo "Failed: $TESTS_FAILED"
  echo "Coverage: $COVERAGE%"

  if [[ "$TESTS_FAILED" -gt 0 ]]; then
    echo ""
    echo "Failed tests:"
    echo "$FAILED_TEST_LIST"
    echo ""
    echo "Actions:"
    echo "  • Review: /test run --verbose"
    echo "  • Debug: /troubleshoot test failures"
  fi
fi
```

### Error Notifications

```bash
# Runtime error
if [[ "$NOTIFICATION_TYPE" == "error" ]]; then
  echo "🚨 Error Detected"
  echo "━━━━━━━━━━━━━━━━"
  echo "Type: $ERROR_TYPE"
  echo "Message: $ERROR_MESSAGE"
  echo "Location: $ERROR_LOCATION"
  echo ""

  # Contextual suggestions
  case "$ERROR_TYPE" in
    "SyntaxError")
      echo "Fix: Check syntax at $ERROR_LOCATION"
      ;;
    "TypeError")
      echo "Fix: Verify type compatibility"
      ;;
    "ReferenceError")
      echo "Fix: Check variable/import declarations"
      ;;
    "NetworkError")
      echo "Fix: Check network connectivity and endpoints"
      ;;
    *)
      echo "Fix: Review error message and stack trace"
      ;;
  esac
fi
```

### Long Task Notifications

```bash
# Task completion
if [[ "$NOTIFICATION_TYPE" == "task_complete" ]]; then
  echo "✅ Task Complete"
  echo "━━━━━━━━━━━━━━━"
  echo "Task: $TASK_NAME"
  echo "Duration: $TASK_DURATION"
  echo "Result: $TASK_RESULT"

  # Play sound or system notification if configured
  if [[ "$NOTIFICATION_SOUND" == "true" ]]; then
    # Platform-specific notification
    case "$OSTYPE" in
      darwin*)
        osascript -e 'display notification "Task complete" with title "Claude Code"'
        ;;
      linux*)
        notify-send "Claude Code" "Task complete"
        ;;
      msys*|cygwin*|win*)
        powershell -c "[console]::beep(1000,500)"
        ;;
    esac
  fi
fi
```

## Output Templates

### Success Template
```
✅ {Operation} Complete
━━━━━━━━━━━━━━━━━━━━━━━
{Details}

Next steps:
• {Suggestion 1}
• {Suggestion 2}
```

### Failure Template
```
❌ {Operation} Failed
━━━━━━━━━━━━━━━━━━━━━
Error: {Error message}

Troubleshooting:
• {Step 1}
• {Step 2}

Need help? Try: /troubleshoot {context}
```

### Warning Template
```
⚠️ {Warning Type}
━━━━━━━━━━━━━━━━
{Warning message}

Action required:
• {Action}
```

## Configuration

```json
{
  "hooks": {
    "notification-handler": {
      "enabled": true,
      "sound_enabled": false,
      "system_notifications": true,
      "notification_types": [
        "build",
        "test",
        "error",
        "task_complete"
      ],
      "quiet_mode": false,
      "verbose_errors": true
    }
  }
}
```

## Integration

### With IDE
- VS Code: Use terminal notifications
- Terminal: Use bell character
- Desktop: Use system notifications

### With CI/CD
- Receive webhook notifications
- Parse CI status updates
- Suggest fixes for failures

### With Monitoring
- Process alert notifications
- Provide contextual troubleshooting
- Suggest monitoring dashboards
