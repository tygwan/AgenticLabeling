#!/bin/bash
#
# Error Recovery Hook
# Provides graceful degradation and failure logging for all hooks
#
# Events: PreToolUse, PostToolUse (wrapper for other hooks)
#

set -e

# Configuration
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
LOG_DIR="$PROJECT_ROOT/.claude/logs"
ERROR_LOG="$LOG_DIR/error.log"
RECOVERY_LOG="$LOG_DIR/recovery.log"
MAX_LOG_SIZE=1048576  # 1MB
MAX_LOG_FILES=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# Logging Functions
# ============================================

ensure_log_dir() {
    if [[ ! -d "$LOG_DIR" ]]; then
        mkdir -p "$LOG_DIR"
    fi
}

rotate_logs() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        local size=$(stat -f%z "$log_file" 2>/dev/null || stat --format=%s "$log_file" 2>/dev/null || echo 0)
        if [[ "$size" -gt "$MAX_LOG_SIZE" ]]; then
            for i in $(seq $((MAX_LOG_FILES-1)) -1 1); do
                if [[ -f "${log_file}.$i" ]]; then
                    mv "${log_file}.$i" "${log_file}.$((i+1))"
                fi
            done
            mv "$log_file" "${log_file}.1"
        fi
    fi
}

log_error() {
    ensure_log_dir
    rotate_logs "$ERROR_LOG"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local message="$1"
    local context="$2"
    echo "[$timestamp] ERROR: $message" >> "$ERROR_LOG"
    if [[ -n "$context" ]]; then
        echo "  Context: $context" >> "$ERROR_LOG"
    fi
}

log_recovery() {
    ensure_log_dir
    rotate_logs "$RECOVERY_LOG"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local message="$1"
    local action="$2"
    echo "[$timestamp] RECOVERY: $message" >> "$RECOVERY_LOG"
    if [[ -n "$action" ]]; then
        echo "  Action: $action" >> "$RECOVERY_LOG"
    fi
}

log_info() {
    echo -e "${BLUE}[Recovery]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[Recovery] OK${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[Recovery] WARN${NC} $1"
}

log_failure() {
    echo -e "${RED}[Recovery] FAIL${NC} $1"
}

# ============================================
# Recovery Functions
# ============================================

# Attempt to recover from hook failure
recover_hook_failure() {
    local hook_name="$1"
    local error_msg="$2"
    local recovery_action=""

    case "$hook_name" in
        "phase-progress.sh")
            # Phase progress failure - try to reset progress tracking
            recovery_action="Skipping progress update, manual sync needed"
            log_warning "Phase progress update skipped: $error_msg"
            log_recovery "phase-progress.sh failed" "$recovery_action"
            return 0  # Continue execution
            ;;
        "auto-doc-sync.sh")
            # Doc sync failure - non-critical, continue
            recovery_action="Skipping doc sync, run /agile-sync manually"
            log_warning "Doc sync skipped: $error_msg"
            log_recovery "auto-doc-sync.sh failed" "$recovery_action"
            return 0  # Continue execution
            ;;
        "post-tool-use-tracker.sh")
            # Tracker failure - non-critical, continue
            recovery_action="Skipping change tracking"
            log_warning "Change tracking skipped: $error_msg"
            log_recovery "post-tool-use-tracker.sh failed" "$recovery_action"
            return 0  # Continue execution
            ;;
        "notification-handler.sh")
            # Notification failure - non-critical, continue
            recovery_action="Notification silently skipped"
            log_recovery "notification-handler.sh failed" "$recovery_action"
            return 0  # Continue execution
            ;;
        "pre-tool-use-safety.sh")
            # Safety check failure - CRITICAL, must handle carefully
            log_failure "Safety check failed: $error_msg"
            log_error "pre-tool-use-safety.sh failed" "$error_msg"
            recovery_action="Safety check failed - manual review required"
            log_recovery "pre-tool-use-safety.sh failed" "$recovery_action"
            return 1  # Block execution for safety
            ;;
        *)
            # Unknown hook - log and continue
            recovery_action="Unknown hook failure, continuing"
            log_warning "Unknown hook '$hook_name' failed: $error_msg"
            log_recovery "$hook_name failed" "$recovery_action"
            return 0  # Continue execution
            ;;
    esac
}

# Check system health
check_system_health() {
    local issues=0

    # Check log directory
    if [[ ! -d "$LOG_DIR" ]]; then
        mkdir -p "$LOG_DIR"
        ((issues++))
    fi

    # Check hook files
    local hooks_dir="$PROJECT_ROOT/.claude/hooks"
    if [[ -d "$hooks_dir" ]]; then
        for hook in "$hooks_dir"/*.sh; do
            if [[ -f "$hook" ]]; then
                if [[ ! -x "$hook" ]]; then
                    log_warning "Hook not executable: $(basename "$hook")"
                    ((issues++))
                fi
            fi
        done
    fi

    # Check docs directory
    if [[ ! -d "$PROJECT_ROOT/docs" ]]; then
        log_warning "docs/ directory missing"
        ((issues++))
    fi

    return $issues
}

# Auto-fix common issues
auto_fix() {
    local fixed=0

    # Fix hook permissions
    local hooks_dir="$PROJECT_ROOT/.claude/hooks"
    if [[ -d "$hooks_dir" ]]; then
        for hook in "$hooks_dir"/*.sh; do
            if [[ -f "$hook" ]] && [[ ! -x "$hook" ]]; then
                chmod +x "$hook" 2>/dev/null && {
                    log_success "Fixed permissions: $(basename "$hook")"
                    ((fixed++))
                }
            fi
        done
    fi

    # Create missing directories
    if [[ ! -d "$PROJECT_ROOT/docs" ]]; then
        mkdir -p "$PROJECT_ROOT/docs" && {
            log_success "Created docs/ directory"
            ((fixed++))
        }
    fi

    if [[ ! -d "$LOG_DIR" ]]; then
        mkdir -p "$LOG_DIR" && {
            log_success "Created logs/ directory"
            ((fixed++))
        }
    fi

    return $fixed
}

# Generate recovery report
generate_report() {
    echo ""
    echo "============================================"
    echo " ERROR RECOVERY REPORT"
    echo "============================================"
    echo ""

    # Recent errors
    echo "Recent Errors (last 10):"
    echo "------------------------"
    if [[ -f "$ERROR_LOG" ]]; then
        tail -20 "$ERROR_LOG" | grep "ERROR:" | tail -10
    else
        echo "No errors logged"
    fi
    echo ""

    # Recent recoveries
    echo "Recent Recoveries (last 5):"
    echo "----------------------------"
    if [[ -f "$RECOVERY_LOG" ]]; then
        tail -10 "$RECOVERY_LOG" | grep "RECOVERY:" | tail -5
    else
        echo "No recoveries logged"
    fi
    echo ""

    # System health
    echo "System Health Check:"
    echo "--------------------"
    check_system_health
    local health_status=$?
    if [[ $health_status -eq 0 ]]; then
        echo -e "${GREEN}All systems healthy${NC}"
    else
        echo -e "${YELLOW}$health_status issue(s) found${NC}"
    fi
    echo ""
}

# ============================================
# Main Logic
# ============================================

ACTION="$1"

case "$ACTION" in
    "check")
        check_system_health
        ;;
    "fix")
        auto_fix
        ;;
    "report")
        generate_report
        ;;
    "clear-logs")
        rm -f "$ERROR_LOG" "$RECOVERY_LOG"
        log_success "Logs cleared"
        ;;
    *)
        # Default: wrapper mode for other hooks
        HOOK_NAME="$1"
        HOOK_ARGS="${@:2}"

        if [[ -n "$HOOK_NAME" ]]; then
            # Execute the target hook with error handling
            HOOK_PATH="$PROJECT_ROOT/.claude/hooks/$HOOK_NAME"

            if [[ -f "$HOOK_PATH" ]]; then
                bash "$HOOK_PATH" $HOOK_ARGS 2>&1 || {
                    error_code=$?
                    recover_hook_failure "$HOOK_NAME" "Exit code: $error_code"
                    exit $?
                }
            fi
        fi
        ;;
esac

exit 0
