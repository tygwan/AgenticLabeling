#!/bin/bash
# Dashboard runner script
# This script helps with running the dashboard and testing its functionality

set -e

# Directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${2}${1}${NC}"
}

# Function to show help message
show_help() {
    echo "Dashboard Runner Script"
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start                Start the dashboard server"
    echo "  start-bg             Start the dashboard server in background"
    echo "  stop                 Stop the dashboard server running in background"
    echo "  status               Check if the dashboard server is running"
    echo "  test                 Run API tests against the dashboard"
    echo "  test [category]      Run API tests for specific category"
    echo "  setup                Create necessary directories for dashboard"
    echo "  help                 Show this help message"
    echo ""
}

# Function to check if required packages are installed
check_requirements() {
    print_message "Checking requirements..." "$BLUE"
    
    # Check for Python packages
    required_packages=("fastapi" "uvicorn" "jinja2" "python-multipart" "requests")
    missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_message "Missing required Python packages: ${missing_packages[*]}" "$YELLOW"
        read -p "Would you like to install them now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_message "Installing packages..." "$BLUE"
            pip install "${missing_packages[@]}"
        else
            print_message "Please install required packages manually." "$RED"
            exit 1
        fi
    else
        print_message "All required packages are installed." "$GREEN"
    fi
}

# Function to set up the dashboard environment
setup_dashboard() {
    print_message "Setting up dashboard environment..." "$BLUE"
    
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT/dashboard_static/css"
    mkdir -p "$PROJECT_ROOT/dashboard_static/js"
    mkdir -p "$PROJECT_ROOT/dashboard_static/images"
    mkdir -p "$PROJECT_ROOT/dashboard_static/templates"
    
    print_message "Dashboard environment set up." "$GREEN"
}

# Function to start the dashboard
start_dashboard() {
    check_requirements
    
    print_message "Starting dashboard server..." "$BLUE"
    
    if [ "$1" = "background" ]; then
        # Start in background
        nohup python3 "$SCRIPT_DIR/dashboard_prototype.py" > "$PROJECT_ROOT/dashboard.log" 2>&1 &
        echo $! > "$PROJECT_ROOT/.dashboard.pid"
        print_message "Dashboard server started in background (PID: $(cat "$PROJECT_ROOT/.dashboard.pid"))." "$GREEN"
        print_message "Logs available at: $PROJECT_ROOT/dashboard.log" "$BLUE"
        print_message "Access the dashboard at: http://localhost:8000" "$BLUE"
        print_message "API documentation at: http://localhost:8000/api/docs" "$BLUE"
    else
        # Start in foreground
        print_message "Starting dashboard server in foreground. Press Ctrl+C to stop." "$YELLOW"
        python3 "$SCRIPT_DIR/dashboard_prototype.py"
    fi
}

# Function to stop the dashboard
stop_dashboard() {
    if [ -f "$PROJECT_ROOT/.dashboard.pid" ]; then
        PID=$(cat "$PROJECT_ROOT/.dashboard.pid")
        if ps -p $PID > /dev/null; then
            print_message "Stopping dashboard server (PID: $PID)..." "$BLUE"
            kill $PID
            rm "$PROJECT_ROOT/.dashboard.pid"
            print_message "Dashboard server stopped." "$GREEN"
        else
            print_message "Dashboard server is not running (stale PID file)." "$YELLOW"
            rm "$PROJECT_ROOT/.dashboard.pid"
        fi
    else
        print_message "Dashboard server is not running." "$YELLOW"
    fi
}

# Function to check dashboard status
check_status() {
    if [ -f "$PROJECT_ROOT/.dashboard.pid" ]; then
        PID=$(cat "$PROJECT_ROOT/.dashboard.pid")
        if ps -p $PID > /dev/null; then
            print_message "Dashboard server is running (PID: $PID)." "$GREEN"
            print_message "Access the dashboard at: http://localhost:8000" "$BLUE"
            print_message "API documentation at: http://localhost:8000/api/docs" "$BLUE"
            return 0
        else
            print_message "Dashboard server is not running (stale PID file)." "$YELLOW"
            rm "$PROJECT_ROOT/.dashboard.pid"
            return 1
        fi
    else
        print_message "Dashboard server is not running." "$YELLOW"
        return 1
    fi
}

# Function to run tests
run_tests() {
    print_message "Running dashboard API tests..." "$BLUE"
    
    # Check if server is running first
    if ! curl -s http://localhost:8000/api > /dev/null; then
        print_message "Dashboard server is not running. Start it before running tests." "$RED"
        exit 1
    fi
    
    # Run tests (with optional category argument)
    if [ -n "$1" ]; then
        print_message "Testing category: $1" "$BLUE"
        python3 "$SCRIPT_DIR/test_dashboard_api.py" "$1"
    else
        python3 "$SCRIPT_DIR/test_dashboard_api.py"
    fi
    
    if [ $? -eq 0 ]; then
        print_message "Tests completed successfully." "$GREEN"
    else
        print_message "Tests failed. Check output for details." "$RED"
        exit 1
    fi
}

# Main command handler
case "$1" in
    start)
        start_dashboard
        ;;
    start-bg)
        start_dashboard "background"
        ;;
    stop)
        stop_dashboard
        ;;
    status)
        check_status
        ;;
    test)
        run_tests "$2"
        ;;
    setup)
        setup_dashboard
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        if [ -z "$1" ]; then
            show_help
        else
            print_message "Unknown command: $1" "$RED"
            show_help
            exit 1
        fi
        ;;
esac

exit 0 