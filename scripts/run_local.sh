#!/bin/bash
# Local development script - runs core services without Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo -e "${GREEN}=== AgenticLabeling Local Runner ===${NC}"
echo "Project directory: $PROJECT_DIR"

# Create log directory
mkdir -p logs

# Function to run a service
run_service() {
    local name=$1
    local port=$2
    local dir=$3

    echo -e "${YELLOW}Starting $name on port $port...${NC}"
    cd "$PROJECT_DIR/services/$dir"

    # Kill existing process on port
    lsof -ti:$port | xargs kill -9 2>/dev/null || true

    # Start service
    uvicorn app.main:app --host 0.0.0.0 --port $port --reload \
        > "$PROJECT_DIR/logs/$name.log" 2>&1 &

    echo "$!" > "$PROJECT_DIR/logs/$name.pid"
    cd "$PROJECT_DIR"
}

# Function to stop all services
stop_all() {
    echo -e "${RED}Stopping all services...${NC}"
    for pidfile in logs/*.pid; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            kill $pid 2>/dev/null || true
            rm "$pidfile"
        fi
    done
    echo "All services stopped."
}

# Function to check service health
check_health() {
    local name=$1
    local port=$2

    sleep 2
    if curl -s "http://localhost:$port/health" > /dev/null 2>&1 || \
       curl -s "http://localhost:$port/" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name is healthy${NC}"
        return 0
    else
        echo -e "${RED}❌ $name failed to start${NC}"
        return 1
    fi
}

case "${1:-start}" in
    start)
        echo -e "${GREEN}Starting core services...${NC}"

        # Start services in order
        run_service "object-registry" 8010 "object-registry"
        sleep 2

        run_service "data-manager" 8006 "data-manager"
        run_service "gateway" 8000 "gateway"
        run_service "preprocessing-agent" 8008 "preprocessing-agent"

        sleep 3

        echo ""
        echo -e "${GREEN}=== Service Status ===${NC}"
        check_health "object-registry" 8010
        check_health "data-manager" 8006
        check_health "gateway" 8000
        check_health "preprocessing-agent" 8008

        echo ""
        echo -e "${GREEN}Services started! View logs in logs/ directory${NC}"
        echo "Gateway API: http://localhost:8000"
        echo "API Docs: http://localhost:8000/docs"
        echo ""
        echo "To stop: ./scripts/run_local.sh stop"
        ;;

    stop)
        stop_all
        ;;

    status)
        echo -e "${GREEN}=== Service Status ===${NC}"
        for pidfile in logs/*.pid; do
            if [ -f "$pidfile" ]; then
                name=$(basename "$pidfile" .pid)
                pid=$(cat "$pidfile")
                if ps -p $pid > /dev/null 2>&1; then
                    echo -e "${GREEN}✅ $name (PID: $pid) running${NC}"
                else
                    echo -e "${RED}❌ $name (PID: $pid) not running${NC}"
                fi
            fi
        done
        ;;

    logs)
        service="${2:-gateway}"
        tail -f "logs/$service.log"
        ;;

    *)
        echo "Usage: $0 {start|stop|status|logs [service]}"
        exit 1
        ;;
esac
