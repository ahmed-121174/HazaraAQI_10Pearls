#!/bin/bash
# Path and execution script
# Usage:
#   ./run.sh start      -> Start server
#   ./run.sh stop       -> Stop server
#   ./run.sh pipeline   -> Run ML pipeline
#   ./run.sh status     -> Check status
#   ./run.sh restart    -> Restart server

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/venv/bin/activate"
PID_FILE="$PROJECT_DIR/.server.pid"
PORT=8000

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

activate_venv() {
    if [ -f "$VENV" ]; then
        source "$VENV"
    else
        echo -e "${RED}Error: Virtual environment not found at $VENV${NC}"
        echo "Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
}

is_running() {
    fuser $PORT/tcp >/dev/null 2>&1
}

start_server() {
    echo -e "${GREEN}  Hazara Division AQI - Starting Dashboard${NC}"

    # Check if already running
    if is_running; then
        echo -e "${YELLOW}Server is already running on port $PORT${NC}"
        echo -e "Dashboard: ${GREEN}http://localhost:$PORT${NC}"
        return
    fi

    activate_venv
    cd "$PROJECT_DIR"

    # Start uvicorn in background
    nohup uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload \
        > "$PROJECT_DIR/server.log" 2>&1 &

    # Poll until server responds (up to 15s)
    for i in $(seq 1 15); do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/ 2>/dev/null)
        [ "$HTTP_CODE" = "200" ] && break
        sleep 1
    done

    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}Server started successfully!${NC}"
        echo -e "Dashboard: ${GREEN}http://localhost:$PORT${NC}"
        echo -e "API Docs:  ${GREEN}http://localhost:$PORT/docs${NC}"
        echo ""
        echo "To stop:   ./run.sh stop"
    else
        echo -e "${RED}Failed to start server. Check server.log${NC}"
    fi
}

stop_server() {
    echo -e "${YELLOW}  Hazara Division AQI - Stopping Dashboard${NC}"

    if is_running; then
        fuser -k $PORT/tcp >/dev/null 2>&1
        sleep 2
        echo -e "${GREEN}Server stopped${NC}"
    else
        echo -e "${YELLOW}Server was not running${NC}"
    fi
}

run_pipeline() {
    echo -e "${GREEN}  Hazara Division AQI - Running Pipeline${NC}"
    echo ""
    echo "This will:"
    echo "  1. Fetch data from Open-Meteo API (6 districts)"
    echo "  2. Clean and preprocess data"
    echo "  3. Engineer 13 features"
    echo "  4. Train 5 ML models + LSTM deep learning model"
    echo "  5. Save best model to models/"
    echo "  6. Upload to Hopsworks Feature Store (if configured)"
    echo ""

    activate_venv
    cd "$PROJECT_DIR"
    TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0 python run_pipeline.py
}

check_status() {
    echo -e "${GREEN}  Hazara Division AQI - Status Check${NC}"

    if is_running; then
        echo -e "Server:    ${GREEN}RUNNING${NC} on port $PORT"
        echo -e "Dashboard: ${GREEN}http://localhost:$PORT${NC}"

        # Test API
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/api/current 2>/dev/null)
        if [ "$HTTP_CODE" = "200" ]; then
            echo -e "API:       ${GREEN}HEALTHY (HTTP $HTTP_CODE)${NC}"
        else
            echo -e "API:       ${RED}ERROR (HTTP $HTTP_CODE)${NC}"
        fi
    else
        echo -e "Server:    ${RED}NOT RUNNING${NC}"
        echo "To start:  ./run.sh start"
    fi

    echo ""

    # Check model files
    echo "Model Artifacts:"
    [ -f "$PROJECT_DIR/models/model.pkl" ] && echo -e "  ${GREEN}✅ model.pkl${NC}" || echo -e "  ${RED}❌ model.pkl missing${NC}"
    [ -f "$PROJECT_DIR/models/lstm_model.keras" ] && echo -e "  ${GREEN}✅ lstm_model.keras${NC}" || echo -e "  ${RED}❌ lstm_model.keras missing${NC}"
    [ -f "$PROJECT_DIR/models/training_results.csv" ] && echo -e "  ${GREEN}✅ training_results.csv${NC}" || echo -e "  ${RED}❌ training_results.csv missing${NC}"

    echo ""
    echo "Data Files:"
    [ -f "$PROJECT_DIR/data/raw_hazara_aqi.csv" ] && echo -e "  ${GREEN}✅ raw_hazara_aqi.csv${NC}" || echo -e "  ${RED}❌ raw data missing${NC}"
    [ -f "$PROJECT_DIR/data/cleaned_hazara_aqi.csv" ] && echo -e "  ${GREEN}✅ cleaned_hazara_aqi.csv${NC}" || echo -e "  ${RED}❌ cleaned data missing${NC}"
}

# Main entrypoint
case "${1}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 1
        start_server
        ;;
    pipeline)
        run_pipeline
        ;;
    status)
        check_status
        ;;
    *)
        echo "  Hazara Division AQI - Command Reference"
        echo ""
        echo "Usage: ./run.sh <command>"
        echo ""
        echo "Commands:"
        echo "  start      Start the FastAPI dashboard (port $PORT)"
        echo "  stop       Stop the running dashboard"
        echo "  restart    Restart the dashboard"
        echo "  pipeline   Run the full ML training pipeline"
        echo "  status     Check server and model status"
        echo ""
        echo "Examples:"
        echo "  ./run.sh pipeline   # Train models first"
        echo "  ./run.sh start      # Then start the dashboard"
        echo "  ./run.sh stop       # Stop when done"
        ;;
esac
