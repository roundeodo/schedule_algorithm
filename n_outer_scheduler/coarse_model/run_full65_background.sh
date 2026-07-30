#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
IDEA_MODEL_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
RESULT_DIR="$SCRIPT_DIR/results"
RESULT_JSON="$RESULT_DIR/final_policy_65_symmetric2_fixed_first.json"
SUMMARY_JSON="$RESULT_DIR/final_policy_65_symmetric2_fixed_first_summary.json"
LOG_FILE="$RESULT_DIR/final_policy_65_symmetric2_fixed_first.log"
PID_FILE="$RESULT_DIR/final_policy_65_symmetric2_fixed_first.pid"

status() {
    if [[ -f "$PID_FILE" ]]; then
        pid="$(tr -d '[:space:]' < "$PID_FILE")"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "RUNNING pid=$pid"
            tail -n 12 "$LOG_FILE" 2>/dev/null || true
            return 0
        fi
    fi
    if [[ -f "$SUMMARY_JSON" ]]; then
        echo "FINISHED summary=$SUMMARY_JSON"
        tail -n 12 "$LOG_FILE" 2>/dev/null || true
        return 0
    fi
    if [[ -f "$LOG_FILE" ]]; then
        echo "STOPPED_OR_FAILED log=$LOG_FILE"
        tail -n 20 "$LOG_FILE" 2>/dev/null || true
        return 1
    fi
    echo "NOT_STARTED"
}

worker() {
    cd "$IDEA_MODEL_DIR"
    python3 -u -m n_outer_scheduler.coarse_model.run_final_policy_eval \
        --calibrate-main-mode-ranking \
        --service-order-ablation \
        --output "$RESULT_JSON"
    python3 -u -m n_outer_scheduler.coarse_model.analyze_final_policy_eval \
        --input "$RESULT_JSON" \
        --output "$SUMMARY_JSON"
}

command="${1:-status}"
case "$command" in
    start)
        mkdir -p "$RESULT_DIR"
        if [[ -f "$PID_FILE" ]]; then
            old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
            if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
                echo "ALREADY_RUNNING pid=$old_pid"
                exit 1
            fi
        fi
        rm -f "$RESULT_JSON" "$SUMMARY_JSON" "$PID_FILE"
        setsid nohup bash "$SCRIPT_DIR/run_full65_background.sh" __worker \
            > "$LOG_FILE" 2>&1 < /dev/null &
        pid="$!"
        printf '%s\n' "$pid" > "$PID_FILE"
        echo "STARTED pid=$pid"
        echo "monitor: bash $0 status"
        ;;
    status)
        status
        ;;
    __worker)
        worker
        ;;
    *)
        echo "usage: bash $0 {start|status}" >&2
        exit 2
        ;;
esac
