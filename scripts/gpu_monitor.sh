#!/usr/bin/env bash
# gpu_monitor.sh — Continuous GPU monitor and auto-deployer for s1s2 pipeline.
#
# Watches a RunPod B200 pod, keeps it busy running the job queue from
# run_all_gpu.py. If GPU goes idle for 2 consecutive checks, kicks off the
# next pending job. Designed for "set it and forget it" overnight runs.
#
# Usage:
#   ./scripts/gpu_monitor.sh "ssh root@209.x.x.x -p 12345 -i ~/.ssh/runpod"
#   ./scripts/gpu_monitor.sh --once "ssh root@209.x.x.x -p 12345 -i ~/.ssh/runpod"
#   ./scripts/gpu_monitor.sh --deploy-only "ssh root@209.x.x.x -p 12345 -i ~/.ssh/runpod"
#
# Flags:
#   --once          Single check, no loop
#   --deploy-only   Sync files to pod and exit
#   --interval N    Check interval in seconds (default: 60)
#   --threshold N   GPU util % below which counts as idle (default: 5)
#   --no-sync       Skip initial file sync
set -uo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/workspace/s1s2"
STATE_FILE="/workspace/gpu_pipeline_state.json"
LOG_FILE="/workspace/gpu_pipeline_log.txt"
IDLE_THRESHOLD=5
CHECK_INTERVAL=60
IDLE_CHECKS_NEEDED=2
RETRY_DELAY=30

# ANSI colors (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
    BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; BOLD=''; NC=''
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODE="loop"   # loop | once | deploy-only
DO_SYNC=1
SSH_CMD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --once)
            MODE="once"
            shift
            ;;
        --deploy-only)
            MODE="deploy-only"
            shift
            ;;
        --interval)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        --threshold)
            IDLE_THRESHOLD="$2"
            shift 2
            ;;
        --no-sync)
            DO_SYNC=0
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--once|--deploy-only] [--interval N] [--threshold N] [--no-sync] SSH_CMD"
            echo ""
            echo "  SSH_CMD    Full SSH command string, e.g. \"ssh root@IP -p PORT -i KEY\""
            echo ""
            echo "Flags:"
            echo "  --once          Single check, no monitoring loop"
            echo "  --deploy-only   Sync files to pod and exit"
            echo "  --interval N    Seconds between checks (default: 60)"
            echo "  --threshold N   GPU util % idle threshold (default: 5)"
            echo "  --no-sync       Skip initial file sync"
            exit 0
            ;;
        *)
            if [[ -z "$SSH_CMD" ]]; then
                SSH_CMD="$1"
            else
                echo "ERROR: Unexpected argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$SSH_CMD" ]]; then
    echo "ERROR: SSH command required."
    echo "Usage: $0 [--once|--deploy-only] \"ssh root@IP -p PORT -i KEY\""
    exit 1
fi

# ---------------------------------------------------------------------------
# Derive SCP command from SSH command
# SSH: ssh root@IP -p PORT -i KEY
# SCP: scp -P PORT -i KEY ... root@IP:...
# ---------------------------------------------------------------------------
derive_scp_and_host() {
    local ssh_parts
    # shellcheck disable=SC2206
    ssh_parts=($SSH_CMD)

    local user_host=""
    local scp_opts=()
    local i=1  # skip "ssh"

    while [[ $i -lt ${#ssh_parts[@]} ]]; do
        case "${ssh_parts[$i]}" in
            -p)
                scp_opts+=("-P" "${ssh_parts[$((i+1))]}")
                i=$((i+2))
                ;;
            -i)
                scp_opts+=("-i" "${ssh_parts[$((i+1))]}")
                i=$((i+2))
                ;;
            -o)
                scp_opts+=("-o" "${ssh_parts[$((i+1))]}")
                i=$((i+2))
                ;;
            -*)
                # Pass through other SSH flags
                scp_opts+=("${ssh_parts[$i]}")
                i=$((i+1))
                ;;
            *)
                user_host="${ssh_parts[$i]}"
                i=$((i+1))
                ;;
        esac
    done

    SCP_OPTS="${scp_opts[*]}"
    USER_HOST="$user_host"
}

derive_scp_and_host

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ts() {
    date +"%H:%M:%S"
}

log_info() {
    echo -e "${CYAN}[$(ts)]${NC} $*"
}

log_ok() {
    echo -e "${GREEN}[$(ts)]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[$(ts)]${NC} $*"
}

log_err() {
    echo -e "${RED}[$(ts)]${NC} $*"
}

# Run a command on the pod. Returns the command's exit code.
pod_exec() {
    $SSH_CMD -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$@" 2>/dev/null
}

# SCP files to the pod.
pod_scp() {
    # $1 = local path, $2 = remote path
    # shellcheck disable=SC2086
    scp -o ConnectTimeout=10 -o StrictHostKeyChecking=no $SCP_OPTS -r "$1" "${USER_HOST}:$2" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Sync files to pod
# ---------------------------------------------------------------------------
sync_files() {
    log_info "${BOLD}Syncing files to pod...${NC}"

    # Ensure remote directories exist
    pod_exec "mkdir -p ${REMOTE_DIR}/data/benchmark ${REMOTE_DIR}/scripts ${REMOTE_DIR}/src ${REMOTE_DIR}/configs"

    # Sync benchmark data
    local bench_file="${PROJECT_ROOT}/data/benchmark/benchmark.jsonl"
    if [[ -f "$bench_file" ]]; then
        log_info "  benchmark.jsonl -> ${REMOTE_DIR}/data/benchmark/"
        pod_scp "$bench_file" "${REMOTE_DIR}/data/benchmark/"
    else
        log_warn "  benchmark.jsonl not found locally, skipping"
    fi

    # Sync all Python scripts
    log_info "  scripts/*.py -> ${REMOTE_DIR}/scripts/"
    for f in "${PROJECT_ROOT}"/scripts/*.py; do
        [[ -f "$f" ]] && pod_scp "$f" "${REMOTE_DIR}/scripts/"
    done

    # Sync configs
    if [[ -d "${PROJECT_ROOT}/configs" ]]; then
        log_info "  configs/ -> ${REMOTE_DIR}/configs/"
        pod_scp "${PROJECT_ROOT}/configs/" "${REMOTE_DIR}/"
    fi

    # Sync src
    if [[ -d "${PROJECT_ROOT}/src" ]]; then
        log_info "  src/ -> ${REMOTE_DIR}/src/"
        pod_scp "${PROJECT_ROOT}/src/" "${REMOTE_DIR}/"
    fi

    log_ok "File sync complete."
}

# ---------------------------------------------------------------------------
# GPU status check
# ---------------------------------------------------------------------------
get_gpu_status() {
    # Returns: "util_pct,mem_used_mb,mem_total_mb" or empty on failure
    local raw
    raw=$(pod_exec "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits" 2>/dev/null)
    if [[ $? -ne 0 ]] || [[ -z "$raw" ]]; then
        echo ""
        return 1
    fi
    # nvidia-smi returns e.g. "85, 45200, 81920" — strip spaces
    echo "$raw" | tr -d ' '
}

# ---------------------------------------------------------------------------
# Pipeline state check
# ---------------------------------------------------------------------------
get_pipeline_state() {
    # Returns JSON content of the state file, or empty
    pod_exec "cat ${STATE_FILE} 2>/dev/null" || echo ""
}

get_current_job() {
    # Check if run_all_gpu.py is running and extract what it's doing
    local ps_out
    ps_out=$(pod_exec "ps aux | grep 'run_all_gpu.py' | grep -v grep" 2>/dev/null || true)

    if [[ -n "$ps_out" ]]; then
        # Running — try to get the current job from the log tail
        local log_tail
        log_tail=$(pod_exec "tail -20 ${LOG_FILE} 2>/dev/null" || true)

        # Look for the most recent JOB line
        local current_job
        current_job=$(echo "$log_tail" | grep -o 'JOB [0-9]*/[0-9]*: .* \[.*\]' | tail -1 || true)
        if [[ -n "$current_job" ]]; then
            # Extract job name from brackets
            local job_name
            job_name=$(echo "$current_job" | grep -o '\[.*\]' | tr -d '[]')
            echo "running:${job_name}"
        else
            echo "running:unknown"
        fi
    else
        echo "idle"
    fi
}

get_completed_jobs() {
    # Parse state file for completed job names
    local state_json
    state_json=$(get_pipeline_state)
    if [[ -z "$state_json" ]]; then
        echo ""
        return
    fi
    # Extract completed job names using python (more reliable than jq which may not be installed)
    echo "$state_json" | python3 -c "
import json, sys
try:
    state = json.load(sys.stdin)
    completed = list(state.get('completed', {}).keys())
    print(','.join(completed))
except:
    pass
" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Start pipeline on pod
# ---------------------------------------------------------------------------
start_pipeline() {
    log_info "${BOLD}Starting run_all_gpu.py on pod...${NC}"

    # Kill any orphaned python processes from prior runs (defensive)
    pod_exec "pkill -f 'python.*run_all_gpu' 2>/dev/null || true"
    sleep 2

    # Start the pipeline in the background via nohup
    pod_exec "cd ${REMOTE_DIR} && nohup python scripts/run_all_gpu.py > ${LOG_FILE} 2>&1 &"

    if [[ $? -eq 0 ]]; then
        log_ok "Pipeline started. Log: ${LOG_FILE}"
    else
        log_err "Failed to start pipeline."
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Show log tail
# ---------------------------------------------------------------------------
show_log_tail() {
    local lines="${1:-5}"
    local tail_out
    tail_out=$(pod_exec "tail -${lines} ${LOG_FILE} 2>/dev/null" || true)
    if [[ -n "$tail_out" ]]; then
        echo -e "${BLUE}--- Log tail ---${NC}"
        echo "$tail_out"
        echo -e "${BLUE}----------------${NC}"
    fi
}

# ---------------------------------------------------------------------------
# Format GPU status line
# ---------------------------------------------------------------------------
format_status() {
    local util="$1"
    local mem_used="$2"
    local mem_total="$3"
    local job_status="$4"

    # Convert MiB to GiB
    local mem_used_gb mem_total_gb
    mem_used_gb=$(echo "scale=1; $mem_used / 1024" | bc 2>/dev/null || echo "$mem_used")
    mem_total_gb=$(echo "scale=1; $mem_total / 1024" | bc 2>/dev/null || echo "$mem_total")

    # Color the utilization
    local util_color
    if [[ "$util" -ge 50 ]]; then
        util_color="${GREEN}"
    elif [[ "$util" -ge 10 ]]; then
        util_color="${YELLOW}"
    else
        util_color="${RED}"
    fi

    printf "[$(ts)] GPU: ${util_color}%3d%%${NC} util | %5s/%5s GB VRAM | Job: %s\n" \
        "$util" "$mem_used_gb" "$mem_total_gb" "$job_status"
}

# ---------------------------------------------------------------------------
# Single check iteration
# ---------------------------------------------------------------------------
idle_count=0
last_completed=""

do_check() {
    # 1. Get GPU status
    local gpu_raw
    gpu_raw=$(get_gpu_status)
    if [[ -z "$gpu_raw" ]]; then
        log_err "Connection failed. Retrying in ${RETRY_DELAY}s..."
        sleep "$RETRY_DELAY"
        return 1
    fi

    local util mem_used mem_total
    util=$(echo "$gpu_raw" | cut -d',' -f1)
    mem_used=$(echo "$gpu_raw" | cut -d',' -f2)
    mem_total=$(echo "$gpu_raw" | cut -d',' -f3)

    # 2. Get job status
    local job_status
    job_status=$(get_current_job)

    local job_display
    if [[ "$job_status" == running:* ]]; then
        local job_name="${job_status#running:}"
        job_display="${job_name} (running)"
    else
        job_display="idle"
    fi

    # 3. Print status
    format_status "$util" "$mem_used" "$mem_total" "$job_display"

    # 4. Check if idle and needs a new job
    if [[ "$util" -lt "$IDLE_THRESHOLD" ]]; then
        idle_count=$((idle_count + 1))
        if [[ $idle_count -ge $IDLE_CHECKS_NEEDED ]]; then
            # Check what's completed
            local completed
            completed=$(get_completed_jobs)

            # All 6 jobs from the registry
            local all_jobs="sae_goodfire,new_items,olmo3_full,reextract_activations,attention_entropy,bootstrap_cis"

            # Check if everything is done
            local all_done=1
            IFS=',' read -ra all_arr <<< "$all_jobs"
            for job in "${all_arr[@]}"; do
                if [[ ! ",$completed," == *",$job,"* ]]; then
                    all_done=0
                    break
                fi
            done

            if [[ "$all_done" -eq 1 ]]; then
                log_ok "${BOLD}All pipeline jobs completed!${NC}"
                show_log_tail 10
                return 2  # Signal: all done
            fi

            # Not all done and GPU is idle — start the pipeline
            log_warn "GPU idle for ${idle_count} consecutive checks. Completed: [${completed:-none}]"

            # Figure out what's next
            local next_job="unknown"
            for job in "${all_arr[@]}"; do
                if [[ ! ",$completed," == *",$job,"* ]]; then
                    next_job="$job"
                    break
                fi
            done

            format_status "$util" "$mem_used" "$mem_total" "${last_completed:-idle} (DONE) -> Starting ${next_job}"
            start_pipeline
            idle_count=0
        fi
    else
        idle_count=0
    fi

    # 5. Show log tail every 5th check (when in loop mode)
    if [[ "$job_status" == running:* ]]; then
        show_log_tail 3
    fi

    return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo -e "${BOLD}========================================${NC}"
    echo -e "${BOLD}  s1s2 GPU Monitor${NC}"
    echo -e "${BOLD}========================================${NC}"
    echo -e "  SSH:       ${SSH_CMD}"
    echo -e "  Mode:      ${MODE}"
    echo -e "  Interval:  ${CHECK_INTERVAL}s"
    echo -e "  Threshold: <${IDLE_THRESHOLD}% for ${IDLE_CHECKS_NEEDED} checks"
    echo -e "${BOLD}========================================${NC}"
    echo ""

    # Test connection first
    log_info "Testing SSH connection..."
    if ! pod_exec "echo ok" >/dev/null 2>&1; then
        log_err "Cannot connect to pod. Check your SSH command."
        exit 1
    fi
    log_ok "Connection OK."

    # Sync files (unless --no-sync)
    if [[ "$DO_SYNC" -eq 1 ]]; then
        sync_files
    fi

    # Deploy-only mode: exit after sync
    if [[ "$MODE" == "deploy-only" ]]; then
        log_ok "Deploy complete. Exiting."
        exit 0
    fi

    # Check if pipeline is already running
    local initial_job
    initial_job=$(get_current_job)
    if [[ "$initial_job" == idle ]]; then
        log_info "No pipeline running. Will start on first idle detection."
        # Start immediately — no reason to wait
        log_info "Starting pipeline now..."
        start_pipeline
    else
        log_info "Pipeline already running: ${initial_job}"
    fi

    echo ""

    # Once mode: single check
    if [[ "$MODE" == "once" ]]; then
        sleep 5  # Brief pause to let the pipeline initialize
        do_check
        exit $?
    fi

    # Loop mode: continuous monitoring
    log_info "Entering monitoring loop (Ctrl+C to stop)..."
    echo ""

    trap 'echo ""; log_info "Monitor stopped."; exit 0' INT TERM

    while true; do
        do_check
        local rc=$?
        if [[ $rc -eq 2 ]]; then
            # All jobs done
            log_ok "${BOLD}All jobs complete. Monitor exiting.${NC}"
            exit 0
        fi
        sleep "$CHECK_INTERVAL"
    done
}

main
