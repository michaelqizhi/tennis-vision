#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Tennis Vision — Harness Script
# Orchestrates: Planner → Generator → Evaluator loop
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/state"
LOG_DIR="$PROJECT_DIR/logs"
AGENTS_DIR="$PROJECT_DIR/.github/agents"

# Config
MAX_SPRINTS="${MAX_SPRINTS:-8}"
RESUME_FROM="${RESUME_FROM:-0}"  # 0 = start from planner
COPILOT_CMD="${COPILOT_CMD:-copilot}"
COPILOT_BASE_FLAGS="--allow-all-tools --allow-all-paths --output-format text"
PLANNER_FLAGS="$COPILOT_BASE_FLAGS --model claude-opus-4.6 --effort high"
GENERATOR_FLAGS="$COPILOT_BASE_FLAGS --model claude-opus-4.6 --effort medium"
EVALUATOR_FLAGS="$COPILOT_BASE_FLAGS --model claude-opus-4.6 --effort medium"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[harness]${NC} $(date '+%H:%M:%S') $*"; }
ok()  { echo -e "${GREEN}[✓]${NC} $*"; }
err() { echo -e "${RED}[✗]${NC} $*"; }
warn(){ echo -e "${YELLOW}[!]${NC} $*"; }

mkdir -p "$STATE_DIR" "$LOG_DIR"

# ============================================================
# Phase 0: Preflight
# ============================================================
preflight() {
    log "Preflight checks..."

    if ! command -v "$COPILOT_CMD" &>/dev/null; then
        err "Copilot CLI not found. Install: npm install -g @githubnext/github-copilot-cli"
        exit 1
    fi

    if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
        err "Python not found"
        exit 1
    fi

    if command -v nvidia-smi &>/dev/null; then
        ok "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    else
        warn "No GPU detected — ML inference will be slow"
    fi

    ok "Preflight passed"
}

# ============================================================
# Phase 1: Planner
# ============================================================
run_planner() {
    if [[ -f "$STATE_DIR/spec.md" && "$RESUME_FROM" -gt 0 ]]; then
        log "spec.md exists, skipping planner (resume mode)"
        return 0
    fi

    log "Running Planner..."

    local prompt="You are the Planner agent. Read the file .github/agents/planner.agent.md for your role definition, then read AGENTS.md for the project spec. Follow your instructions to produce state/spec.md."

    cd "$PROJECT_DIR"
    $COPILOT_CMD -p "$prompt" $PLANNER_FLAGS \
        2>&1 | tee "$LOG_DIR/00-planner.log"

    if [[ -f "$STATE_DIR/spec.md" ]]; then
        ok "Planner complete — spec.md written"
    else
        err "Planner did not produce spec.md"
        exit 1
    fi
}

# ============================================================
# Phase 2-3: Generator → Evaluator Loop
# ============================================================
run_sprint() {
    local sprint=$1

    log "========== Sprint $sprint =========="

    # --- Generator ---
    log "Running Generator (Sprint $sprint)..."

    local gen_prompt="You are the Generator agent. Read .github/agents/generator.agent.md for your role, then AGENTS.md for conventions, then state/spec.md for the plan. This is Sprint $sprint. Read state/checkpoint.md and state/feedback.md if they exist. Implement Sprint $sprint's deliverables and update state/checkpoint.md when done."

    cd "$PROJECT_DIR"
    $COPILOT_CMD -p "$gen_prompt" $GENERATOR_FLAGS \
        2>&1 | tee "$LOG_DIR/sprint-${sprint}-generator.log"

    if [[ -f "$STATE_DIR/checkpoint.md" ]]; then
        ok "Generator Sprint $sprint complete"
    else
        err "Generator did not update checkpoint.md"
        return 1
    fi

    # --- Evaluator ---
    log "Running Evaluator (Sprint $sprint)..."

    local eval_prompt="You are the Evaluator agent. Read .github/agents/evaluator.agent.md for your role. Then read AGENTS.md and state/checkpoint.md. DO NOT read state/spec.md. Evaluate the current state of the project — try to run the code, test features, check quality. Write your findings to state/feedback.md."

    cd "$PROJECT_DIR"
    $COPILOT_CMD -p "$eval_prompt" $EVALUATOR_FLAGS \
        2>&1 | tee "$LOG_DIR/sprint-${sprint}-evaluator.log"

    if [[ -f "$STATE_DIR/feedback.md" ]]; then
        ok "Evaluator Sprint $sprint complete"
    else
        warn "Evaluator did not write feedback.md — continuing anyway"
    fi

    log "Sprint $sprint done"
}

# ============================================================
# Main
# ============================================================
main() {
    echo ""
    echo "============================================"
    echo "  Tennis Vision — Automated Build Harness"
    echo "============================================"
    echo ""

    preflight

    if [[ "$RESUME_FROM" -eq 0 ]]; then
        run_planner
    else
        log "Resuming from Sprint $RESUME_FROM"
    fi

    local start_sprint=$((RESUME_FROM > 0 ? RESUME_FROM : 1))

    for ((i = start_sprint; i <= MAX_SPRINTS; i++)); do
        if ! run_sprint "$i"; then
            err "Sprint $i failed. Resume with: RESUME_FROM=$i bash scripts/harness.sh"
            exit 1
        fi

        # Check if spec says we're done
        if grep -qi "final sprint" "$STATE_DIR/checkpoint.md" 2>/dev/null; then
            ok "All sprints complete!"
            break
        fi
    done

    echo ""
    ok "Harness finished. Check logs/ and state/ for details."
}

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume) RESUME_FROM="$2"; shift 2 ;;
        --max-sprints) MAX_SPRINTS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

main
