# Tennis Vision — Fix Loop Script
# Iterates: Evaluator → Generator Fix → Evaluator → ... until clean or max rounds hit

param(
    [int]$MaxRounds = 5,
    [string]$TestVideo = "",  # Optional: path to a real tennis video for integration testing
    [int]$StartRound = 1,     # Resume from this round number
    [string]$StartPhase = "evaluator"  # Resume from "evaluator" or "generator"
)

$ErrorActionPreference = "Continue"

$ProjectDir = Split-Path -Parent $PSScriptRoot
$StateDir = Join-Path $ProjectDir "state"
$LogDir = Join-Path $ProjectDir "logs"

$CopilotCmd = "copilot"
$BaseFlags = @("--allow-all-tools", "--allow-all-paths", "--output-format", "text")
$GeneratorFlags = $BaseFlags + @("--model", "claude-opus-4.6", "--effort", "medium")
$EvaluatorFlags = $BaseFlags + @("--model", "claude-opus-4.6", "--effort", "high")

function Log($msg) { Write-Host "[fix-loop] $(Get-Date -Format 'HH:mm:ss') $msg" -ForegroundColor Cyan }
function Ok($msg)  { Write-Host "[✓] $msg" -ForegroundColor Green }
function Err($msg) { Write-Host "[✗] $msg" -ForegroundColor Red }
function Warn($msg){ Write-Host "[!] $msg" -ForegroundColor Yellow }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Build the video test instruction
$videoInstruction = ""
if ($TestVideo -ne "" -and (Test-Path $TestVideo)) {
    $videoInstruction = "CRITICAL: A real tennis test video is available at '$TestVideo'. You MUST: 1) Start the API server (uvicorn src.api.main:app --port 8000) in the background, 2) Upload this video via POST /upload, 3) Wait for processing to complete via GET /status/job_id, 4) Check GET /results/job_id for meaningful output (ball detections > 0, court detected, etc.), 5) If the frontend exists, try npm run build in frontend/ and verify heatmap image URLs work, 6) Kill the API server when done. Include the actual results numbers in your feedback (how many balls detected, court confidence, rallies found, etc.)."
    Ok "Test video: $TestVideo"
} else {
    if ($TestVideo -ne "") { Warn "Test video not found: $TestVideo" }
}

# ============================================================
# Evaluator: find all issues
# ============================================================
function Run-Evaluator($round) {
    Log "Round $round — Running Evaluator..."

    $evalPrompt = "You are the Evaluator agent for the Tennis Vision project. Read .github/agents/evaluator.agent.md, AGENTS.md, and state/checkpoint.md. Your job: find EVERY bug, broken feature, code quality issue, and missing piece. Be thorough and ruthless. Check: all Python tests pass (python -m pytest tests/ -v), all imports work, API server starts and all endpoints respond correctly, frontend builds (cd frontend && npm run build), frontend heatmap image URLs are correct (no double-prefixed paths), pipeline test mocks match actual code (iter_frames vs read_all), edge cases (empty video, corrupt file, missing weights), code quality (unused imports, duplicated code, hardcoded values). $videoInstruction Write your findings to state/feedback.md. Use this format: ## Issues Found (numbered list, each with file path and line number), ## Fixed Since Last Round (what improved if round 2+), ## Verdict (either CLEAN or NEEDS_FIX). The verdict line MUST be the last line and MUST be exactly: Verdict: CLEAN or Verdict: NEEDS_FIX"

    Push-Location $ProjectDir
    $logFile = Join-Path $LogDir "fix-round-${round}-evaluator.log"
    Log "Evaluator log: $logFile"
    $evalArgs = @($evalPrompt) + $EvaluatorFlags
    $proc = Start-Process -FilePath $CopilotCmd -ArgumentList (@("-p") + $evalArgs) -NoNewWindow -Wait -RedirectStandardOutput $logFile -RedirectStandardError (Join-Path $LogDir "fix-round-${round}-evaluator-stderr.log") -PassThru
    Log "Evaluator exit code: $($proc.ExitCode)"
    Pop-Location

    $feedbackFile = Join-Path $StateDir "feedback.md"
    if (-not (Test-Path $feedbackFile)) {
        Warn "Evaluator did not write feedback.md"
        return "NEEDS_FIX"
    }

    $content = Get-Content $feedbackFile -Raw
    if ($content -match "Verdict:\s*CLEAN") {
        return "CLEAN"
    }
    return "NEEDS_FIX"
}

# ============================================================
# Generator: fix everything in feedback
# ============================================================
function Run-Fixer($round) {
    Log "Round $round — Running Generator (fix mode)..."

    $fixPrompt = "You are the Generator agent in FIX MODE. Read .github/agents/generator.agent.md, AGENTS.md, state/checkpoint.md, and state/feedback.md. Your ONLY job: fix EVERY issue listed in feedback.md. Do not add new features. Do not skip issues. For each issue: read the relevant file, fix the bug, verify the fix works (run the test, import the module, etc.). After fixing everything: run the full test suite (python -m pytest tests/ -v), verify API starts (python -c 'from src.api.main import app; print(ok)'), update state/checkpoint.md with what you fixed. Be thorough. The evaluator will check your work again."

    Push-Location $ProjectDir
    $logFile = Join-Path $LogDir "fix-round-${round}-generator.log"
    Log "Generator log: $logFile"
    $fixArgs = @($fixPrompt) + $GeneratorFlags
    $proc = Start-Process -FilePath $CopilotCmd -ArgumentList (@("-p") + $fixArgs) -NoNewWindow -Wait -RedirectStandardOutput $logFile -RedirectStandardError (Join-Path $LogDir "fix-round-${round}-generator-stderr.log") -PassThru
    Log "Generator exit code: $($proc.ExitCode)"
    Pop-Location

    Ok "Generator fix round $round complete"
}

# ============================================================
# Main
# ============================================================
Write-Host ""
Write-Host "============================================"
Write-Host "  Tennis Vision — Fix Loop"
Write-Host "  Max rounds: $MaxRounds"
if ($StartRound -gt 1 -or $StartPhase -ne "evaluator") {
    Write-Host "  Resuming: round $StartRound, phase $StartPhase"
}
Write-Host "============================================"
Write-Host ""

for ($round = $StartRound; $round -le $MaxRounds; $round++) {
    Log "==================== Round $round / $MaxRounds ===================="

    # Skip evaluator if resuming into generator phase
    $skipEval = ($round -eq $StartRound -and $StartPhase -eq "generator")

    if (-not $skipEval) {
        $verdict = Run-Evaluator $round

        if ($verdict -eq "CLEAN") {
            Ok "Round $round - Evaluator says CLEAN - all issues resolved!"
            break
        }

        if ($round -eq $MaxRounds) {
            Warn "Reached max rounds ($MaxRounds). Some issues may remain."
            Warn "Check state/feedback.md for remaining issues."
            break
        }
    } else {
        Log "Skipping evaluator (resuming at generator phase)"
    }

    Run-Fixer $round

    # Small git checkpoint
    Push-Location $ProjectDir
    & git add -A 2>$null
    & git commit -m "fix-loop round $round" --allow-empty 2>$null
    Pop-Location
    Ok "Git checkpoint after round $round"
}

# Final git push
Push-Location $ProjectDir
& git add -A 2>$null
& git commit -m "fix-loop complete" --allow-empty 2>$null
& git push 2>$null
Pop-Location

Write-Host ""
Ok "Fix loop finished. Check logs/fix-round-*.log and state/feedback.md"
