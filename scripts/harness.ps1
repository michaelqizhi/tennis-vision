# Tennis Vision — Harness Script (PowerShell)
# Orchestrates: Planner → Generator → Evaluator loop with fix cycles

param(
    [int]$ResumeFrom = 0,
    [int]$MaxSprints = 8,
    [int]$MaxFixCycles = 2
)

$ErrorActionPreference = "Continue"

$ProjectDir = Split-Path -Parent $PSScriptRoot
$StateDir = Join-Path $ProjectDir "state"
$LogDir = Join-Path $ProjectDir "logs"

# Model config
$CopilotCmd = "copilot"
$BaseFlags = @("--allow-all-tools", "--allow-all-paths", "--output-format", "text")
$PlannerFlags = $BaseFlags + @("--model", "claude-opus-4.6", "--effort", "high")
$GeneratorFlags = $BaseFlags + @("--model", "claude-opus-4.6", "--effort", "medium")
$EvaluatorFlags = $BaseFlags + @("--model", "claude-opus-4.6", "--effort", "medium")

function Log($msg) { Write-Host "[harness] $(Get-Date -Format 'HH:mm:ss') $msg" -ForegroundColor Cyan }
function Ok($msg)  { Write-Host "[✓] $msg" -ForegroundColor Green }
function Err($msg) { Write-Host "[✗] $msg" -ForegroundColor Red }
function Warn($msg){ Write-Host "[!] $msg" -ForegroundColor Yellow }

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ============================================================
# Preflight
# ============================================================
function Preflight {
    Log "Preflight checks..."

    if (-not (Get-Command $CopilotCmd -ErrorAction SilentlyContinue)) {
        Err "Copilot CLI not found"
        exit 1
    }

    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Err "Python not found"
        exit 1
    }

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpu = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
        Ok "GPU detected: $gpu"
    } else {
        Warn "No GPU detected"
    }

    Ok "Preflight passed"
}

# ============================================================
# Planner
# ============================================================
function Run-Planner {
    $specFile = Join-Path $StateDir "spec.md"

    if ((Test-Path $specFile) -and ($ResumeFrom -gt 0)) {
        Log "spec.md exists, skipping planner (resume mode)"
        return
    }

    Log "Running Planner..."

    $prompt = "You are the Planner agent. Read the file .github/agents/planner.agent.md for your role definition, then read AGENTS.md for the project spec. Follow your instructions to produce state/spec.md."

    Push-Location $ProjectDir
    & $CopilotCmd -p $prompt @PlannerFlags 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "00-planner.log")
    Pop-Location

    if (Test-Path $specFile) {
        Ok "Planner complete - spec.md written"
    } else {
        Err "Planner did not produce spec.md"
        exit 1
    }
}

# ============================================================
# Check if feedback has critical issues
# ============================================================
function Has-CriticalIssues {
    $feedbackFile = Join-Path $StateDir "feedback.md"
    if (-not (Test-Path $feedbackFile)) { return $false }

    $content = Get-Content $feedbackFile -Raw -ErrorAction SilentlyContinue
    if ($content -match "(?i)what's broken" -and $content -match "(?i)critical") {
        return $true
    }
    if ($content -match "(?i)## What's Broken" -and $content.Length -gt 500) {
        # If there's a "What's Broken" section with substantial content
        $brokenSection = ($content -split "(?i)## What's Broken")[1]
        if ($brokenSection) {
            $brokenSection = ($brokenSection -split "(?i)## ")[0]  # Get just that section
            $lines = ($brokenSection -split "`n" | Where-Object { $_.Trim() -ne "" }).Count
            if ($lines -ge 3) { return $true }
        }
    }
    return $false
}

# ============================================================
# Generator + Evaluator Sprint (with fix cycles)
# ============================================================
function Run-Sprint($sprint) {
    Log "========== Sprint $sprint =========="

    # --- Generator (implement new feature) ---
    Log "Running Generator (Sprint $sprint)..."

    $genPrompt = "You are the Generator agent. Read .github/agents/generator.agent.md for your role, then AGENTS.md for conventions, then state/spec.md for the plan. This is Sprint $sprint. Read state/checkpoint.md and state/feedback.md if they exist. IMPORTANT: If feedback.md reports bugs or issues from the previous sprint, fix those FIRST before implementing new features. Update state/checkpoint.md when done."

    Push-Location $ProjectDir
    & $CopilotCmd -p $genPrompt @GeneratorFlags 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "sprint-${sprint}-generator.log")
    Pop-Location

    $checkpointFile = Join-Path $StateDir "checkpoint.md"
    if (-not (Test-Path $checkpointFile)) {
        Err "Generator did not update checkpoint.md"
        return $false
    }
    Ok "Generator Sprint $sprint complete"

    # --- Evaluator ---
    Log "Running Evaluator (Sprint $sprint)..."

    $evalPrompt = "You are the Evaluator agent. Read .github/agents/evaluator.agent.md for your role. Then read AGENTS.md and state/checkpoint.md. DO NOT read state/spec.md. Evaluate the current state of the project - try to run the code, test features, check quality. Write your findings to state/feedback.md. Mark critical issues clearly with '### Critical' heading."

    Push-Location $ProjectDir
    & $CopilotCmd -p $evalPrompt @EvaluatorFlags 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "sprint-${sprint}-evaluator.log")
    Pop-Location

    if (-not (Test-Path (Join-Path $StateDir "feedback.md"))) {
        Warn "Evaluator did not write feedback.md - continuing anyway"
        Log "Sprint $sprint done"
        return $true
    }
    Ok "Evaluator Sprint $sprint complete"

    # --- Fix cycles: if critical issues, send back to Generator ---
    for ($fix = 1; $fix -le $MaxFixCycles; $fix++) {
        if (-not (Has-CriticalIssues)) {
            Log "No critical issues found - moving to next sprint"
            break
        }

        Log "Critical issues detected - Fix cycle $fix/$MaxFixCycles"

        $fixPrompt = "You are the Generator agent in FIX MODE. Read .github/agents/generator.agent.md, AGENTS.md, state/checkpoint.md, and state/feedback.md. Your ONLY job this session is to fix the issues reported in feedback.md — especially anything marked Critical. Do NOT add new features. Update state/checkpoint.md when done."

        Push-Location $ProjectDir
        & $CopilotCmd -p $fixPrompt @GeneratorFlags 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "sprint-${sprint}-fix${fix}-generator.log")
        Pop-Location

        Ok "Fix cycle $fix Generator complete"

        # Re-evaluate
        Log "Re-evaluating after fix cycle $fix..."

        Push-Location $ProjectDir
        & $CopilotCmd -p $evalPrompt @EvaluatorFlags 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "sprint-${sprint}-fix${fix}-evaluator.log")
        Pop-Location

        Ok "Fix cycle $fix Evaluator complete"
    }

    if (Has-CriticalIssues) {
        Warn "Critical issues remain after $MaxFixCycles fix cycles - continuing anyway"
    }

    Log "Sprint $sprint done"
    return $true
}

# ============================================================
# Main
# ============================================================
Write-Host ""
Write-Host "============================================"
Write-Host "  Tennis Vision - Automated Build Harness"
Write-Host "  (with fix cycles, max $MaxFixCycles per sprint)"
Write-Host "============================================"
Write-Host ""

Preflight

if ($ResumeFrom -eq 0) {
    Run-Planner
} else {
    Log "Resuming from Sprint $ResumeFrom"
}

$startSprint = if ($ResumeFrom -gt 0) { $ResumeFrom } else { 1 }

for ($i = $startSprint; $i -le $MaxSprints; $i++) {
    $result = Run-Sprint $i
    if (-not $result) {
        Err "Sprint $i failed. Resume with: .\scripts\harness.ps1 -ResumeFrom $i"
        exit 1
    }

    $checkpointContent = Get-Content (Join-Path $StateDir "checkpoint.md") -Raw -ErrorAction SilentlyContinue
    if ($checkpointContent -match "(?i)final sprint") {
        Ok "All sprints complete!"
        break
    }
}

Write-Host ""
Ok "Harness finished. Check logs\ and state\ for details."
