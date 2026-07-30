"""
autoresearch_loop.py — Autonomous experiment loop for NSTP-Ω.

Adapted from Karpathy's autoresearch design. Loop forever:
1. Read current branch (autoresearch/<tag>)
2. Modify train.py with one experimental idea
3. git commit
4. Run experiment with fixed time budget
5. Read val_bpb from run.log
6. If improved: keep (advance branch)
7. If worse/equal: git reset (discard)
8. Log to results.tsv

For NSTP-Ω V3 we use val_bpb as the metric (vocab-independent).
Time budget: 30 minutes default (longer than Karpathy's 5 min because
our 144M model is bigger and the recurrent DeltaNet needs more steps).

The agent (you/me) modifies train.py autonomously. This script is the
scaffolding that runs experiments and tracks results.
"""
import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================================
# CONFIG
# ============================================================================
REPO_DIR = Path('C:/Users/user/AppData/Local/Temp/nstp-v2')
PYTHON_EXE = r'C:\Users\user\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe'
TRAIN_SCRIPT = 'train_nstp_omega_v3.py'
RUN_LOG = REPO_DIR / 'run.log'
RESULTS_TSV = REPO_DIR / 'results.tsv'
EXPERIMENT_TIME_BUDGET = 1800  # 30 minutes in seconds
MAX_EXPERIMENT_DURATION = 2400  # hard kill at 40 min


# ============================================================================
# GIT HELPERS
# ============================================================================
def run(cmd: str, cwd: Path = REPO_DIR, check: bool = True) -> str:
    """Run shell command in repo dir, return stdout."""
    print(f"$ {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=60
    )
    if check and result.returncode != 0:
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout.strip()


def git_status() -> str:
    return run('git status --short', check=False)


def git_current_branch() -> str:
    return run('git rev-parse --abbrev-ref HEAD', check=False)


def git_commit_hash() -> str:
    """Get current short commit hash."""
    return run('git rev-parse --short HEAD', check=False)


def git_diff_stats() -> str:
    """Get summary of uncommitted changes."""
    return run('git diff --stat', check=False)


def git_commit(message: str) -> str:
    """Commit current changes. Returns new commit hash."""
    run('git add -A')
    output = run(f'git commit -m "{message}"', check=False)
    return git_commit_hash()


def git_reset_to(commit_hash: str):
    """Hard reset to a specific commit."""
    run(f'git reset --hard {commit_hash}', check=False)


def create_branch(tag: str) -> str:
    """Create autoresearch/<tag> branch from current master."""
    branch = f'autoresearch/{tag}'
    run(f'git checkout -b {branch} master', check=False)
    return branch


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================
def run_experiment(time_budget: int = EXPERIMENT_TIME_BUDGET) -> dict:
    """
    Run train.py as a background process for time_budget seconds.
    Returns dict with parsed metrics from run.log.
    """
    print(f"\n{'='*70}")
    print(f"STARTING EXPERIMENT ({time_budget}s budget)")
    print(f"{'='*70}")

    # Clean log
    if RUN_LOG.exists():
        RUN_LOG.unlink()

    # Launch training as background process (cross-platform: list args + cwd)
    cmd = [
        PYTHON_EXE,
        '-u', TRAIN_SCRIPT
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=open(RUN_LOG, 'w'),
        stderr=subprocess.STDOUT,
        cwd=REPO_DIR
    )
    print(f"  PID: {proc.pid}")

    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > MAX_EXPERIMENT_DURATION:
            print(f"  HARD KILL at {elapsed:.0f}s (exceeded {MAX_EXPERIMENT_DURATION}s)")
            proc.kill()
            return {'status': 'timeout', 'elapsed': elapsed}

        # Check if process still alive
        if proc.poll() is not None:
            print(f"  Process exited at {elapsed:.0f}s (rc={proc.returncode})")
            break

        time.sleep(10)

    # Parse metrics
    return parse_run_log()


def parse_run_log() -> dict:
    """Extract metrics from run.log."""
    if not RUN_LOG.exists():
        return {'status': 'crash', 'reason': 'no log file'}

    content = RUN_LOG.read_text()

    # Check for crash indicators
    if 'Traceback' in content or 'Error' in content[-5000:]:
        # Find last traceback
        tb_idx = content.rfind('Traceback')
        if tb_idx >= 0:
            tb = content[tb_idx:tb_idx+2000]
            return {'status': 'crash', 'traceback': tb}

    # Parse metrics (val_bpb, val_ppl, peak_vram_mb, etc.)
    metrics = {}
    for line in content.split('\n'):
        if 'val_bpb:' in line:
            try:
                metrics['val_bpb'] = float(line.split('val_bpb:')[1].strip())
            except ValueError:
                pass
        elif 'peak_vram_mb:' in line:
            try:
                metrics['peak_vram_mb'] = float(line.split('peak_vram_mb:')[1].strip())
            except ValueError:
                pass
        elif 'training_seconds:' in line:
            try:
                metrics['training_seconds'] = float(line.split('training_seconds:')[1].strip())
            except ValueError:
                pass
        elif 'num_params_M:' in line:
            try:
                metrics['num_params_M'] = float(line.split('num_params_M:')[1].strip())
            except ValueError:
                pass

    if 'val_bpb' not in metrics:
        return {'status': 'crash', 'reason': 'no val_bpb in log'}

    metrics['status'] = 'ok'
    return metrics


# ============================================================================
# RESULTS LOGGING
# ============================================================================
def init_results_tsv():
    """Create results.tsv with header if not exists."""
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, 'w') as f:
            f.write('commit\tval_bpb\tmemory_gb\tstatus\tdescription\n')


def log_result(commit: str, val_bpb: float, memory_gb: float,
               status: str, description: str):
    """Append one row to results.tsv."""
    with open(RESULTS_TSV, 'a') as f:
        f.write(f'{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n')


def read_results() -> list:
    """Read all results from results.tsv."""
    if not RESULTS_TSV.exists():
        return []
    results = []
    with open(RESULTS_TSV) as f:
        lines = f.readlines()
    for line in lines[1:]:  # skip header
        parts = line.strip().split('\t')
        if len(parts) >= 5:
            results.append({
                'commit': parts[0],
                'val_bpb': float(parts[1]),
                'memory_gb': float(parts[2]),
                'status': parts[3],
                'description': parts[4]
            })
    return results


def best_val_bpb() -> float:
    """Get best val_bpb across all kept experiments."""
    results = read_results()
    kept = [r for r in results if r['status'] == 'keep']
    if not kept:
        return float('inf')
    return min(r['val_bpb'] for r in kept)


# ============================================================================
# MAIN LOOP
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Autoresearch loop for NSTP-Ω')
    parser.add_argument('--tag', type=str, default=None,
                        help='Branch tag (default: today\'s date)')
    parser.add_argument('--budget', type=int, default=EXPERIMENT_TIME_BUDGET,
                        help=f'Time budget per experiment (seconds), default {EXPERIMENT_TIME_BUDGET}')
    parser.add_argument('--max-experiments', type=int, default=100,
                        help='Max experiments to run (0 = forever)')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Just run baseline, no loop')
    args = parser.parse_args()

    # Tag = today's date (YYYY-MM-DD)
    if args.tag is None:
        args.tag = datetime.now().strftime('%Y%m%d')

    print(f"Autoresearch loop starting")
    print(f"  Repo: {REPO_DIR}")
    print(f"  Tag: {args.tag}")
    print(f"  Budget: {args.budget}s per experiment")
    print(f"  Train script: {TRAIN_SCRIPT}")

    # Init results.tsv
    init_results_tsv()

    # Create branch
    branch = create_branch(args.tag)
    print(f"  Branch: {branch}")

    # Baseline experiment (no code changes)
    if args.baseline_only:
        print("\n=== BASELINE EXPERIMENT ===")
        commit_before = git_commit_hash()
        print(f"  Baseline commit: {commit_before}")

        result = run_experiment(args.budget)

        if result['status'] == 'ok':
            val_bpb = result['val_bpb']
            memory_gb = result['peak_vram_mb'] / 1024
            log_result(commit_before, val_bpb, memory_gb, 'keep', 'baseline')
            print(f"  Baseline val_bpb: {val_bpb:.4f}")
            print(f"  Baseline VRAM: {memory_gb:.1f}GB")
            return 0

        print(f"  Baseline FAILED: {result}")
        return 1

    # Full loop
    n_experiments = 0
    current_best = best_val_bpb()
    print(f"  Starting best: {current_best:.4f}" if current_best != float('inf') else "  No baseline yet")

    while True:
        if args.max_experiments > 0 and n_experiments >= args.max_experiments:
            print(f"Reached max experiments ({args.max_experiments}). Stopping.")
            break

        n_experiments += 1
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT {n_experiments}")
        print(f"{'#'*70}")

        # Save current commit (to reset to if experiment fails)
        commit_before = git_commit_hash()
        print(f"  Starting commit: {commit_before}")

        # THE AGENT MODIFIES train.py HERE
        # In autonomous mode, the LLM would edit train.py directly.
        # In our setup, we expect manual edits or programmatic edits via other tools.

        # Check if there are uncommitted changes
        diff = git_diff_stats()
        if not diff:
            print("  No changes to test. Exiting loop.")
            print("  (In autonomous mode, the agent would now edit train.py)")
            break

        # Commit the experimental change
        desc = f"experiment {n_experiments}"  # User should provide better description
        commit_hash = git_commit(desc)

        # Run experiment
        result = run_experiment(args.budget)

        if result['status'] != 'ok':
            # Crash or timeout
            print(f"  EXPERIMENT {n_experiments} CRASHED")
            log_result(commit_hash, 0.0, 0.0, 'crash', desc)
            # Don't reset — keep the code for debugging
            continue

        val_bpb = result['val_bpb']
        memory_gb = result['peak_vram_mb'] / 1024
        print(f"  val_bpb: {val_bpb:.4f}")
        print(f"  VRAM: {memory_gb:.1f}GB")

        # Keep or discard?
        if val_bpb < current_best:
            print(f"  ✓ KEEP (improved from {current_best:.4f} to {val_bpb:.4f})")
            log_result(commit_hash, val_bpb, memory_gb, 'keep', desc)
            current_best = val_bpb
        else:
            print(f"  ✗ DISCARD (no improvement: {val_bpb:.4f} >= {current_best:.4f})")
            log_result(commit_hash, val_bpb, memory_gb, 'discard', desc)
            git_reset_to(commit_before)
            print(f"  Reset to {commit_before}")

    print(f"\n{'='*70}")
    print(f"AUTORESEARCH COMPLETE — {n_experiments} experiments")
    print(f"Best val_bpb: {current_best:.4f}")
    print(f"{'='*70}")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
