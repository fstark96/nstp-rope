"""
run_experiment.py — Parameterized experiment runner for NSTP-Ω V3.

Runs `train_nstp_omega_v3.py` with specific config overrides via env vars
or a config file. Logs results to results.tsv.

This is a simpler version of autoresearch — instead of editing code between
runs (Karpathy's design), we override hyperparameters via config dict.

Usage:
    python run_experiment.py --config <json_file> --description "..."
"""
import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from datetime import datetime


REPO_DIR = Path('C:/Users/user/AppData/Local/Temp/nstp-v2')
PYTHON_EXE = r'C:\Users\user\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe'
RESULTS_TSV = REPO_DIR / 'results.tsv'
RUN_LOG = REPO_DIR / 'run.log'
TIME_BUDGET = 1800  # 30 min default
MAX_DURATION = 2400  # 40 min hard kill


def run_experiment(config: dict, description: str, time_budget: int = TIME_BUDGET) -> dict:
    """
    Run experiment with given config overrides.
    Returns parsed metrics.
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {description}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"{'='*70}")

    # Launch training (cross-platform: use env dict, no inline PYTHONPATH=)
    if RUN_LOG.exists():
        RUN_LOG.unlink()

    # Use unique save dir per experiment to avoid checkpoint resume
    exp_save_dir = REPO_DIR / 'experiments' / description.replace(' ', '_').replace('/', '_')[:50]
    exp_save_dir.mkdir(parents=True, exist_ok=True)
    exp_config = {
        **config,
        '_SAVE_DIR': str(exp_save_dir)  # Tell training script where to save
    }
    with open(REPO_DIR / 'experiment_config.json', 'w') as f:
        json.dump(exp_config, f, indent=2)

    env = os.environ.copy()
    env['PYTHONPATH'] = ''
    env['EXPERIMENT_CONFIG'] = str(REPO_DIR / 'experiment_config.json')

    cmd = [
        PYTHON_EXE,
        '-u', 'train_nstp_omega_v3.py'
    ]
    print(f"$ {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=open(RUN_LOG, 'w'),
        stderr=subprocess.STDOUT,
        cwd=REPO_DIR
    )

    # Poll with timeout
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > MAX_DURATION:
            print(f"  HARD KILL at {elapsed:.0f}s")
            proc.kill()
            return {'status': 'timeout', 'elapsed': elapsed}

        if proc.poll() is not None:
            print(f"  Exited at {elapsed:.0f}s (rc={proc.returncode})")
            break

        time.sleep(10)

    return parse_metrics()


def parse_metrics() -> dict:
    """Parse val_bpb and other metrics from run.log."""
    if not RUN_LOG.exists():
        return {'status': 'crash', 'reason': 'no log'}

    content = RUN_LOG.read_text()

    # Check for crashes
    if 'Traceback' in content[-5000:]:
        tb_idx = content.rfind('Traceback')
        return {'status': 'crash', 'traceback': content[tb_idx:tb_idx+1500]}

    metrics = {}
    for line in content.split('\n'):
        for key in ['val_bpb:', 'peak_vram_mb:', 'total_seconds:', 'num_params_M:', 'num_steps:']:
            if key in line:
                try:
                    val_str = line.split(key)[1].strip().split()[0]
                    metrics[key.rstrip(':')] = float(val_str)
                except (ValueError, IndexError):
                    pass

    if 'val_bpb' not in metrics:
        return {'status': 'crash', 'reason': 'no val_bpb in log'}

    metrics['status'] = 'ok'
    return metrics


def log_to_tsv(description: str, val_bpb: float, memory_gb: float, status: str):
    """Append one row to results.tsv."""
    # Get current commit
    result = subprocess.run(
        'git rev-parse --short HEAD', shell=True, cwd=REPO_DIR,
        capture_output=True, text=True
    )
    commit = result.stdout.strip() or 'uncommitted'

    # Init TSV if needed
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, 'w') as f:
            f.write('commit\tval_bpb\tmemory_gb\tstatus\tdescription\n')

    with open(RESULTS_TSV, 'a') as f:
        f.write(f'{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n')

    print(f"  Logged: {commit} | {val_bpb:.4f} | {memory_gb:.1f}GB | {status} | {description}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='JSON config file path')
    parser.add_argument('--description', type=str, required=True)
    parser.add_argument('--budget', type=int, default=TIME_BUDGET)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())

    result = run_experiment(config, args.description, args.budget)

    if result['status'] == 'ok':
        val_bpb = result['val_bpb']
        memory_gb = result.get('peak_vram_mb', 0) / 1024
        log_to_tsv(args.description, val_bpb, memory_gb, 'keep')
        print(f"\n✓ val_bpb: {val_bpb:.4f}")
        return 0
    else:
        log_to_tsv(args.description, 0.0, 0.0, 'crash')
        print(f"\n✗ FAILED: {result}")
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)
