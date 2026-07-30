"""
autoresearch_queue.py — Run multiple V3 experiments sequentially.

Designed for overnight autonomous research. Reads experiment configs
from configs/ directory, runs each via run_experiment.py, logs results
to results.tsv. Skips experiments that exceed VRAM or crash.

Usage:
    python autoresearch_queue.py                  # run all configs
    python autoresearch_queue.py --max N          # max N experiments
    python autoresearch_queue.py --pattern 'lr_*' # glob filter
    python autoresearch_queue.py --wait-for-gpu   # poll GPU mem until free
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


REPO_DIR = Path('C:/Users/user/AppData/Local/Temp/nstp-v2')
CONFIGS_DIR = REPO_DIR / 'configs'
RESULTS_TSV = REPO_DIR / 'results.tsv'
RUN_LOG = REPO_DIR / 'run.log'


def wait_for_gpu_free(threshold_gb: float = 4.0, poll_interval: int = 60):
    """Wait until GPU memory usage is below threshold."""
    print(f"Waiting for GPU to be free (< {threshold_gb}GB)...")
    while True:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            used_mb = float(result.stdout.strip().split('\n')[0])
            used_gb = used_mb / 1024
            print(f"  GPU usage: {used_gb:.1f}GB", end='\r')
            if used_gb < threshold_gb:
                print(f"\n  GPU free ({used_gb:.1f}GB < {threshold_gb}GB)")
                return True
        except Exception as e:
            print(f"  Error checking GPU: {e}")
        time.sleep(poll_interval)


def run_one_experiment(config_path: Path, description: str, time_budget: int = 1800) -> dict:
    """Run one experiment via subprocess."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {description}")
    print(f"Config: {config_path.name}")
    print(f"Time budget: {time_budget}s")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    # Run via run_experiment.py
    cmd = (
        f'cd {REPO_DIR} && '
        f'PYTHONPATH= /c/Users/user/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe '
        f'-u run_experiment.py --config {config_path} --description "{description}" --budget {time_budget}'
    )

    proc = subprocess.Popen(cmd, shell=True)

    # Wait for completion
    proc.wait()

    # Parse last result from results.tsv
    return read_last_result()


def read_last_result() -> dict:
    """Read the most recent result from results.tsv."""
    if not RESULTS_TSV.exists():
        return {'status': 'no_results_file'}
    with open(RESULTS_TSV) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return {'status': 'no_results'}
    last = lines[-1].strip().split('\t')
    if len(last) >= 5:
        return {
            'commit': last[0],
            'val_bpb': float(last[1]),
            'memory_gb': float(last[2]),
            'status': last[3],
            'description': last[4]
        }
    return {'status': 'parse_error'}


def list_configs(pattern: str = '*.json') -> list:
    """List config files matching pattern."""
    return sorted(CONFIGS_DIR.glob(pattern))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=0, help='Max experiments (0 = all)')
    parser.add_argument('--pattern', type=str, default='*.json', help='Config glob')
    parser.add_argument('--budget', type=int, default=1800, help='Per-experiment seconds')
    parser.add_argument('--wait-for-gpu', action='store_true', help='Wait for GPU to be free first')
    parser.add_argument('--gpu-threshold', type=float, default=4.0, help='GPU free threshold (GB)')
    args = parser.parse_args()

    # Find configs
    configs = list_configs(args.pattern)
    if not configs:
        print(f"No configs found matching {args.pattern} in {CONFIGS_DIR}")
        return 1

    print(f"Found {len(configs)} experiments:")
    for c in configs:
        print(f"  - {c.name}")

    if args.max > 0:
        configs = configs[:args.max]
        print(f"\nRunning first {args.max}")

    # Optional GPU wait
    if args.wait_for_gpu:
        wait_for_gpu_free(args.gpu_threshold)

    # Run experiments sequentially
    start = time.time()
    results = []
    for i, config_path in enumerate(configs, 1):
        description = f"{config_path.stem} (auto-queue #{i})"
        try:
            result = run_one_experiment(config_path, description, args.budget)
            results.append(result)
            print(f"\n  Result: {result}")
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"\n  FAILED: {e}")
            results.append({'status': 'error', 'error': str(e)})

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"AUTORESEARCH QUEUE COMPLETE")
    print(f"{'='*70}")
    print(f"Experiments: {len(results)}")
    print(f"Total time: {elapsed/60:.1f}m")
    print(f"Results in {RESULTS_TSV}")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
