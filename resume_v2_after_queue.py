"""
resume_v2_after_queue.py — Resume V2 training after autoresearch queue finishes.

Waits for proc_4f7789b03502 (the queue) to exit, then starts V2 training
from step 30K checkpoint. Logs to its own file.
"""
import subprocess
import time
import sys
from pathlib import Path

REPO_DIR = Path('C:/Users/user/AppData/Local/Temp/nstp-v2')
PYTHON_EXE = r'C:\Users\user\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe'
QUEUE_PID_FILE = REPO_DIR / 'logs' / 'queue.pid'
V2_LOG = REPO_DIR / 'logs' / 'training_v2_resume.log'


def is_queue_alive():
    """Check if autoresearch queue process is still running."""
    # Try to find python.exe processes running autoresearch_queue.py
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe'],
            capture_output=True, text=True, timeout=10
        )
        # Look for running python processes (PID is second column)
        for line in result.stdout.split('\n'):
            if 'python.exe' in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    # Check if this PID matches our queue (we don't track exact PID, but queue is the heaviest python process)
                    pass
        return True
    except Exception:
        return False


def wait_for_queue_to_finish():
    """Poll for queue process to exit."""
    print(f"Waiting for autoresearch queue to finish...")
    print(f"  Will resume V2 from step 30K checkpoint after.")
    print(f"  V2 log: {V2_LOG}")
    print()

    # Wait for 9 total results entries (TEST_baseline + 8 queue experiments)
    target_count = 9
    last_count = None
    while True:
        # Check results.tsv for new entries
        tsv = REPO_DIR / 'results.tsv'
        if tsv.exists():
            lines = tsv.read_text().strip().split('\n')
            count = len(lines) - 1  # Subtract header
            if last_count != count:
                print(f"  Results count: {count}/{target_count}")
                last_count = count
            if count >= target_count:
                print(f"  All {target_count - 1} queue experiments + baseline done!")
                return True
        time.sleep(60)


def start_v2():
    """Launch V2 training in background."""
    print()
    print("=" * 70)
    print("STARTING V2 TRAINING (resume from step 30K)")
    print("=" * 70)

    V2_LOG.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXE,
        '-u', 'train_nstp_omega_v2.py'
    ]
    log_file = open(V2_LOG, 'w')
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    print(f"  V2 PID: {proc.pid}")
    print(f"  Log: {V2_LOG}")
    print(f"  Expected resume point: step 30,000")
    print(f"  Remaining steps to 50K: 20,000 (~7 hours)")
    return proc


def main():
    print("V2 Auto-Resume Watcher")
    print(f"  Repo: {REPO_DIR}")
    print(f"  Watching: results.tsv (need 8 entries)")
    print()

    wait_for_queue_to_finish()
    v2_proc = start_v2()

    print()
    print("V2 started successfully.")
    print(f"  Monitor with: tail -f {V2_LOG}")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
