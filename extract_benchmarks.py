"""Extract frontier model benchmarks from multiple sources"""
import sys, time
sys.path = [p for p in sys.path if 'hermes' not in p.lower()]
sys.path.insert(0, 'C:/Users/user/helix_local/helix/benchmarks')
from web_search_stealth import StealthBrowser

browser = StealthBrowser(fingerprint='benchmarks')
browser.connect()

# Try morphllm Claude benchmarks
browser.goto('https://www.morphllm.com/claude-benchmarks')
time.sleep(8)
body = browser.get_text()
idx = body.find('SWE-bench')
if idx >= 0:
    print("=== MORPHPCLAUDE BENCHMARKS ===")
    print(body[idx:idx+4000])
else:
    print("No SWE-bench found on morphllm")
    print(body[-2000:])

# Try BenchLM
browser.goto('https://benchlm.ai/')
time.sleep(8)
body2 = browser.get_text()
idx2 = body2.find('Fable')
if idx2 >= 0:
    print("\n=== BENCHLM LEADERBOARD ===")
    print(body2[max(0,idx2-200):idx2+5000])
else:
    print("\nNo Fable found on benchlm")
    print(body2[-3000:])

browser.close()
browser.shutdown()