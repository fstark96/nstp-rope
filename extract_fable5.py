"""Extract Claude Fable 5 technical details"""
import sys, time
sys.path = [p for p in sys.path if 'hermes' not in p.lower()]
sys.path.insert(0, 'C:/Users/user/helix_local/helix/benchmarks')
from web_search_stealth import StealthBrowser

browser = StealthBrowser(fingerprint='fable5')
browser.connect()

# Anthropic announcement
browser.goto('https://www.anthropic.com/news/claude-fable-5-mythos-5')
time.sleep(10)
body = browser.get_text()
idx = body.find('Claude Fable 5')
if idx >= 0:
    print("=== ANTHROPIC ANNOUNCEMENT ===")
    print(body[idx:idx+8000])

# Technical analysis
browser.goto('https://agentbreaking.com/blog/claude-fable-5-technical-analysis-anthropic-safety-architecture/')
time.sleep(10)
body2 = browser.get_text()
idx2 = body2.find('architecture')
if idx2 >= 0:
    print("\n=== TECHNICAL ANALYSIS ===")
    print(body2[max(0,idx2-200):idx2+6000])

# AI.cc review
browser.goto('https://www.ai.cc/blogs/claude-fable-5-review-anthropic-most-powerful-ai-model-2026/')
time.sleep(10)
body3 = browser.get_text()
idx3 = body3.find('Mythos')
if idx3 >= 0:
    print("\n=== AI.CC REVIEW ===")
    print(body3[max(0,idx3-200):idx3+6000])

browser.close()
browser.shutdown()