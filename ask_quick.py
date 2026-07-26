"""Quick follow-up: what's the single highest-impact change?"""
import sys, time
sys.path = [p for p in sys.path if 'hermes' not in p.lower()]
sys.path.insert(0, 'C:/Users/user/helix_local/helix/benchmarks')
from web_search_stealth import StealthBrowser

browser = StealthBrowser(fingerprint='chatgpt_quick2')
browser.connect()
browser.goto('https://chatgpt.com')
time.sleep(12)

page = browser._page
if page is None:
    ctx = browser._browser.contexts[0]
    page = ctx.pages[0]

# New conversation to avoid history limit
prompt = "I'm training NSTP (custom HDC+TT-MoE language model) on only 2.4M tokens WikiText-2. I can only run on my local GPU (24GB). What's the ONE change with highest ROI — train on MORE data, or make my architecture more compute-efficient? I have RTX 4070 Ti Super 16GB."
page.keyboard.type(prompt)
time.sleep(2)
page.keyboard.press('Enter')
print("Waiting 90s...")
time.sleep(90)

body = browser.get_text()
idx = body.find('RTX')
print(body[idx:idx+5000] if idx >= 0 else body[-5000:])

browser.close()
browser.shutdown()