"""ChatGPT verification round 2 — simpler, longer wait"""
import sys, time
sys.path = [p for p in sys.path if 'hermes' not in p.lower()]
sys.path.insert(0, 'C:/Users/user/helix_local/helix/benchmarks')
from web_search_stealth import StealthBrowser

browser = StealthBrowser(fingerprint='chatgpt_v3')
browser.connect()
browser.goto('https://chatgpt.com')
time.sleep(12)

page = browser._page
if page is None:
    ctx = browser._browser.contexts[0]
    page = ctx.pages[0]
print(f"Title: {page.title()}")

# Shorter, more direct prompt
prompt = "My model gets PPL=3.82 on WikiText-2 (SEQ=128). I passed 6 sanity checks: labels shifted correctly, model predicts next not current token (85% accuracy), random labels give PPL=32M not 1, data splits are segregated, positions vary, formula is exp(mean NLL). Is this result legitimate? Short answer."
page.keyboard.type(prompt)
time.sleep(2)
page.keyboard.press('Enter')
print("Waiting 90s...")
time.sleep(90)

body = browser.get_text()
print(body[-6000:])

browser.close()
browser.shutdown()