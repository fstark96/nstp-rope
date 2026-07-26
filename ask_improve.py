"""Ask ChatGPT how to improve NSTP to beat top foundation models"""
import sys, time
sys.path = [p for p in sys.path if 'hermes' not in p.lower()]
sys.path.insert(0, 'C:/Users/user/helix_local/helix/benchmarks')
from web_search_stealth import StealthBrowser

browser = StealthBrowser(fingerprint='chatgpt_improve')
browser.connect()
browser.goto('https://chatgpt.com')
time.sleep(12)

page = browser._page
if page is None:
    ctx = browser._browser.contexts[0]
    page = ctx.pages[0]
print(f"Title: {page.title()}")

prompt = """My NSTP (Neuro-Symbolic Tensor Processor) language model currently gets PPL=3.82 on WikiText-2 at SEQ=128. I want to improve it to compete with top foundation models (Claude, GPT-4, DeepSeek, MiniMax, Kimi). My current architecture:

- Continuous HDC (High-Dimensional Computing) attention with FFT-based position binding
- TT-core Mixture of Experts (Tensor Train decomposition for router and expert params)
- 3 layers, 320 hidden dim, 4 heads, 2048 HSA dim, 4 experts, top-2 routing
- 39M parameters, trained on 2.4M tokens of WikiText-2

What's the most impactful thing I should do next to improve? Please give specific, actionable advice in order of priority."""
page.keyboard.type(prompt)
time.sleep(2)
page.keyboard.press('Enter')
print("Waiting 90s...")
time.sleep(90)

body = browser.get_text()
print(body[-8000:])

browser.close()
browser.shutdown()