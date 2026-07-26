"""Extract Sakana research papers + set up data pipeline"""
import sys, time
sys.path = [p for p in sys.path if 'hermes' not in p.lower()]
sys.path.insert(0, 'C:/Users/user/helix_local/helix/benchmarks')
from web_search_stealth import StealthBrowser

browser = StealthBrowser(fingerprint='research_fugu')
browser.connect()
browser.goto('https://sakana.ai/learning-to-orchestrate/')
time.sleep(10)

page = browser._page
if page is None:
    ctx = browser._browser.contexts[0]
    page = ctx.pages[0]
print(f"Title: {page.title()}")

body = browser.get_text()
# Skip navigation, find the research content
idx = body.find('Learning to Orchestrate')
if idx < 0:
    idx = body.find('orchestrate')
if idx >= 0:
    print(body[idx:idx+8000])
else:
    print(body[-8000:])

browser.close()
browser.shutdown()