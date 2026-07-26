"""Extract TRINITY paper details + Fugu benchmark data"""
import sys, time
sys.path = [p for p in sys.path if 'hermes' not in p.lower()]
sys.path.insert(0, 'C:/Users/user/helix_local/helix/benchmarks')
from web_search_stealth import StealthBrowser

browser = StealthBrowser(fingerprint='research_trinity')
browser.connect()

# TRINITY page
browser.goto('https://sakana.ai/trinity/')
time.sleep(10)
page = browser._page
if page is None:
    ctx = browser._browser.contexts[0]
    page = ctx.pages[0]
print("=== TRINITY ===")
print(f"Title: {page.title()}")
body = browser.get_text()
idx = body.find('TRINITY')
if idx < 0:
    idx = body.find('Evolved')
if idx >= 0:
    print(body[idx:idx+8000])
else:
    print(body[-6000:])

# Fugu benchmark page  
browser.goto('https://sakana.ai/fugu/')
time.sleep(10)
body2 = browser.get_text()
idx2 = body2.find('Quantitative Results')
if idx2 >= 0:
    print("\n=== FUGU BENCHMARKS ===")
    print(body2[idx2:idx2+5000])

browser.close()
browser.shutdown()