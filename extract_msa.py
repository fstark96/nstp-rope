"""Extract MSA paper details"""
import sys, time
sys.path = [p for p in sys.path if 'hermes' not in p.lower()]
sys.path.insert(0, 'C:/Users/user/helix_local/helix/benchmarks')
from web_search_stealth import StealthBrowser

browser = StealthBrowser(fingerprint='msa_paper')
browser.connect()
browser.goto('https://xunhaolai.github.io/publication/2026-06-11-msa')
time.sleep(10)
body = browser.get_text()
idx = body.find('Sparse')
if idx >= 0:
    print(body[idx:idx+6000])
else:
    print(body[-4000:])
browser.close()
browser.shutdown()