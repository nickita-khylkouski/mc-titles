"""Images for the README (needs serve.py on :8395):  python3 docs/shots.py"""
import os, sys
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
CHROME = os.path.expanduser("~/Library/Caches/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-mac-arm64/chrome-headless-shell")
CLUSTER = "[...document.querySelectorAll('#mode button')].find(b => b.textContent == 'Cluster').click()"
GAMES = "document.querySelector('#games button[data-v=\"1\"]').click()"
# points only: hide every panel and the label layer, refit to the whole window
BARE = """(() => {
  for (const id of ['side','time','hot','info','links','zoom','ov','tip','insbar']) { const e = document.getElementById(id); if (e) e.style.display = 'none'; }
  const q = (A, f) => { const s = Float32Array.from({length: 20000}, (_, i) => A[Math.floor(i * N / 20000)]).sort(); return s[Math.floor(f * (s.length - 1))]; };
  const x0 = q(X, .001), x1 = q(X, .999), y0 = q(Y, .001), y1 = q(Y, .999);
  S.s = ZOOM * Math.min(innerWidth / (x1 - x0), innerHeight / (y1 - y0)); [S.cx, S.cy] = centreFor((x0 + x1) / 2, (y0 + y1) / 2, S.s); dirty = true;
})()"""

def shot(pg, name, q="", js=(), w=1600, h=1000, wait=900):
    pg.set_viewport_size({"width": w, "height": h})
    pg.goto("http://127.0.0.1:8395/" + q, wait_until="networkidle")
    pg.wait_for_function("!document.getElementById('loading')", timeout=60000)
    pg.wait_for_timeout(800)
    for j in js:
        pg.evaluate(j); pg.wait_for_timeout(wait)
    pg.screenshot(path=os.path.join(OUT, name))
    print("wrote", name)

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(device_scale_factor=2)
    which = sys.argv[1:] or ["ui", "bare", "search", "cluster"]
    if "ui" in which:
        shot(pg, "map.png", js=[GAMES, CLUSTER])
    if "bare" in which:
        for m in ("", "rich", "thumb"):
            shot(pg, f"embed_{m or 'title'}.png", q=f"?map={m}" if m else "", w=1100, h=1100,
                 js=[GAMES, CLUSTER, BARE.replace("ZOOM", "0.9")])
    if "search" in which:
        shot(pg, "search.png", js=["(() => { qIn.value = 'herobrine'; runSearch('herobrine'); })()"], wait=6000)
    if "cluster" in which:
        shot(pg, "cluster.png", js=[CLUSTER, "document.querySelectorAll('#hotlist .row')[6].click()"], wait=1500)
    b.close()
