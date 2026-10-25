"""Screenshot the map UI in headless Chromium (for checking the page). usage: python3 scripts/shot.py out.png [js...]"""
import os, sys
from playwright.sync_api import sync_playwright

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mc-shot.png"
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.path.expanduser("~/Library/Caches/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-mac-arm64/chrome-headless-shell"), args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": int(os.environ.get("VW", 1500)), "height": int(os.environ.get("VH", 920))}, device_scale_factor=int(os.environ.get("DSF", 1)), has_touch=bool(os.environ.get("TOUCH")))
    errs = []
    pg.on("console", lambda m: m.type in ("error", "warning") and errs.append(m.text))
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://127.0.0.1:8395/" + os.environ.get("Q", ""), wait_until="networkidle")
    pg.wait_for_timeout(1500)
    for js in sys.argv[2:]:
        pg.evaluate(js); pg.wait_for_timeout(700)
    pg.screenshot(path=out)
    print("errors:", errs or "none")
    b.close()
