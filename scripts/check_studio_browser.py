"""Optional browser smoke check against an existing, completed sample job."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:7860")
parser.add_argument("--job", required=True)
parser.add_argument("--chrome", default="/usr/bin/google-chrome")
args = parser.parse_args()
artifacts = Path(__file__).resolve().parents[1] / "runs" / "verification"
artifacts.mkdir(parents=True, exist_ok=True)
with urlopen(f"{args.url}/api/jobs/{args.job}") as response:
    original = json.load(response)["items"][0]

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chrome, headless=True)
        page = browser.new_page(viewport={"width":1440,"height":1000}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.evaluate("id => localStorage.setItem('sam3-job', id)", args.job)
        page.reload()
        page.wait_for_function("document.querySelector('#canvas').width > 300 && document.querySelector('#image-name').textContent !== '尚未选择图片'")
        page.wait_for_timeout(1000)
        assert page.locator("#current-count").inner_text() != "0 个目标"
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert page.evaluate("new Set(document.querySelector('#canvas').getContext('2d').getImageData(0,0,100,100).data).size > 20")
        page.screenshot(path=str(artifacts / "desktop.png"), full_page=True)
        page.locator("#mask-mode").click()
        page.screenshot(path=str(artifacts / "segmentation.png"), full_page=True)
        count = page.locator(".instance").count()
        page.locator("#draw-mode").click()
        bounds = page.locator("#canvas").bounding_box()
        page.mouse.move(bounds["x"]+30,bounds["y"]+30)
        page.mouse.down()
        page.mouse.move(bounds["x"]+110,bounds["y"]+100,steps=8)
        page.mouse.up()
        assert page.locator(".instance").count() == count+1
        page.locator("#save").click()
        page.wait_for_function("document.querySelector('#save-state').textContent === '已保存'")
        page.locator(".instance").last.get_by_role("button",name="删除目标").click()
        page.locator("#review").click()
        page.wait_for_function("document.querySelector('#save').disabled")
        with page.expect_download() as download:
            page.locator("#export").click()
        download.value.save_as(artifacts / "browser-export.zip")
        page.locator("#history-button").click()
        page.locator("#history").wait_for(state="visible")
        page.locator("#close-history").click()
        page.set_viewport_size({"width":390,"height":844})
        page.wait_for_timeout(300)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(artifacts / "mobile.png"), full_page=True)
        assert not errors, errors
        browser.close()
        print("Browser checks passed: preview, masks, drawing, deletion, save, review, export, history, desktop/mobile layout.")
finally:
    payload = json.dumps({"instances":original["instances"],"reviewed":original["reviewed"]}).encode()
    with urlopen(Request(f"{args.url}/api/jobs/{args.job}/items/0", data=payload, method="PUT", headers={"Content-Type":"application/json"})):
        pass
