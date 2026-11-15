"""
Build ../Appendix_Worked_Examples.pdf from Appendix_Worked_Examples.md.

    python build_pdf.py

Markdown -> standalone HTML (pandoc, with appendix.css embedded) -> PDF (headless
Chromium or Chrome). Needs `pandoc` and a Chromium-family browser on the machine;
set CHROME=/path/to/browser if it is not found automatically. The page uses
Liberation Serif, DejaVu Sans Mono and a CJK font (WenQuanYi Zen Hei) if present;
other fonts work, the layout just shifts slightly.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "Appendix_Worked_Examples.md")
CSS = os.path.join(HERE, "appendix.css")
OUT = os.path.join(os.path.dirname(HERE), "Appendix_Worked_Examples.pdf")


def find_browser():
    if os.getenv("CHROME"):
        return os.environ["CHROME"]
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome"):
        if shutil.which(name):
            return shutil.which(name)
    for pat in ("/opt/pw-browsers/chromium-*/chrome-linux*/chrome",
                os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome"),
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    sys.exit("No Chromium/Chrome found; set CHROME=/path/to/browser")


def main():
    if not shutil.which("pandoc"):
        sys.exit("pandoc not found")
    with tempfile.TemporaryDirectory() as tmp:
        html = os.path.join(tmp, "appendix.html")
        subprocess.run(["pandoc", SRC, "--from", "markdown", "--to", "html5", "--standalone",
                        "--embed-resources", "--no-highlight", "--css", CSS, "--output", html], check=True)
        subprocess.run([find_browser(), "--headless", "--no-sandbox", "--disable-gpu",
                        "--no-pdf-header-footer", f"--print-to-pdf={OUT}", "file://" + html],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
