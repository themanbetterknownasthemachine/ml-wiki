"""
Bündelt alle Wiki-Markdown-Dateien in eine JSON-Datei,
die vom Chat-Widget geladen wird.
"""
import json
import os
from pathlib import Path

def bundle_wiki():
    wiki_dir = Path("wiki")
    bundle = []

    for md_file in sorted(wiki_dir.rglob("*.md")):
        # Stylesheets und Assets überspringen
        if "stylesheets" in str(md_file) or "assets" in str(md_file):
            continue

        relative_path = str(md_file.relative_to(wiki_dir)).replace("\\", "/")
        content = md_file.read_text(encoding="utf-8")

        bundle.append({
            "path": relative_path,
            "content": content
        })

    output_dir = wiki_dir / "assets"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "wiki-bundle.json"
    output_file.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Gebündelt: {len(bundle)} Dateien -> {output_file}")

if __name__ == "__main__":
    bundle_wiki()
