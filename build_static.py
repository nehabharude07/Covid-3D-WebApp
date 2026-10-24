"""Writes a single self-contained covid_3d_atlas.html (data inlined) - no server needed."""
from jinja2 import Environment, FileSystemLoader
from utils.data_utils import build_payload

html = Environment(loader=FileSystemLoader("templates")).get_template("index.html").render(data=build_payload())
open("covid_3d_atlas.html", "w", encoding="utf-8").write(html)
print("written", len(html) // 1024, "KB")
