"""Build the ordered Markdown study guide with Pandoc 3.13 and Typst 0.15.1."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent.parent
VERSIONS = {"pandoc": "3.13", "typst": "0.15.1"}


def run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, **kwargs)
    if result.returncode:
        detail = result.stderr if isinstance(result.stderr, str) else result.stderr.decode("utf-8")
        raise RuntimeError("documentation command failed (%s):\n%s" % (command[0], detail))
    return result


def document(path, pandoc, revision):
    parsed = json.loads(run([pandoc, "--from=gfm", "--to=json", str(path)]).stdout)

    def rewrite(value):
        if isinstance(value, dict):
            kind, content = value.get("t"), value.get("c")
            if kind in ("Link", "Image"):
                target = content[-1][0]
                url = urlsplit(target)
                if not url.scheme and not url.netloc and url.path:
                    destination = (path.parent / unquote(url.path)).resolve()
                    relative = destination.relative_to(ROOT).as_posix()
                    if not destination.exists():
                        raise ValueError("missing guide target: %s" % target)
                    content[-1][0] = (
                        str(destination) if kind == "Image"
                        else "https://github.com/guptasameer112/tinyserve/blob/%s/%s%s"
                        % (revision, relative, "#" + url.fragment if url.fragment else "")
                    )
            if kind == "Header":
                content[1][0] = path.stem + "-" + content[1][0]
            for child in value.values():
                rewrite(child)
        elif isinstance(value, list):
            for child in value:
                rewrite(child)

    rewrite(parsed)
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pandoc", default="pandoc")
    parser.add_argument("--typst", default="typst")
    parser.add_argument("--revision", required=True, help="Full source commit for PDF links/provenance")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "tinyserve-study-guide.pdf")
    args = parser.parse_args()
    if len(args.revision) != 40 or any(char not in "0123456789abcdef" for char in args.revision):
        raise ValueError("--revision must be a full lowercase Git commit SHA")
    for tool, executable in (("pandoc", args.pandoc), ("typst", args.typst)):
        first = run([executable, "--version"], text=True).stdout.splitlines()[0]
        if first.split()[1] != VERSIONS[tool]:
            raise ValueError("expected %s %s, got %s" % (tool, VERSIONS[tool], first))
    guide = ROOT / "docs" / "guide"
    chapters = sorted(guide.glob("[0-9][0-9]-*.md"))
    if [int(path.name[:2]) for path in chapters] != list(range(1, len(chapters) + 1)) or not chapters:
        raise ValueError("expected consecutively numbered guide chapters")
    inputs = [guide / "README.md", *chapters, ROOT / "docs" / "CORE_REVIEW.md"]
    combined = None
    for path in inputs:
        parsed = document(path, args.pandoc, args.revision)
        if combined is None:
            combined = parsed
        else:
            combined["blocks"].append({"t": "RawBlock", "c": ["typst", "#pagebreak()"]})
            combined["blocks"].extend(parsed["blocks"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.pandoc, "--from=json", "--standalone", "--toc", "--toc-depth=2",
        "--pdf-engine=" + args.typst, "--output=" + str(args.output.resolve()),
        "--include-in-header=" + str(guide / "pdf.typ"),
        "--metadata=title:TinyServe: the core study guide",
        "--metadata=subtitle:From tokens to serving and honest benchmarks",
        "--metadata=author:TinyServe learning project",
        "--variable=papersize:a4", "--variable=fontsize:10pt",
    ]
    built = run(command, input=json.dumps(combined).encode("utf-8"), cwd=ROOT)
    if built.stderr:
        print(built.stderr.decode("utf-8"))
    if not args.output.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("documentation build did not produce a PDF")
    sources = inputs + [guide / "pdf.typ", guide / "img" / "A2_roofline.png"]
    manifest = {
        "source_commit": args.revision,
        "chapter_count": len(chapters),
        "tools": VERSIONS,
        "sources": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sources},
        "pdf_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Built %s (%d chapters) and %s" % (args.output, len(chapters), manifest_path))


if __name__ == "__main__":
    main()
