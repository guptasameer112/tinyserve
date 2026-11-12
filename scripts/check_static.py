"""Static-only checks: parse sources/notebooks, check links and optional archives."""
import argparse
import ast
import json
import re
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent.parent


def check_links(path):
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    for target in re.findall(r"!?\[[^\]]*\]\(([^)\n]+)\)", text):
        target = target.split(' "', 1)[0].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        destination = (path.parent / unquote(parsed.path)).resolve()
        if (destination != ROOT and ROOT not in destination.parents) or not destination.exists():
            raise ValueError("%s: broken or out-of-repository link %s" % (path, target))


def check_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        expected = {
            file.relative_to(ROOT).as_posix()
            for package in ("tinyserve", "bench")
            for file in (ROOT / package).rglob("*.py")
        }
        missing = expected - names
        if missing:
            raise ValueError("wheel missing modules: %s" % sorted(missing))
        if any(name.startswith(("tests/", "notebooks/")) for name in names):
            raise ValueError("wheel must not contain tests or notebooks")
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")
        if "Provides-Extra: server" not in metadata or "# tinyserve" not in metadata:
            raise ValueError("wheel must include serving extras and README metadata")
    print("PASS: wheel contains all tinyserve/bench modules, README and serving extras")


def check_sdist(path):
    with tarfile.open(path, "r:gz") as archive:
        names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
        expected = {"pyproject.toml", "README.md", "MASTERPLAN.md"}
        for directory in ("tinyserve", "bench", "tests", "scripts", "notebooks", "docs", "results"):
            expected.update(
                file.relative_to(ROOT).as_posix()
                for file in (ROOT / directory).rglob("*")
                if file.is_file() and file.suffix in (".py", ".md", ".typ", ".png", ".ipynb", ".json")
            )
        if expected - names:
            raise ValueError("source archive missing files: %s" % sorted(expected - names))
    print("PASS: source archive includes package, benchmark, tests, lessons, docs and evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--sdist", type=Path)
    args = parser.parse_args()
    sources = []
    for directory in ("tinyserve", "bench", "tests", "scripts"):
        sources.extend((ROOT / directory).rglob("*.py"))
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    cells = 0
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook["nbformat"] != 4:
            raise ValueError("unsupported notebook format: %s" % path)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            source = "\n".join(
                "" if line.lstrip().startswith(("%", "!")) else line
                for line in source.splitlines()
            )
            ast.parse(source, filename="%s cell %s" % (path, index))
            cells += 1
    chapters = sorted((ROOT / "docs" / "guide").glob("[0-9][0-9]-*.md"))
    if not chapters or [int(path.name[:2]) for path in chapters] != list(range(1, len(chapters) + 1)):
        raise ValueError("guide chapters must be numbered consecutively from 01")
    documents = [ROOT / "README.md", ROOT / "MASTERPLAN.md", ROOT / "results" / "README.md"]
    documents.extend((ROOT / "docs").rglob("*.md"))
    for path in documents:
        check_links(path)
    results = list((ROOT / "results").glob("*.json"))
    for path in results:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not all(key in result for key in ("name", "timestamp_utc", "env", "data")):
            raise ValueError("result envelope missing required fields: %s" % path)
    if args.wheel:
        check_wheel(args.wheel)
    if args.sdist:
        check_sdist(args.sdist)
    print(
        "PASS: %d Python sources, %d notebooks/%d code cells, %d chapters, "
        "%d linked documents, %d result envelopes; no code cells or tests executed"
        % (len(sources), len(notebooks), cells, len(chapters), len(documents), len(results))
    )


if __name__ == "__main__":
    main()
