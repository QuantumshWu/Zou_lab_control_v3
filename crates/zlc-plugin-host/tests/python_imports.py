"""Every `from <our module> import name` in the science sources names something that module
has, including imports inside functions that run only on one path (a calibration's save):
a missing name is found here, not on the lab machine mid-run."""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[3] / "science" / "src"
PACKAGES = {path.name for path in ROOT.iterdir() if path.is_dir() and path.name.isidentifier()}


def _bound(tree):
    """The names a module binds at its top level (also inside its if/try blocks), and
    whether it may bind more (a star import or a module __getattr__)."""
    names, open_ended = set(), False
    for statement in tree.body:
        for node in [statement] if not isinstance(statement, (ast.If, ast.Try)) else ast.walk(statement):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                open_ended |= node.name == "__getattr__" and node in tree.body
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    names |= {item.id for item in ast.walk(target) if isinstance(item, ast.Name)}
            elif isinstance(node, ast.Import):
                names |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                open_ended |= any(alias.name == "*" for alias in node.names)
                names |= {alias.asname or alias.name for alias in node.names}
    return names, open_ended


def _provides(dotted, cache={}):
    """What `from dotted import …` may name: its bindings and submodules; None when open-ended;
    False when there is no such module."""
    if dotted not in cache:
        base = ROOT.joinpath(*dotted.split("."))
        init = base / "__init__.py"
        if base.is_dir():
            names, open_ended = _bound(ast.parse(init.read_text(encoding="utf8"))) if init.exists() else (set(), False)
            names |= {path.stem for path in base.iterdir() if path.suffix == ".py" or path.is_dir()}
        elif base.with_suffix(".py").exists():
            names, open_ended = _bound(ast.parse(base.with_suffix(".py").read_text(encoding="utf8")))
        else:
            names, open_ended = False, False
        cache[dotted] = None if open_ended else names
    return cache[dotted]


class ImportTests(unittest.TestCase):
    def test_every_imported_name_exists(self):
        missing = []
        for path in sorted(ROOT.rglob("*.py")):
            relative = path.relative_to(ROOT)
            package = list(relative.parts[:-1])
            for node in ast.walk(ast.parse(path.read_text(encoding="utf8"))):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level:
                    parts = package[:len(package) - node.level + 1] + ([node.module] if node.module else [])
                    dotted = ".".join(parts)
                else:
                    dotted = node.module or ""
                if dotted.split(".")[0] not in PACKAGES:
                    continue
                provided = _provides(dotted)
                if provided is False:
                    missing.append(f"{relative}:{node.lineno} no module {dotted}")
                elif provided is not None:
                    missing += [f"{relative}:{node.lineno} {dotted}.{alias.name}" for alias in node.names
                                if alias.name != "*" and alias.name not in provided]
        self.assertEqual(missing, [])


if __name__ == "__main__": unittest.main()
