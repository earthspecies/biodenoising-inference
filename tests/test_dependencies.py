"""Installation metadata regression checks (no model or audio device required).

Run with ``python -m unittest discover -s tests -v`` after installing
``requirements-test.txt``. Set BIODENOISING_WHEEL to a built wheel's path to
also check its actual published dependency metadata.
"""

import ast
from email.parser import BytesParser
import os
from pathlib import Path
import unittest
from zipfile import ZipFile

from packaging.requirements import Requirement

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def requirements_by_name(values):
    requirements = [Requirement(value) for value in values]
    return {requirement.name: requirement for requirement in requirements}


def dependency_sources():
    requirements = [
        line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]["dependencies"]
    setup_tree = ast.parse((ROOT / "setup.py").read_text())
    setup = next(
        ast.literal_eval(node.value)
        for node in setup_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "REQUIRED"
                for target in node.targets)
    )
    return {"requirements.txt": requirements, "pyproject.toml": project, "setup.py": setup}


class DependencyTests(unittest.TestCase):
    def test_installation_routes_declare_the_same_dependencies(self):
        sources = dependency_sources()
        expected = requirements_by_name(sources["requirements.txt"])
        for name, values in sources.items():
            with self.subTest(source=name):
                self.assertEqual(requirements_by_name(values), expected)

    def test_torch_and_torchaudio_allow_2_8_but_not_removed_api_release(self):
        for name, values in dependency_sources().items():
            requirements = requirements_by_name(values)
            for package in ("torch", "torchaudio"):
                with self.subTest(source=name, package=package):
                    specifier = requirements[package].specifier
                    self.assertIn("2.8.0", specifier)
                    self.assertNotIn("2.9.0", specifier)
                    self.assertNotIn("2.10.0", specifier)

    @unittest.skipUnless(os.environ.get("BIODENOISING_WHEEL"), "no wheel path supplied")
    def test_built_wheel_dependencies(self):
        with ZipFile(os.environ["BIODENOISING_WHEEL"]) as wheel:
            metadata_path = next(name for name in wheel.namelist()
                                 if name.endswith(".dist-info/METADATA"))
            metadata = BytesParser().parsebytes(wheel.read(metadata_path))
        self.assertEqual(
            requirements_by_name(metadata.get_all("Requires-Dist", [])),
            requirements_by_name(dependency_sources()["requirements.txt"]),
        )


if __name__ == "__main__":
    unittest.main()
