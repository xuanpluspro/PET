"""Check dataset path routing without requiring CUDA or model dependencies."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DatasetPathTests(unittest.TestCase):
    def setUp(self):
        source = ast.parse((ROOT / "datasets/__init__.py").read_text())
        nodes = [n for n in source.body if
                 isinstance(n, ast.FunctionDef) and n.name == "build_dataset" or
                 isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "data_path" for t in n.targets)]
        def capture(split, args):
            return split, args.data_path
        self.env = dict(build_sha=capture, build_riceleaf=capture, build_wheat1799=capture)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<dataset routing>", "exec"), self.env)

    def test_explicit_paths_survive_train_and_validation(self):
        for dataset in self.env["data_path"]:
            with self.subTest(dataset=dataset):
                path = "/root/gpufree-data/apgcc-leaf-counting/data/RiceW3999"
                args = SimpleNamespace(dataset_file=dataset, data_path=path)
                for split in ("train", "val"):
                    self.assertEqual(self.env["build_dataset"](split, args), (split, path))

    def test_default_paths_when_omitted(self):
        for dataset, expected in self.env["data_path"].items():
            for value in (None, ""):
                with self.subTest(dataset=dataset, value=value):
                    args = SimpleNamespace(dataset_file=dataset, data_path=value)
                    self.assertEqual(self.env["build_dataset"]("train", args), ("train", expected))
            args = SimpleNamespace(dataset_file=dataset)
            self.assertEqual(self.env["build_dataset"]("val", args), ("val", expected))

    def test_unknown_dataset_rejected(self):
        with self.assertRaises(ValueError):
            self.env["build_dataset"]("train", SimpleNamespace(dataset_file="unknown"))

    def test_train_and_eval_parser_paths(self):
        for filename in ("main.py", "eval.py"):
            with self.subTest(filename=filename):
                tree = ast.parse((ROOT / filename).read_text())
                function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_args_parser")
                env = dict(argparse=argparse)
                exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), env)
                parser = env["get_args_parser"]()
                self.assertIsNone(parser.parse_args([]).data_path)
                self.assertEqual(parser.parse_args(["--data_path", "/custom path/data"]).data_path, "/custom path/data")


if __name__ == "__main__":
    unittest.main()
