"""Exercise the lightweight loaders without Linux bridge runtime dependencies."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def isolated_loader(config_path, loader_path, offsets_path):
    config_tree = ast.parse(config_path.read_text())
    names = next(n.value for n in config_tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'ROTATION_OFFSET_JOINTS' for t in n.targets))
    loader_tree = ast.parse(loader_path.read_text())
    function = next(n for n in loader_tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'load_rotation_offsets')
    namespace = dict(np=np, json=json, ROTATION_OFFSETS_PATH=offsets_path,
                     ROTATION_OFFSET_JOINTS=ast.literal_eval(names))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(loader_path), 'exec'), namespace)
    return namespace['load_rotation_offsets']


class RotationConfigTests(unittest.TestCase):
    def test_invalid_json_falls_back_and_valid_offsets_load(self):
        for config_rel, loader_rel in (
            ('g1_sonic/sonic_config.py', 'g1_sonic/sonic_recording.py'),
            ('g1_sonic_bridge/bridge_config.py', 'g1_sonic_bridge/bridge_source.py'),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'offsets.json'
                load = isolated_loader(ROOT/config_rel, ROOT/loader_rel, path)
                for content in ('{}', '[]', 'null', '{invalid json'):
                    with self.subTest(module=loader_rel, content=content):
                        path.write_text(content)
                        self.assertTrue(all(np.array_equal(value, np.zeros(3)) for value in load().values()))
                path.write_text(json.dumps({'left_elbow': [1, 2, 3]}))
                np.testing.assert_array_equal(load()['left_elbow'], [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
