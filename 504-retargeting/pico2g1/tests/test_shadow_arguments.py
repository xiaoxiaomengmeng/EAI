import argparse
import unittest
from unittest.mock import patch
from sim_scenes import add_scene_arguments
from test_scenes import sonic, light


class ShadowArgumentTests(unittest.TestCase):
    def test_shadows_are_opt_in_for_every_scene(self):
        parser = argparse.ArgumentParser()
        add_scene_arguments(parser)
        for scene in ('empty', 'xixi', 'pulse'):
            args = ['--scene', scene]
            self.assertFalse(parser.parse_args(args).render_shadows)
            self.assertTrue(parser.parse_args(args + ['--render-shadows']).render_shadows)

    def test_both_cli_modes_forward_shadow_choice_to_window_runner(self):
        for module in (sonic, light):
            for mode in ('replay', 'live'):
                for enabled in (False, True):
                    with self.subTest(app=module.__name__, mode=mode, enabled=enabled):
                        argv = ['sim', mode, 'unused-input', '--scene', 'xixi']
                        if enabled:
                            argv.append('--render-shadows')
                        with patch('sys.argv', argv), patch.object(module, 'run') as run:
                            module.main()
                        self.assertEqual(run.call_args.kwargs['render_shadows'], enabled)
