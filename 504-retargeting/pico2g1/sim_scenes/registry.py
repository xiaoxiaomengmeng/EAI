"""Runtime-only scene locations; authoring files are intentionally not dependencies."""
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / 'assets'
INTERACTIVE_SCENES = {
    'pulse': ASSETS / 'pulse_arena',
    'strike': ASSETS / 'strike_arena',
    'orbit': ASSETS / 'orbit_arena',
    'switchyard': ASSETS / 'switchyard_arena',
}
SCENE_CHOICES = ('empty', 'xixi', *INTERACTIVE_SCENES)
