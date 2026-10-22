"""Match the main MuJoCo render aspect ratio to its Qt viewport."""
import mujoco


def fit_renderer(renderer, model, viewport):
    """Resize/create `renderer` for a QRect-like viewport or a (width, height) tuple."""
    if isinstance(viewport, tuple):
        width, height = viewport
    else:
        width, height = viewport.width(), viewport.height()
    width, height = max(1, width), max(1, height)
    # Bound the preview's GPU cost on very large windows while keeping its aspect.
    scale = min(1., 1600 / max(width, height))
    width, height = max(1, round(width * scale)), max(1, round(height * scale))
    if renderer is not None and (renderer.width, renderer.height) == (width, height):
        return renderer
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    if renderer is not None:
        # Free the old context on its own thread before creating the new one;
        # otherwise the frees hit the newly current context (Linux snow).
        if renderer._gl_context is not None:
            renderer._gl_context.make_current()
        renderer.close()
    return mujoco.Renderer(model, width=width, height=height)