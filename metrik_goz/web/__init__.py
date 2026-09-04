"""
Web panel — for using the same core from a browser.

The panel does NOT open a new measurement path: every number you see comes from
the same tested functions in `metrik_goz.measure` and `metrik_goz.uncertainty`.
The layer here only collects the "image + clicked points" input and hands the
result back in a readable form.

    from metrik_goz.web import create_app
    create_app().run(port=8000)

or:

    metrik-goz panel --port 8000
"""

from .server import create_app

__all__ = ["create_app"]
