"""Nuke startup hook for the standalone Draw Align tool."""

import os
import sys

_PLUGIN_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIRECTORY not in sys.path:
    sys.path.insert(0, _PLUGIN_DIRECTORY)
