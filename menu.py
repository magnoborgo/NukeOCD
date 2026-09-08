"""Register standalone Draw Align in Nuke's Edit/Node menu."""

import nuke

_application_menu = nuke.menu("Nuke")
_parent = _application_menu.findItem('Edit/Node')
if _parent is None:
    raise RuntimeError("Nuke menu path not found: Edit/Node")
_menu = _application_menu.findItem('Edit/Node/OCD')
if _menu is None:
    _menu = _parent.addMenu('OCD')
_menu.addCommand(
    'Draw Align Nodes',
    "import ocd_align_draw; ocd_align_draw.draw_align_nodes()",
    'shift+l',
)
