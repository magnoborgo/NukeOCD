"""Shared Nuke DAG coordinate and freehand hit-testing helpers."""

from __future__ import annotations

import math


def nodes_intersected_by_stroke(points, nodes, sample_step=4.0, dot_padding=0.0):
    """Return nodes in first-hit stroke order using graph-space rectangles."""

    ordered = []
    seen = set()
    for first, second in zip(points, points[1:]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        steps = max(1, int(math.ceil(math.hypot(dx, dy) / sample_step)))
        for step in range(steps + 1):
            fraction = step / float(steps)
            x, y = first[0] + dx * fraction, first[1] + dy * fraction
            for node in nodes:
                if id(node) in seen:
                    continue
                try:
                    padding = float(dot_padding) if node.Class() == "Dot" else 0.0
                except (AttributeError, RuntimeError):
                    padding = 0.0
                if (
                    node.xpos() - padding <= x <= node.xpos() + node.screenWidth() + padding
                    and node.ypos() - padding <= y <= node.ypos() + node.screenHeight() + padding
                ):
                    seen.add(id(node))
                    ordered.append(node)
    return tuple(ordered)


def screen_to_graph(point, widget, nuke):
    zoom = float(nuke.zoom())
    center = nuke.center()
    return (
        float(center[0]) + (point.x() - widget.width() / 2.0) / zoom,
        float(center[1]) + (point.y() - widget.height() / 2.0) / zoom,
    )


def event_global_point(event):
    global_position = getattr(event, "globalPosition", None)
    if callable(global_position):
        return global_position().toPoint()
    return event.globalPos()


def find_node_graph_widget(app, cursor_position=None):
    visible = [widget for widget in app.allWidgets() if widget.isVisible()]
    viewports = [
        widget
        for widget in visible
        if str(widget.objectName()).upper() == "DAG"
        and "dag" in str(widget.metaObject().className()).lower()
    ]
    if viewports:
        return max(viewports, key=lambda item: item.width() * item.height())
    containers = [
        widget
        for widget in visible
        if str(widget.objectName()).upper().startswith("DAG.")
    ]
    if containers:
        return max(containers, key=lambda item: item.width() * item.height())
    matches = [widget for widget in visible if _looks_like_dag(widget)]
    return max(matches, key=lambda item: item.width() * item.height()) if matches else None


def _looks_like_dag(widget):
    text = " ".join(
        str(value).lower()
        for value in (
            widget.objectName(),
            widget.metaObject().className(),
            widget.windowTitle(),
        )
    )
    return "dag" in text or "node graph" in text or "nodegraph" in text
