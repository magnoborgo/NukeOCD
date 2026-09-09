"""Gesture-based horizontal or vertical node alignment in Nuke's DAG."""

from __future__ import annotations

from collections import deque
from statistics import median

from ocd_dag_gesture import (
    event_global_point as _event_global_point,
    find_node_graph_widget as _find_node_graph_widget,
    nodes_intersected_by_stroke,
    screen_to_graph as _screen_to_graph,
)


_ACTIVE_ALIGN_CAPTURE = None


def alignment_axis(start, end):
    """Return ``y`` for a horizontal gesture or ``x`` for a vertical one."""

    return "y" if abs(end[0] - start[0]) >= abs(end[1] - start[1]) else "x"


def aligned_coordinates(nodes, axis):
    """Return node positions aligned to the first node without mutating them."""

    nodes = tuple(nodes)
    if not nodes:
        return ()
    anchor_x, anchor_y = int(nodes[0].xpos()), int(nodes[0].ypos())
    anchor_center_x = anchor_x + float(nodes[0].screenWidth()) / 2.0
    anchor_center_y = anchor_y + float(nodes[0].screenHeight()) / 2.0
    if axis == "y":
        return tuple(
            (
                int(node.xpos()),
                int(round(anchor_center_y - float(node.screenHeight()) / 2.0)),
            )
            for node in nodes
        )
    if axis == "x":
        return tuple(
            (
                int(round(anchor_center_x - float(node.screenWidth()) / 2.0)),
                int(node.ypos()),
            )
            for node in nodes
        )
    raise ValueError("alignment axis must be 'x' or 'y'")


def resolve_alignment_overlaps(nodes, coordinates, axis, gap=20):
    """Pack aligned rectangles without reversing connected node flow."""

    nodes = tuple(nodes)
    coordinates = tuple(coordinates)
    if len(nodes) < 2:
        return coordinates
    free_index = 0 if axis == "y" else 1
    regular_nodes = [node for node in nodes if not _is_dot(node)]
    regular_width = int(round(median(
        [int(node.screenWidth()) for node in regular_nodes] or [80]
    )))
    regular_height = int(round(median(
        [int(node.screenHeight()) for node in regular_nodes] or [20]
    )))
    actual_sizes = (
        [int(node.screenWidth()) for node in nodes]
        if axis == "y"
        else [int(node.screenHeight()) for node in nodes]
    )
    regular_size = regular_width if axis == "y" else regular_height
    effective_sizes = [
        regular_size if _is_dot(node) else actual_size
        for node, actual_size in zip(nodes, actual_sizes)
    ]
    offsets = [
        (actual_size - effective_size) / 2.0
        for actual_size, effective_size in zip(actual_sizes, effective_sizes)
    ]
    effective_desired = [
        float(coordinate[free_index]) + offset
        for coordinate, offset in zip(coordinates, offsets)
    ]
    ordered = _flow_order_indices(nodes, axis)
    packed = {}
    previous_end = None
    for index in ordered:
        desired = effective_desired[index]
        position = (
            desired
            if previous_end is None
            else max(desired, previous_end + int(gap))
        )
        packed[index] = position
        previous_end = position + effective_sizes[index]

    translation = effective_desired[0] - packed[0]
    result = []
    for index, coordinate in enumerate(coordinates):
        free_position = int(round(packed[index] + translation - offsets[index]))
        if axis == "y":
            result.append((free_position, coordinate[1]))
        else:
            result.append((coordinate[0], free_position))
    return tuple(result)


def offset_around_external_obstacles(nodes, coordinates, all_nodes, axis, gap=20):
    """Translate the packed row/column along its axis past fixed obstacles."""

    nodes = tuple(nodes)
    coordinates = tuple(coordinates)
    aligned_keys = {_node_key(node) for node in nodes}
    obstacles = [
        node for node in all_nodes if _node_key(node) not in aligned_keys
    ]
    if not nodes or not obstacles:
        return coordinates

    regular_nodes = [node for node in nodes if not _is_dot(node)]
    regular_width = int(round(median(
        [int(node.screenWidth()) for node in regular_nodes] or [80]
    )))
    regular_height = int(round(median(
        [int(node.screenHeight()) for node in regular_nodes] or [20]
    )))
    forbidden = []
    for node, (x, y) in zip(nodes, coordinates):
        if _is_dot(node):
            width, height = regular_width, regular_height
            left = x + node.screenWidth() / 2.0 - width / 2.0
            top = y + node.screenHeight() / 2.0 - height / 2.0
        else:
            width, height = node.screenWidth(), node.screenHeight()
            left, top = float(x), float(y)
        right, bottom = left + width, top + height
        for obstacle in obstacles:
            obstacle_left = float(obstacle.xpos())
            obstacle_top = float(obstacle.ypos())
            obstacle_right = obstacle_left + float(obstacle.screenWidth())
            obstacle_bottom = obstacle_top + float(obstacle.screenHeight())
            if axis == "y":
                cross_overlap = (
                    bottom > obstacle_top - gap
                    and top < obstacle_bottom + gap
                )
                if cross_overlap:
                    forbidden.append(
                        (obstacle_left - gap - right, obstacle_right + gap - left)
                    )
            else:
                cross_overlap = (
                    right > obstacle_left - gap
                    and left < obstacle_right + gap
                )
                if cross_overlap:
                    forbidden.append(
                        (obstacle_top - gap - bottom, obstacle_bottom + gap - top)
                    )
    translation = _nearest_allowed_translation(forbidden)
    if not translation:
        return coordinates
    if axis == "y":
        return tuple((int(round(x + translation)), y) for x, y in coordinates)
    return tuple((x, int(round(y + translation))) for x, y in coordinates)


def colliding_external_obstacles(nodes, coordinates, all_nodes, gap=20):
    """Return fixed nodes intersecting the proposed aligned layout and margin."""

    nodes = tuple(nodes)
    coordinates = tuple(coordinates)
    aligned_keys = {_node_key(node) for node in nodes}
    obstacles = [
        node for node in all_nodes if _node_key(node) not in aligned_keys
    ]
    regular_nodes = [node for node in nodes if not _is_dot(node)]
    regular_width = int(round(median(
        [int(node.screenWidth()) for node in regular_nodes] or [80]
    )))
    regular_height = int(round(median(
        [int(node.screenHeight()) for node in regular_nodes] or [20]
    )))
    collided = []
    collided_keys = set()
    for node, (x, y) in zip(nodes, coordinates):
        if _is_dot(node):
            width, height = regular_width, regular_height
            left = x + node.screenWidth() / 2.0 - width / 2.0
            top = y + node.screenHeight() / 2.0 - height / 2.0
        else:
            width, height = node.screenWidth(), node.screenHeight()
            left, top = float(x), float(y)
        right, bottom = left + width, top + height
        for obstacle in obstacles:
            key = _node_key(obstacle)
            if key in collided_keys:
                continue
            obstacle_left = float(obstacle.xpos())
            obstacle_top = float(obstacle.ypos())
            obstacle_right = obstacle_left + float(obstacle.screenWidth())
            obstacle_bottom = obstacle_top + float(obstacle.screenHeight())
            if (
                right > obstacle_left - gap
                and left < obstacle_right + gap
                and bottom > obstacle_top - gap
                and top < obstacle_bottom + gap
            ):
                collided_keys.add(key)
                collided.append(obstacle)
    return tuple(collided)


def _nearest_allowed_translation(intervals):
    merged = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    for start, end in merged:
        if start < 0 < end or start == 0 or end == 0:
            return start if abs(start) <= abs(end) else end
    return 0.0


def _flow_order_indices(nodes, axis):
    """Preserve original left-to-right or top-to-bottom ordering exactly."""

    return sorted(
        range(len(nodes)),
        key=lambda index: (
            nodes[index].xpos() if axis == "y" else nodes[index].ypos(),
            index,
        ),
    )


def adjust_to_external_inputs(aligned_nodes, coordinates, all_nodes, axis):
    """Apply bounded input-first/output-second perpendicular preferences."""

    aligned_nodes = tuple(aligned_nodes)
    all_nodes = tuple(all_nodes)
    aligned_keys = {_node_key(node) for node in aligned_nodes}
    by_key = {_node_key(node): node for node in all_nodes}
    outside_outputs = {key: [] for key in aligned_keys}
    for destination in all_nodes:
        destination_key = _node_key(destination)
        if destination_key in aligned_keys:
            continue
        for input_index in range(int(destination.inputs())):
            source = destination.input(input_index)
            source_key = _node_key(source) if source is not None else None
            if source_key in aligned_keys:
                outside_outputs[source_key].append(destination)

    regular_nodes = [node for node in aligned_nodes if not _is_dot(node)]
    regular_extent = int(round(median(
        ([int(node.screenWidth()) for node in regular_nodes] or [80])
        if axis == "y"
        else ([int(node.screenHeight()) for node in regular_nodes] or [20])
    )))
    adjusted = []
    for node_index, (destination, (destination_x, destination_y)) in enumerate(
        zip(aligned_nodes, coordinates)
    ):
        if node_index == 0:
            adjusted.append((destination_x, destination_y))
            continue
        outside_sources = []
        for input_index in range(int(destination.inputs())):
            source = destination.input(input_index)
            source_key = _node_key(source) if source is not None else None
            if source_key in by_key and source_key not in aligned_keys:
                outside_sources.append(by_key[source_key])

        current = destination_x if axis == "y" else destination_y
        extent = regular_extent if _is_dot(destination) else (
            destination.screenWidth() if axis == "y" else destination.screenHeight()
        )
        maximum_shift = 2.0 * float(extent)

        def qualifying(neighbors):
            preferences = []
            for neighbor in neighbors:
                if axis == "y":
                    target = (
                        neighbor.xpos()
                        + float(neighbor.screenWidth()) / 2.0
                        - float(destination.screenWidth()) / 2.0
                    )
                else:
                    target = (
                        neighbor.ypos()
                        + float(neighbor.screenHeight()) / 2.0
                        - float(destination.screenHeight()) / 2.0
                    )
                shift = abs(float(target) - float(current))
                if shift <= maximum_shift:
                    preferences.append(
                        (shift, _node_key(neighbor), int(round(target)))
                    )
            return preferences

        preferences = qualifying(outside_sources)
        if not preferences:
            preferences = qualifying(
                outside_outputs.get(_node_key(destination), ())
            )
        if preferences:
            target = min(preferences)[2]
            if axis == "y":
                destination_x = target
            else:
                destination_y = target
        adjusted.append((destination_x, destination_y))
    return tuple(adjusted)


def include_connected_between(crossed, nodes):
    """Insert shortest-path nodes between consecutive crossed nodes."""

    crossed = tuple(crossed)
    nodes = tuple(nodes)
    if len(crossed) < 2:
        return crossed
    by_key = {_node_key(node): node for node in nodes}
    neighbors = _connection_neighbors(nodes, by_key)

    result = [crossed[0]]
    included = {_node_key(crossed[0])}
    for first, last in zip(crossed, crossed[1:]):
        path = _shortest_path(_node_key(first), _node_key(last), neighbors)
        additions = (
            [by_key[node_key] for node_key in path[1:]]
            if path
            else [last]
        )
        for node in additions:
            node_key = _node_key(node)
            if node_key not in included:
                result.append(node)
                included.add(node_key)
    return tuple(result)


def include_connected_dots(selected, nodes, reference_nodes=None, padding=30.0):
    """Add attached Dot chains only inside the crossed nodes' local range."""

    selected = tuple(selected)
    nodes = tuple(nodes)
    reference_nodes = tuple(reference_nodes or selected)
    reference_keys = {_node_key(node) for node in reference_nodes}
    bounds = _expanded_node_bounds(reference_nodes, float(padding))
    by_key = {_node_key(node): node for node in nodes}
    neighbors = _connection_neighbors(nodes, by_key)
    result = list(selected)
    included = {_node_key(node) for node in selected}
    queue = deque(included)
    while queue:
        current = queue.popleft()
        for neighbor_key in neighbors.get(current, ()):
            if neighbor_key in included:
                continue
            neighbor = by_key[neighbor_key]
            if not _is_dot(neighbor) or not _node_center_in_bounds(neighbor, bounds):
                continue
            included.add(neighbor_key)
            result.append(neighbor)
            queue.append(neighbor_key)
    # Shortest-path completion may already have inserted a remotely routed Dot.
    # Keep explicitly crossed Dots, but remove automatically added ones outside
    # the same local X/Y range.
    return tuple(
        node
        for node in result
        if _node_key(node) in reference_keys
        or str(node.Class()) != "Dot"
        or _node_center_in_bounds(node, bounds)
    )


def _expanded_node_bounds(nodes, padding):
    if not nodes:
        return None
    return (
        min(float(node.xpos()) for node in nodes) - padding,
        min(float(node.ypos()) for node in nodes) - padding,
        max(float(node.xpos() + node.screenWidth()) for node in nodes) + padding,
        max(float(node.ypos() + node.screenHeight()) for node in nodes) + padding,
    )


def _node_center_in_bounds(node, bounds):
    if bounds is None:
        return False
    center_x = float(node.xpos()) + float(node.screenWidth()) / 2.0
    center_y = float(node.ypos()) + float(node.screenHeight()) / 2.0
    return bounds[0] <= center_x <= bounds[2] and bounds[1] <= center_y <= bounds[3]


def _is_dot(node):
    try:
        return str(node.Class()) == "Dot"
    except (AttributeError, RuntimeError):
        return False


def _connection_neighbors(nodes, by_key=None):
    by_key = by_key or {_node_key(node): node for node in nodes}
    neighbors = {key: [] for key in by_key}
    for destination in nodes:
        destination_key = _node_key(destination)
        for input_index in range(int(destination.inputs())):
            source = destination.input(input_index)
            source_key = _node_key(source) if source is not None else None
            if source_key not in by_key:
                continue
            neighbors[source_key].append(destination_key)
            neighbors[destination_key].append(source_key)
    return neighbors


def _shortest_path(start, end, neighbors):
    queue = deque([start])
    previous = {start: None}
    while queue:
        current = queue.popleft()
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = previous[current]
            return tuple(reversed(path))
        for neighbor in neighbors.get(current, ()):
            if neighbor not in previous:
                previous[neighbor] = current
                queue.append(neighbor)
    return ()


def _node_key(node):
    """Return a stable key across distinct Nuke PythonObject wrappers."""

    full_name = getattr(node, "fullName", None)
    if callable(full_name):
        try:
            return str(full_name())
        except (RuntimeError, TypeError, ValueError):
            pass
    name = getattr(node, "name", None)
    if callable(name):
        try:
            return str(name())
        except (RuntimeError, TypeError, ValueError):
            pass
    return "python:{0}".format(id(node))


def draw_align_nodes():
    """Toggle the continuous alignment gesture session."""

    global _ACTIVE_ALIGN_CAPTURE
    if _ACTIVE_ALIGN_CAPTURE is not None:
        try:
            if _ACTIVE_ALIGN_CAPTURE.active:
                _ACTIVE_ALIGN_CAPTURE.finish()
                return None
        except RuntimeError:
            pass
        _ACTIVE_ALIGN_CAPTURE = None

    try:
        import nuke
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except ImportError:
            from PySide2 import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise RuntimeError("Draw Align Nodes requires Nuke GUI mode") from exc

    candidates = tuple(
        node for node in nuke.allNodes() if node.Class() != "BackdropNode"
    )
    app = QtWidgets.QApplication.instance()
    dag = _find_node_graph_widget(app, QtGui.QCursor.pos())
    if dag is None:
        nuke.message("Could not find the visible Node Graph. Focus it and try again.")
        return None

    dag_origin = dag.mapToGlobal(QtCore.QPoint(0, 0))
    dag_size = dag.size()

    class AlignCapture(QtWidgets.QWidget):
        def __init__(self):
            super().__init__(parent=None)
            self.setWindowFlags(
                QtCore.Qt.Tool
                | QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
            )
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
            self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.ocd_cursor = self._ocd_cursor(QtGui.QColor(255, 183, 77, 245))
            self.invalid_cursor = self._ocd_cursor(QtGui.QColor(229, 57, 53, 250))
            self.setCursor(self.ocd_cursor)
            self.setGeometry(QtCore.QRect(dag_origin, dag_size))
            self.start_point = None
            self.end_point = None
            self.points = []
            self.axis = None
            self.cursor_refresh_pending = False
            self.invalid_cursor_active = False
            self._collision_markers = []
            self.active = True

        def _ocd_cursor(self, background):
            pixmap = QtGui.QPixmap(32, 32)
            pixmap.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(pixmap)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setPen(QtGui.QPen(QtGui.QColor(35, 35, 35, 255), 1))
            painter.setBrush(background)
            painter.drawRoundedRect(QtCore.QRectF(3, 7, 28, 19), 4, 4)
            font = QtGui.QFont()
            font.setBold(True)
            font.setPixelSize(10)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(20, 20, 20, 255))
            painter.drawText(
                QtCore.QRect(3, 7, 28, 19),
                QtCore.Qt.AlignCenter,
                "OCD",
            )
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 255), 1))
            painter.drawLine(0, 0, 6, 0)
            painter.drawLine(0, 0, 0, 6)
            painter.end()
            return QtGui.QCursor(pixmap, 0, 0)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.fillRect(event.rect(), QtGui.QColor(0, 0, 0, 1))
            if self.start_point is None or self.end_point is None:
                return
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor(255, 183, 77, 240),
                    4,
                    QtCore.Qt.SolidLine,
                    QtCore.Qt.RoundCap,
                )
            )
            if len(self.points) > 1:
                painter.drawPolyline(QtGui.QPolygon(self.points))
            painter.setPen(QtGui.QColor(255, 255, 255, 245))
            if self.axis is not None:
                message = "ALIGN Y" if self.axis == "y" else "ALIGN X"
                painter.drawText(self.end_point + QtCore.QPoint(10, -10), message)

        def eventFilter(self, watched, event):
            event_type = event.type()
            if (
                self.start_point is None
                and event_type
                in (QtCore.QEvent.MouseMove, QtCore.QEvent.Wheel, QtCore.QEvent.CursorChange)
            ):
                self._schedule_cursor_refresh()
            if event_type == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Escape:
                self.finish()
                return True
            if event_type == QtCore.QEvent.MouseButtonPress:
                if not self.geometry().contains(_event_global_point(event)):
                    return False
                if (
                    event.button() != QtCore.Qt.LeftButton
                    or event.modifiers() != QtCore.Qt.NoModifier
                ):
                    return False
                point = self.mapFromGlobal(_event_global_point(event))
                self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
                super(AlignCapture, self).show()
                self.raise_()
                self.grabMouse()
                self.start_point = point
                self.end_point = point
                self.points = [point]
                self.axis = None
                self.update()
                return True
            if event_type == QtCore.QEvent.MouseMove and self.start_point is not None:
                self.end_point = self.mapFromGlobal(_event_global_point(event))
                self.points.append(self.end_point)
                self._classify_axis()
                self.update()
                return True
            if (
                event_type == QtCore.QEvent.MouseButtonRelease
                and self.start_point is not None
                and event.button() == QtCore.Qt.LeftButton
            ):
                self.end_point = self.mapFromGlobal(_event_global_point(event))
                self.points.append(self.end_point)
                self._classify_axis(force=True)
                self._apply_alignment()
                return True
            return False

        def _classify_axis(self, force=False):
            if self.axis is not None:
                return
            dx = self.end_point.x() - self.start_point.x()
            dy = self.end_point.y() - self.start_point.y()
            if force or dx * dx + dy * dy >= 100:
                self.axis = alignment_axis((0, 0), (dx, dy))

        def _apply_alignment(self):
            graph_points = tuple(
                _screen_to_graph(point, self, nuke) for point in self.points
            )
            crossed = nodes_intersected_by_stroke(
                graph_points, candidates, dot_padding=10.0
            )
            if len(crossed) < 2:
                self._reset_gesture()
                self._flash_invalid_cursor()
                return
            aligned_nodes = include_connected_between(crossed, candidates)
            aligned_nodes = include_connected_dots(
                aligned_nodes,
                candidates,
                reference_nodes=crossed,
                padding=30.0,
            )
            coordinates = aligned_coordinates(aligned_nodes, self.axis)
            coordinates = adjust_to_external_inputs(
                aligned_nodes, coordinates, candidates, self.axis
            )
            coordinates = resolve_alignment_overlaps(
                aligned_nodes, coordinates, self.axis
            )
            collided_obstacles = colliding_external_obstacles(
                aligned_nodes, coordinates, candidates
            )
            coordinates = offset_around_external_obstacles(
                aligned_nodes, coordinates, candidates, self.axis
            )
            nuke.Undo.begin("Nuke OCD: Draw Align Nodes")
            try:
                for node, (x, y) in zip(aligned_nodes, coordinates):
                    node.setXYpos(x, y)
            except Exception:
                nuke.Undo.cancel()
                self.finish()
                raise
            else:
                nuke.Undo.end()
                self._show_collision_markers(collided_obstacles)
                self._reset_gesture()

        def _show_collision_markers(self, obstacles):
            """Flash a small comic starburst at each blocking node's center."""

            class CollisionMarker(QtWidgets.QWidget):
                def paintEvent(self, event):
                    import math

                    painter = QtGui.QPainter(self)
                    painter.setRenderHint(QtGui.QPainter.Antialiasing)
                    points = []
                    for index in range(24):
                        angle = -math.pi / 2.0 + index * math.pi / 12.0
                        radius = 15.0 if index % 2 == 0 else 9.0
                        points.append(QtCore.QPointF(
                            17.0 + math.cos(angle) * radius,
                            17.0 + math.sin(angle) * radius,
                        ))
                    painter.setPen(QtGui.QPen(QtGui.QColor(190, 55, 20), 2))
                    painter.setBrush(QtGui.QColor(255, 210, 45, 250))
                    painter.drawPolygon(QtGui.QPolygonF(points))
                    painter.setPen(QtGui.QColor(70, 25, 10))
                    font = painter.font()
                    font.setBold(True)
                    font.setPixelSize(15)
                    painter.setFont(font)
                    painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "!")

            for obstacle in obstacles:
                try:
                    zoom = float(nuke.zoom())
                    center = nuke.center()
                    graph_x = obstacle.xpos() + obstacle.screenWidth() / 2.0
                    # In Nuke's live DAG projection the marker's top-level Qt
                    # window lands half a node-height above the graph-space Y
                    # requested here. Include that offset so the visible icon,
                    # rather than its window anchor, is centered on the node.
                    graph_y = obstacle.ypos() + obstacle.screenHeight()
                    local_x = self.width() / 2.0 + (graph_x - center[0]) * zoom
                    local_y = self.height() / 2.0 + (graph_y - center[1]) * zoom
                    global_point = self.mapToGlobal(
                        QtCore.QPoint(int(round(local_x)), int(round(local_y)))
                    )
                except RuntimeError:
                    continue

                marker = CollisionMarker(parent=None)
                marker.setWindowFlags(
                    QtCore.Qt.Tool
                    | QtCore.Qt.FramelessWindowHint
                    | QtCore.Qt.WindowStaysOnTopHint
                )
                marker.setAttribute(QtCore.Qt.WA_TranslucentBackground)
                marker.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
                marker.setFixedSize(34, 34)
                marker.move(global_point.x() - 17, global_point.y() - 17)

                marker.show()
                marker.raise_()
                self._collision_markers.append(marker)

                def remove_marker(widget=marker):
                    try:
                        widget.close()
                    except RuntimeError:
                        pass
                    try:
                        self._collision_markers.remove(widget)
                    except (ValueError, RuntimeError):
                        pass

                QtCore.QTimer.singleShot(1000, remove_marker)

        def _reset_gesture(self):
            try:
                self.releaseMouse()
            except RuntimeError:
                pass
            self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
            self.start_point = None
            self.end_point = None
            self.points = []
            self.axis = None
            self.update()
            self.hide()

        def _flash_invalid_cursor(self):
            self.invalid_cursor_active = True
            app.changeOverrideCursor(self.invalid_cursor)

            def restore_cursor():
                try:
                    self.invalid_cursor_active = False
                    if self.active:
                        self._apply_cursor_override(self.ocd_cursor)
                except RuntimeError:
                    pass

            QtCore.QTimer.singleShot(1000, restore_cursor)

        def _schedule_cursor_refresh(self):
            if self.cursor_refresh_pending or self.invalid_cursor_active:
                return
            self.cursor_refresh_pending = True

            def refresh():
                try:
                    if self.active and not self.invalid_cursor_active:
                        self._apply_cursor_override(self.ocd_cursor)
                finally:
                    # Keep the guard raised while changeOverrideCursor emits
                    # its own synchronous CursorChange event.
                    self.cursor_refresh_pending = False

            QtCore.QTimer.singleShot(0, refresh)

        def _apply_cursor_override(self, cursor):
            if app.overrideCursor() is None:
                app.setOverrideCursor(cursor)
            else:
                app.changeOverrideCursor(cursor)

        def finish(self):
            global _ACTIVE_ALIGN_CAPTURE
            self.active = False
            app.removeEventFilter(self)
            try:
                self.releaseMouse()
            except RuntimeError:
                pass
            app.restoreOverrideCursor()
            super().close()
            if _ACTIVE_ALIGN_CAPTURE is self:
                _ACTIVE_ALIGN_CAPTURE = None

        def show(self):
            # Stay fully hidden while armed. A translucent top-level widget can
            # still block Nuke's OpenGL viewport on macOS even when Qt marks it
            # mouse-transparent, so it exists visibly only during a stroke.
            self.hide()
            self._apply_cursor_override(self.ocd_cursor)
            app.installEventFilter(self)

    capture = AlignCapture()
    _ACTIVE_ALIGN_CAPTURE = capture
    capture.show()
    return capture
