"""Predicates for geom2drobotenvs/ThreeTables-v0."""

from typing import Dict, Iterator, Optional, Sequence

import numpy as np
from geom2drobotenvs.concepts import is_inside, is_movable_rectangle
from geom2drobotenvs.structs import MultiBody2D, ZOrder
from geom2drobotenvs.utils import (
    get_suctioned_objects,
    object_to_multibody2d,
    rectangle_object_to_geom,
    z_orders_may_collide,
)
from gym.spaces import Space
from relational_structs import (
    Object,
    ObjectCentricStateSpace,
    Predicate,
    State,
)
from tomsgeoms2d.structs import LineSegment, Rectangle
from tomsgeoms2d.utils import geom2ds_intersect

from manual_relational_models.geom2drobotenvs.three_tables.helpers import (
    get_objects_on_object,
    get_shelf_empty_side_center,
)


def iter_predicates(observation_space: Space) -> Iterator[Predicate]:
    """Create manual predicates."""

    # NOTE: the static object caches are currently not shared, out of an
    # abundance of caution. Future optimizations should consider carefully
    # using a shared static object cache, but beware of objects with the same
    # names in different tasks.

    # Extract the necessary types.
    assert isinstance(observation_space, ObjectCentricStateSpace)
    type_name_to_type = {t.name: t for t in observation_space.types}
    CRVRobotType = type_name_to_type["crv_robot"]
    RectangleType = type_name_to_type["rectangle"]

    # IsBlock.
    def _is_block_holds(state: State, objs: Sequence[Object]) -> bool:
        (block,) = objs
        return is_movable_rectangle(state, block)

    yield Predicate("IsBlock", [RectangleType], _is_block_holds)

    # IsShelf.
    def _is_shelf_holds(state: State, objs: Sequence[Object]) -> bool:
        # This is very naive -- doesn't even check the pose or shape of the
        # static objects, just assumes 3 static objects == is shelf.
        (shelf,) = objs
        static_object_cache: Dict[Object, MultiBody2D] = {}  # see note above
        objs_on_top = get_objects_on_object(state, shelf, static_object_cache)
        static_objs = {o for o in objs_on_top if state.get(o, "static") > 0.5}
        return len(static_objs) == 3

    yield Predicate("IsShelf", [RectangleType], _is_shelf_holds)

    # IsBlockOrBackWall.
    def _is_back_wall_holds(state: State, objs: Sequence[Object]) -> bool:
        # The angle between the center of the wall and the empty side is near
        # a multiple of pi / 2.
        (wall,) = objs
        if state.get(wall, "static") <= 0.5:
            return False
        # Identify the shelf.
        shelf: Optional[Object] = None
        for obj in state.get_objects(RectangleType):
            if not _is_shelf_holds(state, [obj]):
                continue
            if _on_shelf_holds(state, [wall, obj]):
                shelf = obj
                break
        if shelf is None:
            return False
        static_object_cache: Dict[Object, MultiBody2D] = {}  # see note above
        empty_x, empty_y = get_shelf_empty_side_center(
            state, shelf, static_object_cache
        )
        rect = rectangle_object_to_geom(state, wall, static_object_cache)
        rect_cx, rect_cy = rect.center
        angle = np.arctan2(empty_y - rect_cy, empty_x - rect_cx)
        is_back_wall = abs(angle % (np.pi / 2)) < 1e-3
        return is_back_wall

    def _is_block_or_back_wall(state: State, objs: Sequence[Object]) -> bool:
        return _is_block_holds(state, objs) or _is_back_wall_holds(state, objs)

    yield Predicate("IsBlockOrBackWall", [RectangleType], _is_block_or_back_wall)

    # Smaller.
    def _smaller_holds(state: State, objs: Sequence[Object]) -> bool:
        obj1, obj2 = objs
        smaller_area = state.get(obj1, "width") * state.get(obj1, "height")
        larger_area = state.get(obj2, "width") * state.get(obj2, "height")
        return smaller_area < larger_area

    yield Predicate("Smaller", [RectangleType, RectangleType], _smaller_holds)

    # OnShelf.
    def _on_shelf_holds(state: State, objs: Sequence[Object]) -> bool:
        target, shelf = objs
        if not _is_shelf_holds(state, [shelf]):
            return False
        static_object_cache: Dict[Object, MultiBody2D] = {}  # see note above
        return is_inside(state, target, shelf, static_object_cache)

    yield Predicate("OnShelf", [RectangleType, RectangleType], _on_shelf_holds)

    # InFrontOnShelf.
    def _in_front_on_shelf(state: State, objs: Sequence[Object]) -> bool:
        # First draw a line from the behind object to the empty side of the
        # shelf and check if the front object intersects that line. If so,
        # draw a line from the behind object to the front object and check if
        # any other objects intersect that line.
        front_obj, behind_obj, shelf = objs
        if front_obj == behind_obj:
            return False
        if not _is_shelf_holds(state, [shelf]):
            return False
        if not _on_shelf_holds(state, [front_obj, shelf]):
            return False
        if not _on_shelf_holds(state, [behind_obj, shelf]):
            return False
        static_object_cache: Dict[Object, MultiBody2D] = {}  # see note above
        empty_x, empty_y = get_shelf_empty_side_center(
            state, shelf, static_object_cache
        )
        front_rect = rectangle_object_to_geom(state, front_obj, static_object_cache)
        behind_rect = rectangle_object_to_geom(state, behind_obj, static_object_cache)
        behind_to_empty = LineSegment(
            behind_rect.center[0], behind_rect.center[1], empty_x, empty_y
        )
        if not geom2ds_intersect(front_rect, behind_to_empty):
            return False
        behind_to_front = LineSegment(
            behind_rect.center[0],
            behind_rect.center[1],
            front_rect.center[0],
            front_rect.center[1],
        )
        for other_obj in state.get_objects(RectangleType):
            if other_obj in {front_obj, behind_obj}:
                continue
            if not _is_block_holds(state, [other_obj]):
                continue
            other_rect = rectangle_object_to_geom(state, other_obj, static_object_cache)
            if geom2ds_intersect(other_rect, behind_to_front):
                return False
        return True

    yield Predicate(
        "InFrontOnShelf",
        [RectangleType, RectangleType, RectangleType],
        _in_front_on_shelf,
    )

    # ClearToPick.
    def _clear_to_pick_holds(state: State, objs: Sequence[Object]) -> bool:
        if not _on_shelf_holds(state, objs):
            return False
        target, shelf = objs
        if not _is_block_or_back_wall(state, [target]):
            return False
        # Draw a line from the target to the empty side of the shelf. If that
        # line doesn't intersect anything, it's clear to pick.
        static_object_cache: Dict[Object, MultiBody2D] = {}  # see note above
        shelf_mb = object_to_multibody2d(shelf, state, static_object_cache)
        assert len(shelf_mb.bodies) == 1
        shelf_rect = shelf_mb.bodies[0].geom
        assert isinstance(shelf_rect, Rectangle)
        target_mb = object_to_multibody2d(target, state, static_object_cache)
        assert len(target_mb.bodies) == 1
        target_rect = target_mb.bodies[0].geom
        assert isinstance(target_rect, Rectangle)
        target_x, target_y = target_rect.center
        target_z_order = ZOrder(int(state.get(target, "z_order")))
        obstacles = set(state) - {target, shelf}
        side_x, side_y = get_shelf_empty_side_center(state, shelf, static_object_cache)
        line_geom = LineSegment(target_x, target_y, side_x, side_y)
        for obstacle in obstacles:
            obstacle_multibody = object_to_multibody2d(
                obstacle, state, static_object_cache
            )
            for obstacle_body in obstacle_multibody.bodies:
                if not z_orders_may_collide(target_z_order, obstacle_body.z_order):
                    continue
                if geom2ds_intersect(line_geom, obstacle_body.geom):
                    return False
        return True

    yield Predicate("ClearToPick", [RectangleType, RectangleType], _clear_to_pick_holds)

    # HandEmpty.
    def _hand_empty_holds(state: State, objs: Sequence[Object]) -> bool:
        (robot,) = objs
        return not get_suctioned_objects(state, robot)

    yield Predicate("HandEmpty", [CRVRobotType], _hand_empty_holds)

    # Holding.
    def _holding_holds(state: State, objs: Sequence[Object]) -> bool:
        obj, robot = objs
        held_objs = {o for o, _ in get_suctioned_objects(state, robot)}
        return obj in held_objs

    yield Predicate("Holding", [RectangleType, CRVRobotType], _holding_holds)
