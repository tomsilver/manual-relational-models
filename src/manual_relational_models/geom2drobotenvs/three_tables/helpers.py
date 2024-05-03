"""Helper functions for geom2drobotenvs/ThreeTables-v0 models."""

from typing import Dict, Set, Tuple

from geom2drobotenvs.object_types import RectangleType
from geom2drobotenvs.structs import MultiBody2D
from geom2drobotenvs.utils import (
    rectangle_object_to_geom,
)
from relational_structs import (
    Object,
    State,
)


def get_objects_on_object(
    state: State, obj: Object, static_object_cache: Dict[Object, MultiBody2D]
) -> Set[Object]:
    """Find all objects that are on the given object in the given state."""
    ret_objs: Set[Object] = set()
    surface = rectangle_object_to_geom(state, obj, static_object_cache)
    for other_obj in state.get_objects(RectangleType):
        if other_obj == obj:
            continue
        rect = rectangle_object_to_geom(state, other_obj, static_object_cache)
        x, y = rect.center
        if surface.contains_point(x, y):
            ret_objs.add(other_obj)
    return ret_objs


def get_shelf_empty_side_center(
    state: State, shelf: Object, static_object_cache: Dict[Object, MultiBody2D]
) -> Tuple[float, float]:
    """Find the position of the empty side of the given shelf."""
    objs_on_top = get_objects_on_object(state, shelf, static_object_cache)
    walls = {o for o in objs_on_top if state.get(o, "static") > 0.5}
    assert len(walls) == 3
    wall_rects = {
        rectangle_object_to_geom(state, w, static_object_cache) for w in walls
    }
    example_wall_rect = next(iter(wall_rects))
    wall_thickness = min(example_wall_rect.height, example_wall_rect.width)
    pad = wall_thickness / 2
    shelf_rect = rectangle_object_to_geom(state, shelf, static_object_cache)
    height_scale = (shelf_rect.height - pad) / shelf_rect.height
    width_scale = (shelf_rect.width - pad) / shelf_rect.width
    inner_shelf_rect = shelf_rect.scale_about_center(
        width_scale=width_scale, height_scale=height_scale
    )
    for v1, v2 in zip(
        inner_shelf_rect.vertices,
        inner_shelf_rect.vertices[1:] + [inner_shelf_rect.vertices[0]],
    ):
        cx = (v1[0] + v2[0]) / 2
        cy = (v1[1] + v2[1]) / 2
        contained_in_wall = False
        for wall_rect in wall_rects:
            if wall_rect.contains_point(cx, cy):
                contained_in_wall = True
                break

        if not contained_in_wall:
            return (cx, cy)
    raise ValueError("There is no empty side on the shelf.")
