"""Operators for geom2drobotenvs/ThreeTables-v0."""

from typing import Iterator, Set

from relational_structs import (
    LiftedOperator,
    Predicate,
)


def iter_operators(
    predicates: Set[Predicate], types: Set[Predicate]
) -> Iterator[LiftedOperator]:
    """Create manual operators."""

    # Extract predicates.
    pred_name_to_pred = {p.name: p for p in predicates}
    Smaller = pred_name_to_pred["Smaller"]
    IsBlock = pred_name_to_pred["IsBlock"]
    IsShelf = pred_name_to_pred["IsShelf"]
    IsBlockOrBackWall = pred_name_to_pred["IsBlockOrBackWall"]
    OnShelf = pred_name_to_pred["OnShelf"]
    HandEmpty = pred_name_to_pred["HandEmpty"]
    ClearToPick = pred_name_to_pred["ClearToPick"]
    Holding = pred_name_to_pred["Holding"]
    InFrontOnShelf = pred_name_to_pred["InFrontOnShelf"]

    # Extract types.
    type_name_to_type = {t.name: t for t in types}
    CRVRobotType = type_name_to_type["crv_robot"]
    RectangleType = type_name_to_type["rectangle"]

    # Pick.
    robot = CRVRobotType("?robot")
    target = RectangleType("?target")
    behind = RectangleType("?behind")
    shelf = RectangleType("?shelf")
    preconditions = {
        IsShelf([shelf]),
        IsBlock([target]),
        IsBlockOrBackWall([behind]),
        OnShelf([target, shelf]),
        ClearToPick([target, shelf]),
        HandEmpty([robot]),
        InFrontOnShelf([target, behind, shelf]),
    }
    add_effects = {
        Holding([target, robot]),
        ClearToPick([behind, shelf]),
    }
    delete_effects = {
        OnShelf([target, shelf]),
        ClearToPick([target, shelf]),
        HandEmpty([robot]),
        InFrontOnShelf([target, behind, shelf]),
    }
    yield LiftedOperator(
        "Pick",
        [robot, target, behind, shelf],
        preconditions,
        add_effects,
        delete_effects,
    )

    # Place.
    robot = CRVRobotType("?robot")
    held = RectangleType("?held")
    behind = RectangleType("?behind")
    shelf = RectangleType("?shelf")
    preconditions = {
        IsBlock([held]),
        IsShelf([shelf]),
        Holding([held, robot]),
        IsBlockOrBackWall([behind]),
        ClearToPick([behind, shelf]),
        Smaller([held, behind]),  # NOTE
    }
    add_effects = {
        OnShelf([held, shelf]),
        ClearToPick([held, shelf]),
        HandEmpty([robot]),
        InFrontOnShelf([held, behind, shelf]),
    }
    delete_effects = {
        Holding([held, robot]),
        ClearToPick([behind, shelf]),
    }
    yield LiftedOperator(
        "Place",
        [robot, held, behind, shelf],
        preconditions,
        add_effects,
        delete_effects,
    )
