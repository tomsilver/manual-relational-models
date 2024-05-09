"""Temporary script to verify non-breaking changes during refactor."""

from typing import Dict

# Needed to register environments for gym.make().
import geom2drobotenvs  # pylint: disable=unused-import
import gym
from geom2drobotenvs.concepts import is_movable_rectangle
from geom2drobotenvs.object_types import CRVRobotType, Geom2DType, RectangleType
from geom2drobotenvs.structs import ZOrder
from gym.wrappers.record_video import RecordVideo
from relational_structs import (
    GroundOperator,
    Option,
    ParameterizedOption,
    PDDLDomain,
    PDDLProblem,
    State,
)
from relational_structs.utils import abstract, parse_pddl_plan
from tomsutils.pddl_planning import run_pddl_planner

from manual_relational_models.geom2drobotenvs.three_tables.operators import (
    iter_operators,
)
from manual_relational_models.geom2drobotenvs.three_tables.options import iter_options
from manual_relational_models.geom2drobotenvs.three_tables.predicates import (
    iter_predicates,
)


def _ground_op_to_option(
    ground_op: GroundOperator, option_name_to_option: Dict[str, ParameterizedOption]
) -> Option:
    if ground_op.name == "Pick":
        param_option = option_name_to_option["RectangleVacuumPick"]
        robot, target, _, _ = ground_op.parameters
        return param_option.ground([robot, target])

    if ground_op.name == "Place":
        param_option = option_name_to_option["RectangleVacuumPlace"]
        robot, target, _, shelf = ground_op.parameters
        return param_option.ground([robot, target, shelf])

    raise NotImplementedError


def _main() -> None:
    env = gym.make("geom2drobotenvs/ThreeTables-v0")
    env = RecordVideo(env, "videos")
    seed = 0
    obs, _ = env.reset(seed=seed)
    assert isinstance(obs, State)
    env.action_space.seed(seed)

    # Get the relevant objects.
    blocks = [o for o in obs if is_movable_rectangle(obs, o)]
    blocks = sorted(blocks, key=lambda b: obs.get(b, "width") * obs.get(b, "height"))
    shelfs = [
        o
        for o in obs
        if o.is_instance(RectangleType)
        and obs.get(o, "static") > 0.5
        and int(obs.get(o, "z_order")) == ZOrder.FLOOR.value
    ]
    bx = obs.get(blocks[0], "x")
    by = obs.get(blocks[0], "y")
    dist = lambda t: (bx - obs.get(t, "x")) ** 2 + (by - obs.get(t, "y")) ** 2
    shelf = max(shelfs, key=dist)

    # Construct a PDDL domain and problem.
    types = {Geom2DType, CRVRobotType, RectangleType}
    predicates = set(iter_predicates(env.observation_space))
    operators = set(iter_operators(predicates, types))
    domain = PDDLDomain("three-shelfs", operators, predicates, types)

    objects = set(obs)
    init_atoms = abstract(obs, predicates)
    pred_name_to_pred = {p.name: p for p in predicates}
    OnShelf = pred_name_to_pred["OnShelf"]
    goal = {OnShelf([block, shelf]) for block in blocks}
    problem = PDDLProblem(domain.name, "problem0", objects, init_atoms, goal)

    print("Running planning...")
    ground_op_strs = run_pddl_planner(str(domain), str(problem), planner="fd-opt")
    ground_op_plan = parse_pddl_plan(ground_op_strs, domain, problem)
    print("Found plan!")
    for op in ground_op_plan:
        print(op.short_str)
    print()
    option_name_to_option = {
        o.name: o for o in iter_options(env.observation_space, env.action_space)
    }
    option_plan = [
        _ground_op_to_option(o, option_name_to_option) for o in ground_op_plan
    ]

    for option in option_plan:
        print("Starting option", option)
        assert option.initiable(obs)
        for _ in range(100):
            action = option.policy(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            assert not terminated or truncated
            if option.terminal(obs):
                break
        else:
            assert False, "Option did not terminate"

    env.close()


if __name__ == "__main__":
    _main()
