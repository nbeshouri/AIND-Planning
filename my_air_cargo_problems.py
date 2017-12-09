from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr, Expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache
from typing import List


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()
        # Cache actions that have a direct impact on goals. This is used
        # to make h_ignore_preconditions faster to calculate.
        self._goal_actions = [a for a in self.actions_list if
                              any(e in self.goal for e in a.effect_add)]
        # Cache a map from fluents to their in index in the state map.
        # This is ued in the result and ture_in_state methods.
        self._fluent_to_map_index = {state: index for index, state
                                     in enumerate(self.state_map)}

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            loads = []
            # TODO create all load ground actions from the domain Load action
            for cargo in self.cargos:
                for plane in self.planes:
                    for airport in self.airports:
                        preconds_pos = [expr(f'At({cargo}, {airport})'), expr(f'At({plane}, {airport})')]
                        preconds_neg = []
                        effect_add = [expr(f'In({cargo}, {plane})')]
                        effect_rem = [expr(f'At({cargo}, {airport})')]
                        loads.append(Action(expr(f'Load({cargo}, {plane}, {airport})'),
                                            [preconds_pos, preconds_neg],
                                            [effect_add, effect_rem]))
            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            unloads = []
            # TTODO create all Unload ground actions from the domain Unload action
            for cargo in self.cargos:
                for plane in self.planes:
                    for airport in self.airports:
                        preconds_pos = [expr(f'In({cargo}, {plane})'),
                                                  expr(f'At({plane}, {airport})')]
                        preconds_neg = []
                        effect_add = [expr(f'At({cargo}, {airport})')]
                        effect_rem = [expr(f'In({cargo}, {plane})')]
                        unloads.append(Action(expr(f'Unload({cargo}, {plane}, {airport})'),
                                              [preconds_pos, preconds_neg],
                                              [effect_add, effect_rem]))
            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # TODO implement <---- this where you start.
        possible_actions = []
        for action in self.actions_list:
            is_possible = True
            for clause in action.precond_pos:
                if not self.true_in_state(clause, state):
                    is_possible = False
            for clause in action.precond_neg:
                if self.true_in_state(clause, state):
                    is_possible = False
            if is_possible:
                possible_actions.append(action)
        return possible_actions

    def result(self, state: str, action: Action) -> str:
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        # TODO implement
        # Convert the state into a list of chars.
        new_state = list(state)
        # Set all fluents in the actions's remove list to false.
        for f in action.effect_rem:
            new_state[self._fluent_to_map_index[f]] = 'F'
        # Set all fluents in the actions's add list to true.
        for f in action.effect_add:
            new_state[self._fluent_to_map_index[f]] = 'T'
        # All other fluents are assumed not to have changed (because PDDL
        # requires them not to.)
        return ''.join(new_state)

    def goal_test(self, state: str) -> bool:
        """Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        for clause in self.goal:
            if not self.true_in_state(clause, state):
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        uncovered_goals = set(g for g in self.goal
                              if not self.true_in_state(g, node.state))
       
        def pick_action():
            """Returns that action that covers the most uncovered goals."""
            def rank_action(action):
                return len(uncovered_goals.intersection(action.effect_add))
            return max(self._goal_actions, key=rank_action)

        # Pick actions in a greedy way until all goals are covered
        # and then return the count.
        count = 0
        while uncovered_goals:
            count += 1
            a = pick_action()
            uncovered_goals.difference_update(a.effect_add)
        return count

    @lru_cache(maxsize=8192)
    def h_unsatisified_goals(self, node: Node) -> int:
        """Return the number of unsatisfied goals in a node's state.

        This is a simpler version of the ignore preconditions heuristic
        which assumes that no action can satisfy more than one goal and
        no action undoes the action of another.
        """
        count = 0
        for clause in self.goal:
            if not self.true_in_state(clause, node.state):
                count += 1
        return count

    def true_in_state(self, literal: Expr, state: str) -> bool:
        """Check whether or not a literal is true in a given state."""
        return state[self._fluent_to_map_index[literal]] == 'T'


def air_cargo_p0():
    """Return a simple problem for debugging."""
    planes = 'P1'.split()
    cargos = 'C1'.split()
    airports = 'JFK SFO'.split()
    init = 'At(C1, SFO) ∧ At(P1, SFO)'.split(' ∧ ')
    goal = 'At(C1, JFK)'.split(' ∧ ')
    return problem_helper(cargos, planes, airports, init, goal)


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    planes = 'P1 P2 P3'.split()
    cargos = 'C1 C2 C3'.split()
    airports = 'JFK SFO ATL'.split()
    init = ('At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) ∧ At(P1, SFO) ∧ '
            'At(P2, JFK) ∧ At(P3, ATL)'.split(' ∧ '))
    goal = 'At(C1, JFK) ∧ At(C2, SFO) ∧ At(C3, SFO)'.split(' ∧ ')
    return problem_helper(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    planes = 'P1 P2'.split()
    cargos = 'C1 C2 C3 C4'.split()
    airports = 'JFK SFO ATL ORD'.split()
    init = ('At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) ∧ At(C4, ORD) '
            '∧ At(P1, SFO) ∧ At(P2, JFK)'.split(' ∧ '))
    goal = 'At(C1, JFK) ∧ At(C3, JFK) ∧ At(C2, SFO) ∧ At(C4, SFO)'.split(' ∧ ')
    return problem_helper(cargos, planes, airports, init, goal)


def problem_helper(cargos: List[str], planes: List[str], airports: List[str],
                   init: List[str], goal: List[str]) -> AirCargoProblem:
    """
    Create an AirCargoProblem out of lists of literals as strings.

    This is a helper function that makes it easier to create problems.
    It takes literals as lists of strings rather than Expr objects and
    automatically supplies the negative literals in the initial state.

    :param cargos: list all cargo objects as strings
    :param planes: list of plane objects as strings
    :param airports: list of airport objects as strings
    :param init: list of positive literals in the initial state as strings
    :param goal: list of true literals goal state
    :return: AirCargoProblem constructed from arguments
    """
    # Convert the strings in the init list to Expr objects. These will
    # be used as the positive literals in the initial state.
    pos = [expr(p) for p in init]
    # To find the negative literals, first construct a list of all possible
    # literals.
    neg = []
    for plane in planes:
        for airport in airports:
            neg.append(expr(f'At({plane}, {airport})'))
    for cargo in cargos:
        for plane in planes:
            neg.append(expr(f'In({cargo}, {plane})'))
        for airport in airports:
            neg.append(expr(f'At({cargo}, {airport})'))
    # Then remeove all literals in the positive literal list.
    neg = [n for n in neg if n not in pos]
    init = FluentState(pos, neg)
    goal = [expr(g) for g in goal]
    return AirCargoProblem(cargos, planes, airports, init, goal)
