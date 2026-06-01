# Represents one probabilistic destination branch of an edge.
class EdgeDestination(object):
    # Memory-optimized fixed attribute layout for destination objects.
    __slots__ = ["_dest", "_prob", "_assgns", "_reward"]

    def __init__(self, dest, prob, assgns, reward=None):
        # Destination locations reached by this branch (immutable set for hash safety).
        self._dest = frozenset(dest)
        # Probability expression/value of taking this destination branch.
        self._prob = prob
        # Assignment map applied when this destination is taken: variable_name -> expression.
        self._assgns = assgns
        # Optional reward expression/value attached to this transition branch.
        self._reward = reward

    @property
    def destination(self):
        # Return destination-location set.
        return self._dest

    @property
    def probability(self):
        # Return branch probability expression/value.
        return self._prob

    @property
    def assignments(self):
        # Return variable assignments executed on transition firing.
        return self._assgns

    @property
    def reward(self):
        # Reward is optional; raise explicit error when absent.
        if self._reward is None:
            raise AttributeError("Edges destination has no reward attribute")
        return self._reward


# Represents one automaton transition rule from source locations with guarded behavior.
class Edge(object):
    # Memory-optimized fixed attribute layout for edge objects.
    __slots__ = ["_src", "_action", "_guard", "_edgeDests"]

    def __init__(self, src, action, guard, edgeDests):
        # Source-location set required for this edge to be location-enabled.
        self._src = frozenset(src)
        # Action label associated with this edge.
        self._action = action
        # Guard expression that must evaluate to True for the edge to fire.
        self._guard = guard
        # List of probabilistic destination branches (EdgeDestination objects).
        self._edgeDests = edgeDests

    @property
    def source(self):
        # Return required source-location set.
        return self._src

    @property
    def action(self):
        # Return action label.
        return self._action

    @property
    def guard(self):
        # Return guard expression.
        return self._guard

    @property
    def edgeDestinations(self):
        # Return list of possible destination branches.
        return self._edgeDests

    def isSatisfied(self, setOfLoc, varGetter, funcGetter):
        """Return True if guard is satisfied, otherwise return False."""
        # Edge is enabled iff:
        # 1) current active locations contain all edge source locations, and
        # 2) guard evaluates to True in current variable/function environment.
        return setOfLoc.issuperset(self._src) and self._guard.eval(varGetter, funcGetter)


# Represents one automaton (locations + initial locations + transition edges).
class Automata(object):
    # Memory-optimized fixed attribute layout for automaton objects.
    __slots__ = ["_name", "_locs", "_initLoc", "_edges"]

    def __init__(self, name, locs, initLoc, edges):
        # Automaton identifier.
        self._name = name
        # Nested location map: location_name -> {transient_variable_name -> expression}.
        self._locs = locs
        # Initial active location set of this automaton.
        self._initLoc = initLoc
        # Transition edge list defining automaton dynamics.
        self._edges = edges

    @property
    def name(self):
        # Return automaton name.
        return self._name

    @property
    def locations(self):
        # Return location metadata/map.
        return self._locs

    @property
    def initLocation(self):
        # Return initial location set.
        return self._initLoc

    @property
    def edges(self):
        # Return transition edge list.
        return self._edges
