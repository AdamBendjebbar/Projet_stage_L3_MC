class EdgeDestination(object):
    __slots__ = ["_dest", "_prob", "_assgns", "_reward"]

    def __init__(self, dest, prob, assgns, reward=None):
        self._dest = frozenset(dest)
        self._prob = prob
        self._assgns = assgns
        self._reward = reward

    @property
    def destination(self):
        return self._dest

    @property
    def probability(self):
        return self._prob

    @property
    def assignments(self):
        return self._assgns

    @property
    def reward(self):
        if self._reward is None:
            raise AttributeError("Edges destination has no reward attribute")
        return self._reward


class Edge(object):
    __slots__ = ["_src", "_action", "_guard", "_edgeDests"]

    def __init__(self, src, action, guard, edgeDests):
        self._src = frozenset(src)
        self._action = action
        self._guard = guard
        self._edgeDests = edgeDests

    @property
    def source(self):
        return self._src

    @property
    def action(self):
        return self._action

    @property
    def guard(self):
        return self._guard

    @property
    def edgeDestinations(self):
        return self._edgeDests

    def isSatisfied(self, setOfLoc, varGetter, funcGetter):
        """Return True if guard is satisfied, otherwise return False."""
        return setOfLoc.issuperset(self._src) and self._guard.eval(varGetter, funcGetter)


class Automata(object):
    __slots__ = ["_name", "_locs", "_initLoc", "_edges"]

    def __init__(self, name, locs, initLoc, edges):
        self._name = name
        self._locs = locs
        self._initLoc = initLoc
        self._edges = edges

    @property
    def name(self):
        return self._name

    @property
    def locations(self):
        return self._locs

    @property
    def initLocation(self):
        return self._initLoc

    @property
    def edges(self):
        return self._edges
