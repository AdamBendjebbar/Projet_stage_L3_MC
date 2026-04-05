from copy import deepcopy

class State(object):
    __slots__ = ["_transientVars", "_nonTransientVars", "_setOfLocation", "_hashRepr"]

    def __init__(self, transientVars, nonTransientVars, setOfLocation):
        self._transientVars = transientVars
        self._nonTransientVars = nonTransientVars
        self._setOfLocation = frozenset(setOfLocation)
        self._hashRepr = (frozenset(self._nonTransientVars.values()), self._setOfLocation)

    @property
    def setOfLocation(self):
        return self._setOfLocation

    def clone(self, src, dest, assgns, locs=None, funcGetter=None):
        """Clone a new state based on the given arguments."""
        transientVars = deepcopy(self._transientVars)
        nonTransientVars = deepcopy(self._nonTransientVars)

        for var in transientVars.values():
            var.resetToInitValue()
        
        for ref, value in assgns.items():
            if ref not in nonTransientVars:
                transientVars[ref].setValueTo(value.eval(self, funcGetter))
            else:
                nonTransientVars[ref].setValueTo(value.eval(self, funcGetter))
        setOfLocation = self._setOfLocation.difference(src).union(dest)

        varGetter = { var.name: var.value for var in nonTransientVars.values() }
        for loc in locs:
            if loc not in locs:
                continue
            for ref, value in locs[loc].items():
                transientVars[ref].setValueTo(value.eval(varGetter, funcGetter))
        return State(transientVars, nonTransientVars, setOfLocation)

    def get(self, name, default=None):
        """Return the value of the associated variable (both transient or non-transient),
        if it existsn otherwise return the default value."""
        var = self._transientVars.get(name)
        if var is not None:
            return var.value
        var = self._nonTransientVars.get(name)
        if var is not None:
            return var.value
        return default

    # def getLowMemRepr(self):
    #     """Return a string representation of the state (low memory representation)."""
    #     nonTransientVars, setOfLocation = self._hashRepr
    #     return str((sorted(nonTransientVars, key=lambda v: v.name), sorted(setOfLocation)))

    def getTupRepr(self, stateTemplate):
        """Return a tuple representation (an immutable list) representation of the state respecting stateTemplate."""
        repr = [0] * len(stateTemplate)
        for var in self._nonTransientVars.values():
            repr[stateTemplate[var.name]] = var.value
        if len(self._setOfLocation) > 1:
            for loc in self._setOfLocation:
                repr[stateTemplate[loc]] = 1
        return tuple(repr)

    def __contains__(self, item):
        return item in self._transientVars or item in self._nonTransientVars

    def __eq__(self, value):
        return isinstance(value, State) and self._hashRepr == value._hashRepr

    def __hash__(self):
        return hash(self._hashRepr)
