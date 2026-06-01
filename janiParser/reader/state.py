from copy import deepcopy

class State(object):
    # Memory-optimized fixed attribute layout for State instances.
    __slots__ = ["_transientVars", "_nonTransientVars", "_setOfLocation", "_hashRepr"]

    def __init__(self, transientVars, nonTransientVars, setOfLocation):
        # Mapping of transient variable name -> variable object.
        self._transientVars = transientVars
        # Mapping of non-transient variable name -> variable object.
        self._nonTransientVars = nonTransientVars
        # Immutable set of active locations in this state.
        self._setOfLocation = frozenset(setOfLocation)
        # Canonical immutable representation used for equality/hash operations.
        self._hashRepr = (frozenset(self._nonTransientVars.values()), self._setOfLocation)

    @property
    def setOfLocation(self):
        # Expose active locations as read-only state information.
        return self._setOfLocation

    def clone(self, src, dest, assgns, locs=None, funcGetter=None):
        """Clone a new state based on the given arguments."""
        # Deep-copy variables to build an independent successor state.
        transientVars = deepcopy(self._transientVars)
        nonTransientVars = deepcopy(self._nonTransientVars)

        # Reset all transient variables to their initial values before applying updates.
        for var in transientVars.values():
            var.resetToInitValue()
        
        # Apply assignments to transient or non-transient target variables.
        for ref, value in assgns.items():
            if ref not in nonTransientVars:
                transientVars[ref].setValueTo(value.eval(self, funcGetter))
            else:
                nonTransientVars[ref].setValueTo(value.eval(self, funcGetter))
        # Compute successor location set from source/destination location update.
        setOfLocation = self._setOfLocation.difference(src).union(dest)

        # Build current non-transient valuation for evaluating location transient-values.
        varGetter = { var.name: var.value for var in nonTransientVars.values() }
        for loc in locs:
            if loc not in locs:
                continue
            # Apply location-specific transient assignments after location transition.
            for ref, value in locs[loc].items():
                transientVars[ref].setValueTo(value.eval(varGetter, funcGetter))
        # Return a brand-new successor state object.
        return State(transientVars, nonTransientVars, setOfLocation)

    def get(self, name, default=None):
        """Return the value of the associated variable (both transient or non-transient),
        if it existsn otherwise return the default value."""
        # First try transient variables.
        var = self._transientVars.get(name)
        if var is not None:
            return var.value
        # Then try non-transient variables.
        var = self._nonTransientVars.get(name)
        if var is not None:
            return var.value
        # Fallback when variable name is not present.
        return default

    # def getLowMemRepr(self):
    #     """Return a string representation of the state (low memory representation)."""
    #     nonTransientVars, setOfLocation = self._hashRepr
    #     return str((sorted(nonTransientVars, key=lambda v: v.name), sorted(setOfLocation)))

    def getTupRepr(self, stateTemplate):
        """Return a tuple representation (an immutable list) representation of the state respecting stateTemplate."""
        # Initialize a dense vector following the variable/location order in stateTemplate.
        repr = [0] * len(stateTemplate)
        # Fill non-transient variable values at their template indices.
        for var in self._nonTransientVars.values():
            repr[stateTemplate[var.name]] = var.value
        # Encode active locations as binary indicators when multiple locations are tracked.
        if len(self._setOfLocation) > 1:
            for loc in self._setOfLocation:
                repr[stateTemplate[loc]] = 1
        # Return immutable tuple representation used by state-space indexing.
        return tuple(repr)

    def __contains__(self, item):
        # Membership checks both transient and non-transient variable namespaces.
        return item in self._transientVars or item in self._nonTransientVars

    def __eq__(self, value):
        # Two states are equal when their canonical hash representations match.
        return isinstance(value, State) and self._hashRepr == value._hashRepr

    def __hash__(self):
        # Hash is derived from immutable canonical representation.
        return hash(self._hashRepr)
