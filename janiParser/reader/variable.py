class Type(object):
    __slots__ = ["_type", "_bounds"]

    def __init__(self, type, bounds=None):
        self._type = type
        self._bounds = bounds

    @property
    def type(self):
        return self._type

    @property
    def bounds(self):
        if self._bounds is None:
            raise AttributeError("Type has no bounds attribute")
        return self._bounds

    def hasBounds(self):
        """Return True if type has bounds, otherwise return False."""
        return self._bounds is not None

    def isBoundedConsistent(self, value):
        """Return True if bounds are consistent, otherwise return False."""
        return not self.hasBounds() or (self._bounds[0] <= value <= self._bounds[1])

    def __eq__(self, value):
        return isinstance(value, Type) and self._type == value._type and self._bounds == value._bounds

    def __hash__(self):
        return hash((self._type, self._bounds))


class Constant(object):
    def __init__(self, name, type, value):
        self._name = name
        self._type = type
        self._value = value

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def value(self):
        return self._value


class Variable(object):
    __slots__ = ["_name", "_type", "_scope", "_initValue", "_value", "_transient"]
    def __init__(self, name, type, scope, initValue=None, transient=False):
        self._name = name
        self._type = type
        self._scope = scope
        self._initValue = initValue
        self._transient = transient
        self._value = self._initValue

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def scope(self):
        return self._scope

    @property
    def transient(self):
        return self._transient

    @property
    def value(self):
        return self._value

    @property
    def initValue(self):
        return self._initValue

    def hasInitValue(self):
        """Return True if variable has initial value, otherwise return False."""
        return self._initValue is not None

    def isGlobal(self):
        """Return True if variable is declared as a global variable, otherwise return False."""
        return self._scope[0] == "global"

    def resetToInitValue(self):
        """Reset variable to its initial value."""
        if not self.hasInitValue():
            raise AttributeError("Variable has no initial value")
        self._value = self._initValue

    def setValueTo(self, value):
        """Set variable value to the given value."""
        if self._type.isBoundedConsistent(value):
            self._value = value

    def instantiate(self):
        """Return a list of instantiated variables with all possible values."""
        assert not self._transient
        clone = lambda value: Variable(self._name, self._type, self._scope, value, self._transient)
        if self.hasInitValue():
            return [ clone(self._initValue) ]
        type = self.type
        if type.type == "real" or not type.hasBounds():
            raise Exception(f"Variable '{self._name}' contains a infinite range")
        low, up = type.bounds
        return list(map(clone, range(low, up + 1)))

    def __eq__(self, value):
        return isinstance(value, Variable) and self._name == value._name and \
            self._type == value._type and self._scope == value._scope and \
            self._transient == value._transient and self._value == value._value

    def __hash__(self):
        return hash((self._name, self._type, self._scope, self._transient, self._value))
