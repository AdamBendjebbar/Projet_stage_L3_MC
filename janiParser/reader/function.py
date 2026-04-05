class Function(object):
    def __init__(self, name, type, scope, params, body=None):
        self._name = name
        self._type = type
        self._scope = scope
        self._params = params
        self._body = body

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
    def parameters(self):
        return self._params

    @property
    def body(self):
        return self._body

    def addBody(self, body):
        if self._body is not None:
            raise Exception()
        self._body = body

    def isGlobal(self):
        """Return True if function is declared as a global function, otherwise return False."""
        return self._scope[0] == "global"
