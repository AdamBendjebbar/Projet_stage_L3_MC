import numpy as np

class Expression(object):
    _OPERATORS = {
        # Logical operator.
        "∨": np.logical_or,
        "∧": np.logical_and,
        "¬": np.logical_not,
        "⇒": lambda x, y: np.logical_or(np.logical_not(x), y),

        # Comparison operator.
        "=": np.equal,
        "≠": np.not_equal,
        "<": np.less,
        "≤": np.less_equal,
        ">": np.greater,
        "≥": np.greater_equal,

        # Arithmetic operator.
        "+": np.add,
        "-": np.subtract,
        "*": np.multiply,
        "%": np.mod,
        "/": np.divide,
        "pow": np.power,
        "log": np.log,
        "floor": np.floor,
        "ceil": np.ceil,
        "abs": np.abs,
        "sgn": np.sign,
        "min": np.minimum,
        "max": np.maximum,
        "trc": np.trunc
    }
    
    def __init__(self, kind, *args):
        self._kind = kind
        self._args = args

        if self._kind not in self._OPERATORS and self._kind not in ["int", "real", "bool", "var", "ite", "call"]:
            raise Exception(f"Unsupported kind '{self._kind}'")
        
    @property
    def kind(self):
        return self._kind
    
    def isConstExpression(self):
        """Return True if it is a constante expression, otherwise return False."""
        return self.kind in ["int", "real", "bool"]

    def eval(self, varGetter=None, funcGetter=None, funcVarGetter=None):
        """Evaluate the expression."""
        kind = self.kind
        # Evaluate a constant expression.
        if self.isConstExpression():
            return self._args[0]
        # Evaluate a variable (model variable or function or function variable) expression.
        if kind == "var":
            name = self._args[0]
            # Return the associated value.
            if varGetter is not None and name in varGetter:
                return varGetter.get(name)
            if funcVarGetter is not None and name in funcVarGetter:
                return funcVarGetter.get(name)
            # Return the associated Function class.
            if funcGetter is not None and name in funcGetter:
                return funcGetter.get(name)
            raise KeyError(f"Unrecognized variable name '{name}'")
        # Evaluate an if-else expression.
        if kind == "ite":
            condition, then, otherwise = self._args
            if condition.eval(varGetter, funcGetter, funcVarGetter):
                return then.eval(varGetter, funcGetter, funcVarGetter)
            return otherwise.eval(varGetter, funcGetter, funcVarGetter)
        # Evaluate a function call expression.
        if kind == "call":
            name, args = self._args
            if funcGetter is None or name not in funcGetter:
                raise KeyError(f"Unrecognized function name '{name}'")
            function = funcGetter.get(name)
            # Evaluate all the function arguments.
            evaluatedArgs = [
                arg.eval(varGetter, funcGetter, funcVarGetter)
                for arg in args
            ]
            return function.body.eval(
                varGetter,
                funcGetter,
                # Build the function variable getter.
                funcVarGetter={
                    funcVar: arg
                    for funcVar, arg in zip(function.parameters, evaluatedArgs)
                }
            )
        # Evaluate a unary or binary expression.
        if kind in self._OPERATORS:
            operator = self._OPERATORS[kind]
            # Evaluate a unary expression.
            if len(self._args) == 1:
                return operator(self._args[0].eval(varGetter, funcGetter, funcVarGetter))
            # Evaluate a binary expression.
            left, right = self._args
            return operator(
                left.eval(varGetter, funcGetter, funcVarGetter),
                right.eval(varGetter, funcGetter, funcVarGetter)
            )

    @staticmethod
    def createConstExpression(value):
        """Create a constant expression."""
        if isinstance(value, (int, np.integer)):
            return Expression("int", int(value))
        if isinstance(value, (float, np.floating)):
            return Expression("real", float(value))
        if isinstance(value, (bool, np.bool_)):
            return Expression("bool", bool(value))
        raise Exception(f"Unsupported value type '{type(value)}' for constant expression.")
    
    @staticmethod
    def reduceExpression(op, *args: 'Expression'):
        """Return a reduced expression."""
        # If-else or call expression.
        if op not in Expression._OPERATORS:
            if op == "ite":
                condition, then, otherwise = args
                if condition.isConstExpression():
                    if condition.eval():
                        return then
                    return otherwise
                return Expression("ite", *args)
            if op == "call":
                return Expression("call", *args)
            raise Exception(f"Unrecognized operator '{op}'")
        arity = len(args)
        operator = Expression._OPERATORS[op]
        # Unary expression.
        if arity == 1:
            if args[0].isConstExpression():
                return Expression.createConstExpression(args[0].eval())
        # Binary expression.
        elif arity == 2:
            left, right = args
            leftConstExpression = left.isConstExpression()
            rightConstExpression = right.isConstExpression()
            if leftConstExpression and rightConstExpression:
                return Expression.createConstExpression(operator(left.eval(), right.eval()))
            # Case with binary logical expression.
            if op == "∨":
                # True or expr | expr or True -> True.
                # False or expr | expr or False -> expr.
                if leftConstExpression:
                    if left.eval():
                        return Expression("bool", True)
                    return right
                if rightConstExpression:
                    if right.eval():
                        return Expression("bool", True)
                    return left
            if op == "∧":
                # True and expr | expr and True -> expr.
                # False and expr | expr and False -> False.
                if leftConstExpression:
                    if not left.eval():
                        return Expression("bool", False)
                    return right
                if rightConstExpression:
                    if not right.eval():
                        return Expression("bool", False)
                    return left
            if op == "⇒":
                # (not False) or expr -> True.
                # (not True) or expr -> expr.
                # (not expr) ot True -> True.
                # (not expr) or False -> not expr.
                if leftConstExpression:
                    if not left.eval():
                        return Expression("bool", True)
                    return right
                if rightConstExpression:
                    if right.eval():
                        return Expression("bool", True)
                    return Expression("¬", left)
        return Expression(op, *args)
    
    def toJaniRRepresentation(self):
        kind = self._kind
        if self.isConstExpression():
            return self._args[0]
        if kind == "var":
            return self._args[0].split(".")[0]
        if kind == "ite":
            condition, then, otherwise = self._args
            return {
                "op": "ite",
                "if": condition.toJaniRRepresentation(),
                "then": then.toJaniRRepresentation(),
                "else": otherwise.toJaniRRepresentation()
            }
        if kind == "call":
            identifier, args = self._args
            return {
                "op": "call",
                "function": identifier,
                "args": [
                    arg.toJaniRRepresentation()
                    for arg in args
                ]
            }
        if kind in self._OPERATORS:
            arity = len(self._args)
            if arity == 1:
                return {
                    "op": kind,
                    "exp": self._args[0].toJaniRRepresentation()
                }
            else:
                left, right = self._args
                return {
                    "op": kind,
                    "left": left.toJaniRRepresentation(),
                    "right": right.toJaniRRepresentation()
                }

    def __str__(self):
        arity = len(self._args)
        kind = self._kind
        if arity == 1:
            if kind in ["int", "real", "bool", "var"]:
                return str(self._args[0])
            return f"{kind}{self._args[0]}"
        if arity == 2:
            if kind == "call":
                pass
            left, right = self._args
            if kind in ("pow", "log", "floor", "ceil", "abs", "sgn", "min", "max", "trc"):
                return f"{kind} ({left}, {right})"
            return f"({left}) {kind} ({right})"
        if arity == 3:
            if kind == "ite":
                condition, then, otherwise = self._args
                return f"if ({condition}) {{{then}}} else {{{otherwise}}}"
        pass

    def __repr__(self):
        return self.__str__()
