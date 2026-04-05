from json import loads
import numpy as np

from janiParser.reader.model import JaniModel, JaniRModel
from janiParser.reader.variable import Type, Constant, Variable
from janiParser.reader.expression import Expression
from janiParser.reader.automata import Automata, Edge, EdgeDestination
from janiParser.reader.function import Function
from janiParser.reader.property import Property
from janiParser.exception import *

try:
    from typing_extensions import override
except ImportError:
    pass

import requests

# Set of supported unary operators.
UNARY_OPERATORS = {
    '¬',
    'floor',
    'ceil',
    'abs',
    'sgn',
    'trc'
}

# Set of supported binary operators.
BINARY_OPERATORS = {
    '∨',
    '∧',
    '⇒',
    '=',
    '≠',
    '<',
    '≤',
    '>',
    '≥',
    '+',
    '-',
    '*',
    '%',
    '/',
    'pow',
    'log',
    'min',
    'max'  
}

class JaniReader(object):
    """A reader for JANI (Json Automata Network Interface) model files.
    Parse the JANI model specification and build the corresponding model objects."""

    def __init__(self, path, modelParams=dict(), isLocalPath=True):
        """Initialize the JANI reader.

        Parameters:
            path: Local path or url path to the target JANI file.

            modelParams: Dictionary of model parameters (constants).

            isLocalPath: Indicate whether the 'path' filed is a local path or not.
        """
        assert isinstance(modelParams, dict)
        self._path = path
        self._modelParams = modelParams
        self._isLocalPath = isLocalPath
    
    def _load(self):
        if self._isLocalPath:
            with open(self._path, "r", encoding="utf-8-sig") as file:
                res = loads(file.read())
        else:
            res = requests.get(self._path)
            res.raise_for_status()
            res = loads(res.text)
        return res


    def build(self):
        """Parse and build a JANI model from the target file."""
        modelStruct = self._load()
        print("Start parsing")
        model = self._parseModel(modelStruct)
        print("Parsing success")
        return model

    def _parseModel(self, modelStruct):
        """Parse the root model structure."""
        if "name" not in modelStruct:
            raise JaniSyntaxError("Model must have a name")
        name = modelStruct["name"]

        if "type" not in modelStruct:
            raise JaniSyntaxError(f"The type of model '{name}' must be specified")
        type = modelStruct["type"]
        if type not in ["mdp", "dtmc"]:
            raise UnsupportedFeatureError(f"Unsupported type '{type}'\n"
                                          "Only 'mdp' and 'dtmc' are supported")
        model = JaniModel(name, type)

        self._parseActions(model, modelStruct.get("actions", []))
        if "system" not in modelStruct:
            raise JaniSyntaxError(f"Model '{name}' must have a system composition")
        self._parseSystem(model, modelStruct["system"])
        self._parseConstants(model, modelStruct.get("constants", []))
        self._parseVariables(model, modelStruct.get("variables", []), ("global", ))
        self._parseFunctions(model, modelStruct.get("functions", []), ("global", ))

        if "automata" not in modelStruct:
            raise JaniSyntaxError(f"Model '{name}' must have an automata composition")
        for autamata in modelStruct["automata"]:
            model.addAutomata(self._parseAutomata(model, autamata))
        
        if "properties" not in modelStruct or type == "dtmc":
            pass
        else:
            self._parseProperties(model, modelStruct["properties"])
        model.synchronize()
        return model

    def _parseActions(self, model: JaniModel, actions):
        """Parse action declarations."""
        for act in actions:
            if "name" not in act:
                raise JaniSyntaxError("Action must have a name")
            model.addAction(act["name"])
    
    def _parseSystem(self, model: JaniModel, system):
        """Parse system composition structure."""
        automataIndices = dict()
        preSyncActionss = list()
        if "elements" not in system:
            raise JaniSyntaxError("System composition must contain at least one automata")        
        for idx, elem in enumerate(system["elements"]):
            if "automaton" not in elem:
                raise JaniSyntaxError("System element must have an 'automaton' field")
            automataIndices[elem["automaton"]] = idx
        
        if "syncs" in system:
            for idx, sync in enumerate(system["syncs"]):
                preSyncActions = []
                for act in sync.get("synchronise", []):
                    if act is not None and not model.containsAction(act):
                        raise JaniSyntaxError(f"Undefined action '{act}' reference in synchronise")
                    preSyncActions.append(act)
                preSyncActionss.append((sync.get("result", f"act{idx}.sync"),
                                        preSyncActions))
        model.setSystemInformation(automataIndices, preSyncActionss)

    def _parseExpression(self, model: JaniModel, expr, scope, funcParams=set()):
        """Parse expression.
        
        Parameters:
            model: The current construction model.

            expr: The Jani expression to parse.

            scope: The current scope.

            funcParams: Function parameters when analyzing the function body, it's useful
                to distinguish between function parameter names and the names of other variables.
        
        Returns:
            out: The corresponding Expression object.
        """
        # Parse primitive constant expressions.
        if isinstance(expr, (int, float, bool)):
            return Expression.createConstExpression(expr)
        # Parse variable references.
        if isinstance(expr, str):
            if model.isConstantVariable(expr):
                return Expression.createConstExpression(model.getConstantValue(expr))
            if model.isGlobalVariable(expr) or expr in funcParams:
                name = expr
            else:
                assert len(scope) == 2
                name = f"{expr}.{scope[1]}"
                if not model.containsVariable(name):
                    raise JaniSyntaxError(f"Unknown variable '{name}' reference")
            return Expression("var", name)
        # Parse complex expressions.
        if "op" not in expr:
            raise JaniSyntaxError("Invalid expression structure.\n"
                                  "Complex expression must contain a 'op' field")
        operator = expr["op"]
        if operator in UNARY_OPERATORS:
            if "exp" not in expr:
                raise JaniSyntaxError("Unary expression must have an 'exp' field")
            return Expression.reduceExpression(operator,
                                               self._parseExpression(model, expr["exp"], scope, funcParams))
        if operator in BINARY_OPERATORS:
            if "left" not in expr or "right" not in expr:
                raise JaniSyntaxError("Binary expression must have a 'left' field and a 'right' field")
            return Expression.reduceExpression(operator,
                                               self._parseExpression(model, expr["left"], scope, funcParams),
                                               self._parseExpression(model, expr["right"], scope, funcParams))
        # Parse if-then-else expressions.
        if operator == "ite":
            if "if" not in expr or "then" not in expr or "else" not in expr:
                raise JaniSyntaxError("If-then-else expression must have an 'if' field, "
                                      "a 'then' field and an 'else' field")
            return Expression.reduceExpression("ite",
                                               self._parseExpression(model, expr["if"], scope, funcParams),
                                               self._parseExpression(model, expr["then"], scope, funcParams),
                                               self._parseExpression(model, expr["else"], scope, funcParams))
        # Parse function calls.
        if operator == "call":
            if "function" not in expr:
                raise JaniSyntaxError("Missing calling function identifier")
            name = expr["function"]
            if not model.containsFunction(name):
                raise JaniSyntaxError(f"Unknown function '{name}' reference")
            
            if "args" not in expr:
                raise JaniSyntaxError(f"Function '{name}' must have an 'args' field, "
                                      "even if the function takes no arguments")
            args = [ self._parseExpression(model, arg, scope, funcParams) for arg in expr["args"] ]
            if len(model.getFunction(name).parameters) != len(args):
                raise JaniSyntaxError(f"Mismatched argument size for function '{name}'")
            return Expression("call", name, args)
        raise UnsupportedFeatureError(f"Unsupported operator '{operator}'")

    def _parseType(self, model: JaniModel, type, scope):
        """Parse type definition."""
        # Parse primitive types.
        if isinstance(type, str):
            if type not in ["int", "real", "bool"]:
                raise UnsupportedFeatureError(f"Unsupported type '{type}'")
            return Type(type)
        # Parse complex types.
        if "kind" not in type:
            raise JaniSyntaxError("Complex type must have a 'kind' field")
        kind = type["kind"]
        if kind != "bounded":
            raise UnsupportedFeatureError(f"Unsupported kind '{kind}'.\n"
                                          "Only 'bounded' type are supported.")
        if "base" not in type:
            raise JaniSyntaxError("Complex type must have a 'base' field")
        base = type["base"]
        if base not in ["int", "real"]:
            raise UnsupportedFeatureError(f"Unsupported base '{base}'.\n",
                                          "Only 'int' and 'real' are supported")
        if "lower-bound" not in type:
            raise JaniSyntaxError("Complex type (bounded) must have a 'lower-bound' field")
        if "upper-bound" not in type:
            raise JaniSyntaxError("Complex type (bounded) must have a 'upper-bound' field")
        lower = self._parseExpression(model, type["lower-bound"], scope)
        upper = self._parseExpression(model, type["upper-bound"], scope)
        if not lower.isConstExpression() or not upper.isConstExpression():
            raise RequiredConstantExpressionError("Type bounds must be constant expression")
        lower = lower.eval()
        upper = upper.eval()
        return Type(base, (lower, upper))

    def _parseConstants(self, model: JaniModel, constants):
        """Parse constant declarations."""
        for const in constants:
            if "name" not in const:
                raise JaniSyntaxError("Constant must have a name")
            name = const["name"]

            if "type" not in const:
                raise JaniSyntaxError(f"The type of constant '{name}' must be specified")
            type = self._parseType(model, const["type"], ("global", ))

            if "value" not in const:
                if name not in self._modelParams:
                    raise MissingModelParameterError(f"Missing constant value '{name}'")
                value = self._modelParams[name]
            else:
                value = self._parseExpression(model, const["value"], ("global", ))
                if not value.isConstExpression():
                    raise RequiredConstantExpressionError(f"Constant '{name}' value must be constant expression")
                value = value.eval()
            model.addConstant(Constant(name, type, value))

    def _parseVariables(self, model: JaniModel, variables, scope):
        """Parse variables declarations."""
        isGlobal = scope[0] == "global"
        for var in variables:
            if "name" not in var:
                raise JaniSyntaxError("Variable must have a name")
            name = var["name"]
            if not isGlobal:
                assert len(scope) == 2
                name = f"{name}.{scope[1]}"

            if "type" not in var:
                raise JaniSyntaxError(f"The type of variable '{name}' must be specified")
            type = self._parseType(model, var["type"], scope)

            transient = var.get("transient", False)

            initValue = None
            if "initial-value" in var:
                initValue = self._parseExpression(model, var["initial-value"], scope)
                if not initValue.isConstExpression():
                    raise RequiredConstantExpressionError(f"Variable '{name}' initial value must be constant expression")
                initValue = initValue.eval()
            elif transient:
                raise JaniSyntaxError(f"Transient variable '{name}' must have an initial value")
            elif not type.hasBounds() and type.type == "real":
                raise JaniRRequirementError(f"Unbounded and real variable '{name}' must have an initial value")
            model.addVariable(Variable(name, type, scope, initValue, transient))

    def _parseFunctions(self, model: JaniModel, functions, scope):
        """Parse function declarations and definitions."""
        isGlobal = scope[0] == "global"
        # First parse - declare all functions.
        for func in functions:
            if "name" not in func:
                raise JaniSyntaxError("Function must have a name")
            name = func["name"]
            if not isGlobal:
                assert len(scope) == 2
                name = f"{name}.{scope[1]}"

            if "type" not in func:
                raise JaniSyntaxError(f"The type of function '{name}' must be specified")
            type = self._parseType(model, func["type"], scope)

            if "parameters" not in func:
                raise JaniSyntaxError(f"Function '{name}' must have a 'parameters' structure")
            params = self._parseParameters(model, func["parameters"], scope, name)
            model.declareFunction(Function(name, type, scope, params))
        
        # Second parse - process definition and check dependencies.
        dependencies = dict()
        for func in functions:
            name = func["name"]
            if not isGlobal:
                name = f"{name}.{scope[1]}"
            
            params = model.getFunction(name).parameters
            if "body" not in func:
                raise JaniSyntaxError(f"Function '{name}' must have a 'body' field")
            body = self._parseExpression(model, func["body"], scope, params)

            varRefs = self._getVarRefsFromExprStruct(model, func["body"])
            dependencies[name] = set(filter(model.containsFunction, varRefs))
            model.addFunctionBody(name, body)
        self._checkFuncDependencies(dependencies)

    def _parseParameters(self, model: JaniModel, parameters, scope, funcName):
        """Parse function parameters."""
        paramMap = dict()
        for param in parameters:
            if "name" not in param:
                raise JaniSyntaxError("Function parameter must have a name")
            name = param["name"]
            if name in paramMap:
                raise JaniSyntaxError(f"Duplicate function parameter '{name}' in function '{funcName}'")
            if model.containsVariable(name):
                raise JaniSyntaxError(f"Function parameter '{name}' conflicts with existing variable")
                
            if "type" not in param:
                raise JaniSyntaxError(f"The type of function parameter '{name}' must be specified")
            type = self._parseType(model, param["type"], scope)
            paramMap[name] = type
        return paramMap
    
    def _getVarRefsFromExprStruct(self, model: JaniModel, tarExpr):
        """Get all variable references in an expression."""
        varRefs = set()
        stack = [ tarExpr ]
        while stack:
            expr = stack.pop()
            if isinstance(expr, str) and not model.isConstantVariable(expr):
                varRefs.add(expr)
            elif isinstance(expr, dict):
                operator = expr["op"]
                if operator in UNARY_OPERATORS:
                    stack.append(expr["exp"])
                elif operator in BINARY_OPERATORS:
                    stack.append(expr["left"])
                    stack.append(expr["right"])
                elif operator == "ite":
                    stack.append(expr["if"])
                    stack.append(expr["then"])
                    stack.append(expr["else"])
                elif operator == "call":
                    stack.append(expr["function"])
                    stack.extend(expr["args"])
        return varRefs

    def _checkFuncDependencies(self, dependencies):
        """Check function dependencies."""
        def hasCircularDependency(v, status):
            if status[v] is not None:
                return not status[v]
            status[v] = False
            for u in dependencies[v]:
                if hasCircularDependency(u, status):
                    return True
            status[v] = True
            return False
        status = { v: None for v in dependencies }
        for v in dependencies:
            if status[v] is None and hasCircularDependency(v, status):
                raise JaniSyntaxError(f"Circular dependency in function '{v}'")

    def _parseLocations(self, model: JaniModel, locations, scope):
        """Parse automata locations."""
        _, automata = scope
        locMap = dict()
        for loc in locations:
            if "name" not in loc:
                raise JaniSyntaxError("Location must have a name")
            name = loc["name"]
            name = f"{name}.{automata}"

            transientValues = dict()
            for transValue in loc.get("transient-values", []):
                if "ref" not in transValue:
                    raise JaniSyntaxError("The 'assignment' structure must have a 'ref' field")
                ref = transValue["ref"]
                if not model.isGlobalVariable(ref):
                    ref = f"{ref}.{automata}"
                if not model.containsVariable(ref):
                    raise JaniSyntaxError(f"Unknown variable '{ref}' reference in location '{name}'")
                if not model.isTransientVariable(ref):
                    raise JaniSyntaxError(f"Contains non-transient variable reference '{ref}'")
                
                if "value" not in transValue:
                    raise JaniSyntaxError(f"The 'assignment' structure must have a 'value' field")
                value = value = self._parseExpression(model, transValue["value"], scope)

                varRefs = self._getVarRefsFromExprStruct(model, transValue["value"])
                if varRefs and np.all(list(map(model.isTransientVariable, varRefs))):
                    raise JaniSyntaxError("Value expression for a 'transient-values' structure must not refer to another transient variable")
                transientValues[ref] = value
            locMap[name] = transientValues
        return locMap

    def _parseAutomata(self, model: JaniModel, automata):
        """Parse automata."""
        if "name" not in automata:
            raise JaniSyntaxError("Automata must have a name")
        name = automata["name"]
        scope = ("local", name)

        self._parseVariables(model, automata.get("variables", []), scope)
        self._parseFunctions(model, automata.get("functions", []), scope)

        if "locations" not in automata:
            raise JaniSyntaxError(f"Automata '{name}' must have a 'locations' structure")
        locs = self._parseLocations(model, automata["locations"], scope)

        if "initial-locations" not in automata:
            raise JaniSyntaxError(f"Automata '{name}' must have a 'initial-locations' field")
        initLoc = automata["initial-locations"]
        if len(initLoc) == 0:
            raise JaniSyntaxError(f"Automata '{name}' must have at least one initial location")
        if len(initLoc) > 1:
            raise JaniRRequirementError(f"Automata '{name}' must have only one initial location")
        initLoc = { f"{initLoc[0]}.{name}" }
        if not initLoc.issubset(locs):
            raise JaniSyntaxError(f"Unknown location '{next(iter(initLoc))}' in automata '{name}'")
        
        if "edges" not in automata:
            raise JaniSyntaxError(f"Automara '{name}' must have an 'edges' structure")
        edges = [ self._parseEdge(model, edge, scope) for edge in automata["edges"] ]
        return Automata(name, locs, initLoc, edges)

    def _parseEdge(self, model: JaniModel, edge, scope):
        """Parse automata edge."""
        _, automata = scope
        if "location" not in edge:
            raise JaniSyntaxError(f"Edge in automata '{automata}' must have a 'location' field")
        src = edge["location"]
        src = { f"{src}.{automata}" }

        if "action" not in edge:
            act = "silent-action"
        else:
            act = edge["action"]
            if not model.containsAction(act):
                raise JaniSyntaxError(f"Unknown action '{act}' in automata '{automata}'")

        if "guard" not in edge:
            guard = Expression("bool", True)
        else:
            guard = edge["guard"]
            if "exp" not in guard:
                raise JaniSyntaxError(f"Guard in automata '{automata}' must have an 'exp' field")
            guard = self._parseExpression(model, guard["exp"], scope)
        
        if "destinations" not in edge:
            raise JaniSyntaxError(f"Edge in automata '{automata}' must have a 'destinations' structure")
        edgeDests = list()
        for destination in edge["destinations"]:
            if "location" not in destination:
                raise JaniSyntaxError("Edge destination in automata '{automata}' must have a 'location' field")
            dest = destination["location"]
            dest = { f"{dest}.{automata}" }

            if "probability" not in destination:
                prob = Expression("real", 1.)
            else:
                prob = destination["probability"]
                if "exp" not in prob:
                    raise JaniSyntaxError("The 'probability' structure must have a 'exp' field")
                prob = self._parseExpression(model, prob["exp"], scope)
            
            assgns = dict()
            for assgn in destination.get("assignments", []):
                if "ref" not in assgn:
                    raise JaniSyntaxError("The 'assignment' structure must have a 'ref' field")
                ref = assgn["ref"]
                if not model.isGlobalVariable(ref):
                    ref = f"{ref}.{automata}"
                if not model.containsVariable(ref):
                    raise JaniSyntaxError(f"Unknown variable '{ref}' reference in automata '{automata}'")

                if "value" not in assgn:
                    raise JaniSyntaxError(f"The 'assignment' structure must have a 'value' field")
                value = self._parseExpression(model, assgn["value"], scope)
                assgns[ref] = value
            edgeDests.append(EdgeDestination(dest, prob, assgns))
        return Edge(src, act, guard, edgeDests)


    # TODO
    def _parseProperties(self, model: JaniModel, properties):
        scope = ("global", )
        for property in properties:
            if "name" not in property:
                raise SyntaxError("A property must have a name")
            name = property["name"]

            if "expression" not in property:
                raise SyntaxError(f"Property '{name}' must have a property expression")
            propertyExpression = property["expression"]

            if "op" not in propertyExpression:
                raise SyntaxError("A property expression must have an operator")
            operator = propertyExpression["op"]
            if operator != "filter":
                raise SyntaxError("The top level operator must be 'filter'")
            
            if "values" not in propertyExpression:
                raise SyntaxError()
            values = propertyExpression["values"]

            try:
                criterion, mdpType, fs, reward, horizon = self.parsePropExprValues(model, values, scope)
                # print(name, criterion, mdpType, fs, reward, horizon)
                model.addProperty(Property(name, criterion, mdpType, fs, reward, horizon))
            except:
                # print(f"Failed to parser property '{name}'")
                continue
    
    def parsePropExprValues(self, model, propExprValues, scope):
        operator = propExprValues["op"]
        if operator in ["Pmax", "Pmin"]:
            criterion = operator[1:]
            if propExprValues["exp"]["op"] == "F":
                rewardExpr = self._parseExpression(model, propExprValues["exp"]["exp"], scope)
                terminalStateExpr = rewardExpr
            elif propExprValues["exp"]["op"] == "U":
                if propExprValues["exp"]["left"] == True:
                    rewardExpr = self._parseExpression(model, propExprValues["exp"]["right"], scope)
                    terminalStateExpr = rewardExpr
                else:
                    leftExpr = self._parseExpression(model, propExprValues["exp"]["left"], scope)
                    rightExpr = self._parseExpression(model, propExprValues["exp"]["right"], scope)
                    rewardExpr = rightExpr
                    terminalStateExpr = Expression("∨",
                                                   Expression("¬", leftExpr),
                                                   rightExpr)
            return criterion, "TotalRewardMDP", terminalStateExpr, rewardExpr, None
        if operator in ["Emax", "Emin"]:
            criterion = operator[1:]
            if "reach" in propExprValues:
                reach = self._parseExpression(model, propExprValues["reach"], scope)
                horizon = None
                resolutionModel = "AverageMDP"
            else:
                reach = Expression("bool", False)
                horizon = self._parseExpression(model, propExprValues["step-instant"], scope)
                if not horizon.isConstExpression():
                    raise RequiredConstantExpressionError(f"step-instant '{horizon}' value must be constant expression")
                horizon = horizon.eval()
                resolutionModel = "FiniteHorizonMDP"
            exp = self._parseExpression(model, propExprValues["exp"], scope)
            return criterion, resolutionModel, reach, exp, horizon
            
        
        if operator in UNARY_OPERATORS:
            return self.parsePropExprValues(model, propExprValues["exp"], scope)
        if operator in BINARY_OPERATORS:
            # TODO
            return self.parsePropExprValues(model, propExprValues["left"], scope)
        raise Exception()


class JaniRReader(JaniReader):
    def __init__(self, path, modelParams=dict()):
        super().__init__(path, modelParams)
    
    @override
    def _parseModel(self, modelStruct):
        """Parse the root model structure."""
        if "name" not in modelStruct:
            raise JaniSyntaxError("Model must have a name")
        name = modelStruct["name"]

        if "type" not in modelStruct:
            raise JaniSyntaxError(f"The type of model '{name}' must be specified")
        type = modelStruct["type"]
        gamma = horizon = None
        if type == "DiscountedMDP":
            if "gamma" not in modelStruct:
                raise JaniRSyntaxError("Discounted MDP must have a 'gamme' (discounted factor) field")
            gamma = modelStruct["gamma"]
            if gamma < 0 or gamma >= 1:
                raise JaniRSyntaxError("Discounted factor must be in range [0, 1)")
        elif type == "AverageMDP":
            pass # Do nothing.
        elif type == "TotalRewardMDP":
            pass # Do nothing.
        elif type == "FiniteHorizonMDP":
            if "gamma" not in modelStruct:
                raise JaniRSyntaxError("Finite horizon MDP must have a 'gamme' (discounted factor) field")
            if "horizon" not in modelStruct:
                raise SyntaxError("Finite horizon MDP must have a 'horizon' field")
            gamma = modelStruct["gamma"]
            horizon = modelStruct["horizon"]
        elif type == "MarkovChain":
            pass # Do nothing.
        else:
            raise UnsupportedFeatureError(f"Unsupported type '{type}' for JaniR model '{name}'")
        
        criterion = None
        if type != "MarkovChain":
            if "criterion" not in modelStruct:
                raise JaniRSyntaxError(f"Model '{name}' which is not 'MarkovChain' type must have a 'criterion' field")
            criterion = modelStruct["criterion"]
            if criterion not in ["max", "min"]:
                raise UnsupportedFeatureError(f"Unsupported criterion '{criterion}' for JaniR model '{name}'\n"
                                              "Only 'max' and 'min' are supported")
        model = JaniRModel(name, type, criterion, gamma, horizon)

        self._parseActions(model, modelStruct["actions"])
        if "system" not in modelStruct:
            raise JaniSyntaxError(f"Model '{name}' must have a system composition")
        self._parseSystem(model, modelStruct["system"])

        self._parseConstants(model, modelStruct.get("constants", []))
        self._parseVariables(model, modelStruct.get("variables", []), ("global", ))
        self._parseFunctions(model, modelStruct.get("functions", []), ("global", ))

        if "automata" not in modelStruct:
            raise JaniSyntaxError(f"Model '{name}' must have an automata composition")
        automata = modelStruct["automata"]
        if len(automata) != 1:
            raise JaniRSyntaxError(f"Model '{name}' supports only single automata")
        model.addAutomata(self._parseAutomata(model, automata[0]))
        return model
    
    @override
    def _parseSystem(self, model, system):
        """Parse system composition structure."""
        automataIndices = dict()
        if "elements" not in system:
            raise JaniSyntaxError("System composition must contain at least one automata")
        if len(system["elements"]) != 1:
            raise JaniRSyntaxError(f"Model '{model.name}' supports only single automata")
        
        elem = system["elements"][0]
        if "automaton" not in elem:
            raise JaniSyntaxError("System element must have an 'automaton' field")
        automataIndices[elem["automaton"]] = 0
        
        model.setSystemInformation(automataIndices)
    
    @override
    def _parseEdge(self, model, edge, scope):
        """Parse automata edge."""
        _, automata = scope
        if "location" not in edge:
            raise JaniSyntaxError(f"Edge in automata '{automata}' must have a 'location' field")
        src = edge["location"]
        src = { f"{src}.{automata}" }

        if "action" not in edge:
            raise JaniRSyntaxError(f"Edge in automata '{automata}' must have a 'action' field")
        act = edge["action"]
        if not model.containsAction(act):
            raise JaniSyntaxError(f"Unknown action '{act}' in automata '{automata}'")

        if "guard" not in edge:
            guard = Expression("bool", True)
        else:
            guard = edge["guard"]
            if "exp" not in guard:
                raise JaniSyntaxError(f"Guard in automata '{automata}' must have an 'exp' field")
            guard = self._parseExpression(model, guard["exp"], scope)
        
        if "destinations" not in edge:
            raise JaniSyntaxError(f"Edge in automata '{automata}' must have a 'destinations' structure")
        edgeDests = list()
        for destination in edge["destinations"]:
            if "location" not in destination:
                raise JaniSyntaxError("Edge destination in automata '{automata}' must have a 'location' field")
            dest = destination["location"]
            dest = { f"{dest}.{automata}" }

            if "probability" not in destination:
                prob = Expression("real", 1.)
            else:
                prob = destination["probability"]
                if "exp" not in prob:
                    raise JaniSyntaxError("The 'probability' structure must have a 'exp' field")
                prob = self._parseExpression(model, prob["exp"], scope)

            if "reward" not in destination:
                reward = Expression("real", 0.)
            else:
                reward = destination["reward"]
                if "exp" not in reward:
                    raise JaniRSyntaxError("The 'reward' structure must have a 'exp' field")
                reward = self._parseExpression(model, reward["exp"], scope)            
            
            assgns = dict()
            for assgn in destination.get("assignments", []):
                if "ref" not in assgn:
                    raise JaniSyntaxError("The 'assignment' structure must have a 'ref' field")
                ref = assgn["ref"]
                if not model.isGlobalVariable(ref):
                    ref = f"{ref}.{automata}"
                if not model.containsVariable(ref):
                    raise JaniSyntaxError(f"Unknown variable '{ref}' reference in automata '{automata}'")

                if "value" not in assgn:
                    raise JaniSyntaxError(f"The 'assignment' structure must have a 'value' field")
                value = self._parseExpression(model, assgn["value"], scope)
                assgns[ref] = value
            edgeDests.append(EdgeDestination(dest, prob, assgns, reward))
        return Edge(src, act, guard, edgeDests)
    
# if __name__ == "__main__":
#     ###########################################################
#     # PPDDL instances

#     # 462400 states
#     # 22 actions
#     # 3851520 transitions and 0 deadlocks
#     # In total, 3851520 transitions
#     # python3 reader.py  1149,72s user 3,50s system 99% cpu 19:17,56 total

#     # path = "../../benchmarks/ppddl2jani/zenotravel.4-2-2.v1man.jani"
#     # reader = JaniReader(path, modelParams={})

#     ###########################################################
#     # Prism instances

#     # 1296 states
#     # 3 actions
#     # 2412 transitions and 0 deadlocks
#     # In total, 2412 transitions
#     # python3 reader.py  0,44s user 0,02s system 99% cpu 0,470 total
#     path = "../../benchmarks/prism2jani/consensus.2.v1.jani"
#     reader = JaniReader(path, modelParams={ "K": 10 })

#     # path = "../../benchmarks/prism2jani/csma.2-2.v1.jani"
#     # reader = JaniReader(path, modelParams={})

#     # 12828 states
#     # 1 actions
#     # 21795 transitions and 0 deadlocks
#     # In total, 21795 transitions
#     # python3 reader.py  343,71s user 0,28s system 99% cpu 5:44,92 total
#     # path = "../../benchmarks/prism2jani/eajs.2.v1.jani"
#     # reader = JaniReader(path, modelParams={ "energy_capacity": 100, "B": 100 })

#     # 956 states
#     # 6 actions
#     # 3696 transitions and 0 deadlocks
#     # In total, 3696 transitions
#     # python3 reader.py  0,50s user 0,02s system 99% cpu 0,522 total
#     # path = "../../benchmarks/prism2jani/philosophers-mdp.3.v1.jani"
#     # reader = JaniReader(path, modelParams={})


#     # 96894 states
#     # 10 actions
#     # 129170 transitions and 0 deadlocks
#     # In total, 129170 transitions
#     # python3 reader.py  1714,69s user 2,12s system 99% cpu 28:43,84 total
#     # path = "../../benchmarks/prism2jani/pacman.v2.jani"
#     # reader = JaniReader(path, modelParams={ "MAXSTEPS": 15 })

#     # reader.build().writeJaniR("out.txt", "all_before_min")
#     # model = JaniRReader("out.txt", {}).build()
#     # stateSpace, actionSpace, Transitions, Rewards = model.buildTransitionAndReward()
#     # mdp = mmdp.TotalRewardMDP(model.criterion, stateSpace, actionSpace, Transitions, Rewards)
#     # with open("out_1.txt", "w") as file:
#     #     print(mdp.ValueIteration(1e-10, 1000), file=file)
