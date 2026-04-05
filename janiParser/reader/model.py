from itertools import product
from collections import deque
from copy import deepcopy

import numpy as np

from janiParser.reader.automata import Automata, Edge, EdgeDestination
from janiParser.reader.variable import Type
from janiParser.reader.expression import Expression
from janiParser.reader.state import State
from janiParser.exception import *

try:
    from typing_extensions import override
except ImportError:
    pass

class JaniModel(object):
    """A class represents a JANI (Json Automata Network Interface) model."""
    def __init__(self, name, type):
        self._name = name
        self._type = type
        
        self._actions = dict()
        self._actionCounter = 0
        self._constants = dict()

        self._transientVars = dict()
        self._nonTransientVars = dict()

        self._functions = dict()

        self._automataIndices = None
        self._preSyncActionss = None
        self._nonSyncAutomatas = None
        self._automata = None

        self._properties = dict()

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    def addAction(self, action):
        """Add an action to the model."""
        if action in self._actions:
            raise KeyError(f"Action '{action}' already exists in model '{self.name}'")
        self._actions[action] = self._actionCounter
        self._actionCounter += 1

    def containsAction(self, action):
        """Return True if the model contains action, otherwise return False."""
        return action in self._actions

    def addConstant(self, constant):
        """Add a constant variable to the model."""
        name = constant.name
        if name in self._constants:
            raise KeyError(f"Constant variable '{name}' already exists in model '{self.name}'")
        self._constants[name] = constant
    
    def isConstantVariable(self, name):
        """Return True if name is declared as a constant variable in the model."""
        return name in self._constants
    
    def getConstantValue(self, name):
        """Get the associated value of a constant variable."""
        constant = self._constants.get(name)
        if constant is not None:
            return constant.value
        raise KeyError(f"Unrecognized constant name '{name}' for model '{self.name}'")
    
    def addVariable(self, variable):
        """Add a variable to the model."""
        name = variable.name
        transient = variable.transient
        if transient:
            if name in self._transientVars:
                raise KeyError(f"Transient variable '{name}' already exists in model '{self.name}'")
            self._transientVars[name] = variable
        else:
            if name in self._nonTransientVars:
                    raise KeyError(f"Non-transient variable '{name}' already exists in model '{self.name}'")
            self._nonTransientVars[name] = variable
    
    def isGlobalVariable(self, name):
        """Return True if the variable name is declared as a constant or global variable."""
        if self.isConstantVariable(name):
            return True
        variable = self._transientVars.get(name)
        if variable is not None:
            return variable.isGlobal()
        variable = self._nonTransientVars.get(name)
        if variable is not None:
            return variable.isGlobal()
        return False

    def isTransientVariable(self, name):
        """Return True if name is declared as a transient variable."""
        return name in self._transientVars
    
    def containsVariable(self, name):
        """Return True if the model contains variable name, otherwise return False."""
        return name in self._constants or name in self._transientVars or name in self._nonTransientVars

    def declareFunction(self, function):
        """Declare a function in the model."""
        name = function.name
        if name in self._functions:
            raise KeyError(f"Function '{name}' already exists in model '{self.name}'")
        self._functions[name] = function

    def addFunctionBody(self, name, body):
        """Add a function body to a declared function."""
        if name not in self._functions:
            raise KeyError(f"Undeclared function name '{name}' for model '{self.name}'")
        self._functions[name].addBody(body)
    
    def containsFunction(self, name):
        """Return True if the model contains function, otherwise return False."""
        return name in self._functions
    
    def getFunction(self, name):
        """Return the function associated with the name."""
        if not self.containsFunction(name):
            raise SyntaxError(f"Unrecognized function '{name}' for model '{self.name}'")
        return self._functions[name]

    def setSystemInformation(self, automataIndices, preSysnActionss):
        """Add system information to the model."""
        if self._automataIndices is not None and self._preSyncActionss is not None:
            raise Exception("System information has already been defined and cannot be redefined")
        self._automataIndices = automataIndices
        self._preSyncActionss = preSysnActionss
        self._nonSyncAutomatas = [None] * len(self._automataIndices)

    def addAutomata(self, automata):
        """Add a automata to the model."""
        if self._automataIndices is None:
            raise Exception("System information not defined")
        name = automata.name
        if name not in self._automataIndices:
            raise KeyError(f"Unrecognized automata '{name}' for model '{self.name}'")
        idx = self._automataIndices[name]
        self._nonSyncAutomatas[idx] = automata

    def getInitStates(self):
        """Return all possible initial states."""
        states = []
        locs = self._automata.locations
        setOfLocation = self._automata.initLocation
        for vars in product(*map(lambda x: x.instantiate(), self._nonTransientVars.values())):
            transientVars = deepcopy(self._transientVars)
            nonTransientVars = { var.name: var for var in vars }
            varGetter = { var.name: var.value for var in vars }
            for loc in setOfLocation:
                if loc not in locs:
                    continue
                for ref, value in locs[loc].items():
                    transientVars[ref].setValueTo(value.eval(varGetter, self._functions))
            states.append(State(transientVars, nonTransientVars, setOfLocation))
        return states

    def synchronize(self):
        """Synchronize automata."""
        if self._automata is not None:
            raise Exception("Synchronization is complete")
        print("Start synchronization")
        self._automata = self._synchronizeAutomata()
        print("Synchronization success")
        del self._nonSyncAutomatas
    
    def _synchronizeAutomata(self):
        # If there is only 1 automata, then synchronization is not necessary,
        # we simply return this automata directly.
        if len(self._nonSyncAutomatas) == 1:
            return self._nonSyncAutomatas[0]
        
        syncActions = dict()
        syncEdges = list()
        syncInitLoc = set()
        syncLocs = dict()

        # Preprocessing.
        actionMapEdgesList = []
        for automata in self._nonSyncAutomatas:
            syncInitLoc.update(automata.initLocation)
            syncLocs.update(automata.locations)

            actionMapEdges = dict()
            for edge in automata.edges:
                action = edge.action
                if action == "silent-action":
                    syncEdges.append(edge)
                else:
                    actionMapEdges.setdefault(action, []).append(edge)
            actionMapEdgesList.append(actionMapEdges)

        # Synchronization.
        syncActionCounter = 0
        for result, preSyncActions in self._preSyncActionss:
            syncActions[result] = syncActionCounter
            syncActionCounter += 1
            preSyncEdgesList = []
            for i, preSyncAcition in enumerate(preSyncActions):
                if preSyncAcition is not None:
                    preSyncEdgesList.append(actionMapEdgesList[i][preSyncAcition])

            for edgeComb in product(*preSyncEdgesList):
                syncEdges.append(self._synchronizeEdge(edgeComb, result))
        self._actions = syncActions
        self._actionCounter = syncActionCounter
        return Automata("Main", syncLocs, syncInitLoc, syncEdges)
    
    def _synchronizeEdge(self, edgeComb, action):
        syncSrc = set()
        syncGuard = Expression("bool", True)

        preSyncEdgeDestsList = []
        for edge in edgeComb:
            syncSrc.update(edge.source)
            syncGuard = Expression.reduceExpression("∧", syncGuard, edge.guard)
            preSyncEdgeDestsList.append(edge.edgeDestinations)
        
        syncEdgeDests = [
            self._synchronizeEdgeDestination(edgeDestComb) for edgeDestComb in product(*preSyncEdgeDestsList)
        ]
        return Edge(syncSrc, action, syncGuard, syncEdgeDests)
    
    def _synchronizeEdgeDestination(self, edgeDestComb):
        syncDest = set()
        syncProb = Expression("real", 1.)
        syncAssngs = dict()

        for edgeDest in edgeDestComb:
            syncDest.update(edgeDest.destination)
            syncProb = Expression.reduceExpression("*", syncProb, edgeDest.probability)
            syncAssngs.update(edgeDest.assignments)
        return EdgeDestination(syncDest, syncProb, syncAssngs)

    def addProperty(self, property):
        """Add a property to the model."""
        name = property.name
        if name in self._properties:
            raise KeyError(f"Property '{name}' already exists in model '{self.name}'")
        self._properties[name] = property

    def getPropertyNames(self):
        """Return all property names."""
        return self._properties.keys()
    
    def exploreStateSpace(self, initStates, stateTemplate, terminalStateExpr, rewardExpr, singleActRequirement=False):
        """Explore all reachable states from initial states.

        Parameters:
            initStates: List of all possible initial states.

            stateTemplate: State template.

            terminalStateExpr: Terminal state expression.

            rewardExpr: Reward expression.

            singleActRequirement: Single action requirement (as in case of MC).

        Returns:
            out:
            * A state dict which maps a tuple representation (or immutable list representation) to each state.
            * An absorbing state set.
            * A transition dict which maps a tuple (probability, reward) to each triplet (s, s', a).
            * An actions dict which maps an unique index to each action.
        """
        visitedStates = set()
        transitions = dict()
        absorbingStates = set()
        maxSilentActCnt = 0

        stateToTupReprs = dict()
        queue = deque()
        for initState in initStates:
            stateToTupReprs[initState] = initState.getTupRepr(stateTemplate)
            queue.append(initState)
        
        edges = self._automata.edges
        locs = self._automata.locations
        funcGetter = self._functions
        while queue:
            s = queue.pop()
            if s not in visitedStates:
                sTupRepr = s.getTupRepr(stateTemplate)
                visitedStates.add(s)

                if terminalStateExpr.eval(s):
                    absorbingStates.add(s)
                    continue

                print(len(visitedStates), end="\r")

                deadlock = True
                silentActCnt = 0
                for edge in edges:
                    if not edge.isSatisfied(s.setOfLocation, s, funcGetter):
                        continue

                    deadlock = False
                    src = edge.source
                    act = edge.action
                    if singleActRequirement:
                        act = "act"
                    elif act == "silent-action":
                        act = f"{act}_{silentActCnt}"
                        silentActCnt += 1
                    for edgeDest in edge.edgeDestinations:
                        sPrime = s.clone(src,
                                         edgeDest.destination,
                                         edgeDest.assignments,
                                         locs,
                                         funcGetter)
                        prob = edgeDest.probability.eval(s, funcGetter)
                        reward = rewardExpr.eval(sPrime)

                        if prob <= 0.:
                            continue

                        sPrimeTupRepr = stateToTupReprs.get(sPrime)

                        if sPrimeTupRepr is None:
                            stateToTupReprs[sPrime] = sPrimeTupRepr = sPrime.getTupRepr(stateTemplate)
                        key = (sTupRepr, sPrimeTupRepr, act)
                        transitions[key] = transitions.get(key, np.array([0., 0.])) + [prob, reward]
                        if sPrime not in visitedStates:
                            queue.append(sPrime)
                        else:
                            del sPrime
                if deadlock:
                    absorbingStates.add(s)
                maxSilentActCnt = max(maxSilentActCnt, silentActCnt)
            else:
                del s
        if singleActRequirement:
            actions = { "act": 0 }
        else:
            actions = deepcopy(self._actions)
            actions.update({ f"silent-action_{i}": i + self._actionCounter for i in range(maxSilentActCnt) })
        return stateToTupReprs, absorbingStates, transitions, actions
        
    def _getStateVarInformation(self):
        """"""
        nonTransientVars = self._nonTransientVars.values()
        locations = self._automata.locations
        # Build a state template, which gives a fixed order to state variables (non-transient variables and locations)
        # and facilitates conversion from state to matrix index.
        stateTemplate = { var.name: idx for idx, var in enumerate(nonTransientVars) }

        # Build 2 dictionaries which associates each state variables with its type and initial value.
        # These 2 information are useful when rewriting the model in a JaniR file.
        stateVarTypes = { var.name: var.type for var in nonTransientVars }
        stateVarInitValues = { var.name: var.initValue for var in nonTransientVars }

        # If set of locations is greater than 1 (as in case of synchronization), then we add the location
        # as a state variable. Otherwise, it's unnecessary since it's always true.
        if len(locations) > 1:
            dev = len(stateTemplate)
            stateTemplate.update({ loc: idx + dev for idx, loc in enumerate(locations) })
            # Add locations as binary variables
            stateVarTypes.update({ loc: Type("bool") for loc in locations })
            initLoc = self._automata.initLocation
            stateVarInitValues.update({ loc: loc in initLoc for loc in locations })
        return stateTemplate, stateVarTypes, stateVarInitValues

    def getMDPData(self, name):
        """Get all required and useful data to build a Marmote MDP."""
        if self._type != "mdp":
            raise Exception(f"Inconsistent model type '{self._type}'")

        if name not in self._properties:
            raise KeyError(f"Unknown property '{name}' for model '{self._name}'")
        prop = self._properties[name]
        criterion = prop.criterion

        initStates = self.getInitStates()
        print(f"{len(initStates)} initial states.")

        stateTemplate, stateVarTypes, stateVarInitValues = self._getStateVarInformation()

        # Explore all reachable states from initial states.
        stateToTupReprs, absorbingStates, transitons, actions = self.exploreStateSpace(initStates,
                                                                                       stateTemplate,
                                                                                       prop.terminalStateExpression,
                                                                                       prop.rewardExpression)
        print(f"{len(stateToTupReprs)} states, "
              f"{len(actions)} actions, "
              f"{len(transitons) + len(absorbingStates)} transitions")
        
        # Build and fill 'transitionDict', which is an intermediary structure used to facilitate access
        # to transition probabilities
        transitionDict = {
            action: {
                sTupRepr: dict() for sTupRepr in stateToTupReprs.values()
            } for action in actions
        }
        for (sTupRepr, sPrimeTupRepr, action), data in transitons.items():
            prob, reward = data
            transitionDict[action][sTupRepr][sPrimeTupRepr] = np.array([prob, reward])

        MDPData = {
            "name": self._name,
            "type": prop.resolutionModel,
            "criterion": criterion,
            "horizon": prop.horizon,
            "states": set(stateToTupReprs.values()),
            "initial-states": [ stateToTupReprs[s] for s in initStates ],
            "absorbing-states": { stateToTupReprs[s] for s in absorbingStates },
            "actions": actions,
            "transition-dict": transitionDict,
            "state-template": stateTemplate,
            "state-variable-types": stateVarTypes,
            "state-variable-initial-values": stateVarInitValues,
        }
        return MDPData

    def getMCData(self):
        """Get all required and useful data to build a Marmote MC."""
        if self._type != "dtmc":
            raise Exception(f"Inconsistent model type '{self._type}'")

        initStates = self.getInitStates()
        print(f"{len(initStates)} initial states.")

        stateTemplate, stateVarTypes, stateVarInitValues = self._getStateVarInformation()

        # Explore all reachable states from initial states.
        stateToTupReprs, absorbingStates, transitons, actions = self.exploreStateSpace(initStates,
                                                                                       stateTemplate,
                                                                                       Expression("bool", False),
                                                                                       Expression("int", 0),
                                                                                       singleActRequirement=True)
        assert len(actions) == 1
        print(f"{len(stateToTupReprs)} states, "
              f"{len(actions)} actions, "
              f"{len(transitons) + len(absorbingStates)} transitions")
        
        # Build and fill 'transitionDict', which is an intermediary structure used to facilitate access
        # to transition probabilities
        transitionDict = {
            sTupRepr: dict() for sTupRepr in stateToTupReprs.values()
        }
        for (sTupRepr, sPrimeTupRepr, _), data in transitons.items():
            prob, _ = data
            transitionDict[sTupRepr][sPrimeTupRepr] = prob

        MCData = {
            "name": self._name,
            "type": "MarkovChain",
            "states": set(stateToTupReprs.values()),
            "initial-states": [ stateToTupReprs[s] for s in initStates ],
            "absorbing-states": { stateToTupReprs[s] for s in absorbingStates },
            "actions": actions,
            "transition-dict": transitionDict,
            "state-template": stateTemplate,
            "state-variable-types": stateVarTypes,
            "state-variable-initial-values": stateVarInitValues,
            "number-transitions": len(transitons) + len(absorbingStates)
        }
        return MCData


class JaniRModel(JaniModel):
    def __init__(self, name, type, criterion, gamma=None, horizon=None):
        super().__init__(name, type)
        self._criterion = criterion
        self._gamma = gamma
        self._horizon = horizon

    @override
    def addAutomata(self, automata):
        """Add a automata to the model."""
        if self._automataIndices is None:
            raise Exception("System information not defined")
        name = automata.name
        if name not in self._automataIndices:
            raise KeyError(f"Unknown automata '{name}' for model '{self.name}'")
        self._automata = automata

    @override
    def setSystemInformation(self, automataIndices):
        """Add system information to the model."""
        if self._automataIndices is not None and self._preSyncActionss is not None:
            raise Exception("System information has already been defined and cannot be redefined")
        self._automataIndices = automataIndices
    
    @override
    def synchronize(self):
        raise UnsupportedFeatureError(f"Synchronization does not support by JaniR model '{self._name}'")

    @override
    def exploreStateSpace(self, initStates, stateTemplate, singleActRequirement=False):
        """Explore all reachable states from initial states.

        Parameters:
            initStates: List of all possible initial states.

            stateTemplate: State template.

            singleActRequirement: Single action requirement (as in case of MC).

        Returns:
            out:
            * A state dict which maps a tuple representation (or immutable list representation) to each state.
            * An absorbing state set.
            * A transition dict which maps a tuple (probability, reward) to each triplet (s, s', a).
            * An actions dict which maps an unique index to each action.
        """
        visitedStates = set()
        transitions = dict()
        absorbingStates = set()

        stateToTupReprs = dict()
        queue = deque()
        for initState in initStates:
            stateToTupReprs[initState] = initState.getTupRepr(stateTemplate)
            queue.append(initState)
        
        edges = self._automata.edges
        locs = self._automata.locations
        funcGetter = self._functions
        while queue:
            s = queue.popleft()
            if s not in visitedStates:
                sTupRepr = s.getTupRepr(stateTemplate)
                visitedStates.add(s)

                print(len(visitedStates), end="\r")

                deadlock = True
                for edge in edges:
                    if not edge.isSatisfied(s.setOfLocation, s, funcGetter):
                        continue
        
                    deadlock = False
                    src = edge.source
                    act = edge.action
                    if singleActRequirement:
                        act = "act"
                    for edgeDest in edge.edgeDestinations:
                        sPrime = s.clone(src,
                                         edgeDest.destination,
                                         edgeDest.assignments,
                                         locs,
                                         funcGetter)
                        prob = edgeDest.probability.eval(s, funcGetter)
                        reward = edgeDest.reward.eval(sPrime, funcGetter)

                        if prob <= 0.:
                            continue

                        sPrimeTupRepr = stateToTupReprs.get(sPrime)
                        if sPrimeTupRepr is None:
                            stateToTupReprs[sPrime] = sPrimeTupRepr = sPrime.getTupRepr(stateTemplate)
                        
                        key = (sTupRepr, sPrimeTupRepr, act)
                        transitions[key] = transitions.get(key, np.array([0., 0.])) + [prob, reward]
                        if sPrime not in visitedStates:
                            queue.append(sPrime)
                        else:
                            del sPrime
                if deadlock:
                    absorbingStates.add(s)
            else:
                del s
        actions = { "act": 0 } if singleActRequirement else deepcopy(self._actions)

        return stateToTupReprs, absorbingStates, transitions, actions
    
    @override
    def getMDPData(self):
        """Get all required and useful data to build a Marmote MDP."""
        if self._type not in ["DiscountedMDP", "AverageMDP", "TotalRewardMDP", "FiniteHorizonMDP"]:
            raise Exception(f"Inconsistent model type '{self._type}'")
        
        initStates = self.getInitStates()
        print(f"{len(initStates)} initial states.")

        stateTemplate, stateVarTypes, stateVarInitValues = self._getStateVarInformation()

        # Explore all reachable states from initial states.
        stateToTupReprs, absorbingStates, transitons, actions = self.exploreStateSpace(initStates,
                                                                                       stateTemplate)
        print(f"{len(stateToTupReprs)} states, "
              f"{len(actions)} actions, "
              f"{len(transitons) + len(absorbingStates)} transitions")
        
        # Build and fill 'transitionDict', which is an intermediary structure used to facilitate access
        # to transition probabilities
        transitionDict = {
            action: {
                sTupRepr: dict() for sTupRepr in stateToTupReprs.values()
            } for action in actions
        }
        for (sTupRepr, sPrimeTupRepr, action), data in transitons.items():
            prob, reward = data
            transitionDict[action][sTupRepr][sPrimeTupRepr] = np.array([prob, reward])

        MDPData = {
            "name": self._name,
            "type": self._type,
            "criterion": self._criterion,
            "states": set(stateToTupReprs.values()),
            "initial-states": [ stateToTupReprs[s] for s in initStates ],
            "absorbing-states": { stateToTupReprs[s] for s in absorbingStates },
            "actions": actions,
            "transition-dict": transitionDict,
            "state-template": stateTemplate,
            "state-variable-types": stateVarTypes,
            "state-variable-initial-values": stateVarInitValues,
            "gamma": self._gamma,
            "horizon": self._horizon
        }
        return MDPData
    
    @override
    def getMCData(self):
        """Get all required and useful data to build a Marmote MC."""
        if self._type != "MarkovChain":
            raise Exception(f"Inconsistent model type '{self._type}'")
        
        initStates = self.getInitStates()
        print(f"{len(initStates)} initial states.")

        stateTemplate, stateVarTypes, stateVarInitValues = self._getStateVarInformation()

        # Explore all reachable states from initial states.
        stateToTupReprs, absorbingStates, transitons, actions = self.exploreStateSpace(initStates,
                                                                                       stateTemplate)
        assert len(actions) == 1
        print(f"{len(stateToTupReprs)} states, "
              f"{len(actions)} actions, "
              f"{len(transitons) + len(absorbingStates)} transitions")
        
        # Build and fill 'transitionDict', which is an intermediary structure used to facilitate access
        # to transition probabilities
        transitionDict = {
            sTupRepr: dict() for sTupRepr in stateToTupReprs.values()
        }
        for (sTupRepr, sPrimeTupRepr, _), data in transitons.items():
            prob, _ = data
            transitionDict[sTupRepr][sPrimeTupRepr] = prob

        MCData = {
            "name": self._name,
            "type": self._type,
            "states": set(stateToTupReprs.values()),
            "initial-states": [ stateToTupReprs[s] for s in initStates ],
            "absorbing-states": { stateToTupReprs[s] for s in absorbingStates },
            "actions": actions,
            "transition-dict": transitionDict,
            "state-template": stateTemplate,
            "state-variable-types": stateVarTypes,
            "state-variable-initial-values": stateVarInitValues,
        }
        return MCData
    
    @override
    def getPropertyNames(self):
        raise UnsupportedFeatureError(f"Propertices does not support by JaniR model '{self._name}'")
