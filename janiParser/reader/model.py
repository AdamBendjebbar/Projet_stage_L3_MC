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
    #A class represents a JANI (Json Automata Network Interface) model.
    def __init__(self, name, type):
        # Model identity: user-facing name and declared model kind (e.g., mdp, dtmc).
        self._name = name
        self._type = type

        # Action registry: action label -> integer index.
        self._actions = dict()

        # Running counter used to assign stable action indices.
        self._actionCounter = 0
        
        # Constant registry: constant name -> Constant object.
        self._constants = dict()

        # Transient variable registry: variable name -> transient Variable object.
        self._transientVars = dict()

        # Non-transient variable registry: variable name -> persistent Variable object.
        self._nonTransientVars = dict()

        # Function registry: function name -> Function object.
        self._functions = dict()

        # System-level automata index map: automaton name -> position in composition.
        self._automataIndices = None

        # Synchronization specification:
        # list of (resulting action, vector of pre-synchronization actions).
        self._preSyncActionss = None

        # Per-automaton parsed automata objects before synchronization:
        # index -> Automata.
        self._nonSyncAutomatas = None

        # Final synchronized automaton representing global behavior.
        self._automata = None

        # Property registry: property name -> Property object.
        self._properties = dict()

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    def addAction(self, action):
        #Add an action to the model.
        # Enforce unique action names in the model action registry.
        if action in self._actions:
            raise KeyError(f"Action '{action}' already exists in model '{self.name}'")
        # Assign the next available integer index to this action.
        self._actions[action] = self._actionCounter
        # Advance index counter for the next action insertion.
        self._actionCounter += 1

    def containsAction(self, action):
        """Return True if the model contains action, otherwise return False."""
        # Fast membership check in action-name -> index dictionary.
        return action in self._actions

    def addConstant(self, constant):
        """Add a constant variable to the model."""
        # Read constant identifier used as dictionary key.
        name = constant.name
        # Reject duplicate constant declarations.
        if name in self._constants:
            raise KeyError(f"Constant variable '{name}' already exists in model '{self.name}'")
        # Register constant object under its unique name.
        self._constants[name] = constant
    
    def isConstantVariable(self, name):
        #Return True if name is declared as a constant variable in the model.
        # Constant lookup by name in constant registry.
        return name in self._constants
    
    def getConstantValue(self, name):
       #Get the associated value of a constant variable.
        # Retrieve constant object if it exists.
        constant = self._constants.get(name)
        if constant is not None:
            # Return scalar/expression value stored in constant.
            return constant.value
        # Unknown constant names are treated as model-definition errors.
        raise KeyError(f"Unrecognized constant name '{name}' for model '{self.name}'")
    
    def addVariable(self, variable):
        #Add a variable to the model.
        # Extract variable identity and lifetime category.
        name = variable.name
        transient = variable.transient
        if transient:
            # Register transient variables in dedicated registry with uniqueness check.
            if name in self._transientVars:
                raise KeyError(f"Transient variable '{name}' already exists in model '{self.name}'")
            self._transientVars[name] = variable
        else:
            # Register non-transient variables in dedicated registry with uniqueness check.
            if name in self._nonTransientVars:
                    raise KeyError(f"Non-transient variable '{name}' already exists in model '{self.name}'")
            self._nonTransientVars[name] = variable
    
    def isGlobalVariable(self, name):
        #Return True if the variable name is declared as a constant or global variable.
        # Constants are globally visible by definition.
        if self.isConstantVariable(name):
            return True
        # Check transient variable scope if found.
        variable = self._transientVars.get(name)
        if variable is not None:
            return variable.isGlobal()
        # Check non-transient variable scope if found.
        variable = self._nonTransientVars.get(name)
        if variable is not None:
            return variable.isGlobal()
        # Unknown names are not global variables.
        return False

    def isTransientVariable(self, name):
        """Return True if name is declared as a transient variable."""
        # Membership check in transient variable registry.
        return name in self._transientVars
    
    def containsVariable(self, name):
        # Return True if the model contains variable name, otherwise return False.
        return name in self._constants or name in self._transientVars or name in self._nonTransientVars

    def declareFunction(self, function):
        # Declare a function in the model."""
        name = function.name
        if name in self._functions:
            raise KeyError(f"Function '{name}' already exists in model '{self.name}'")
        self._functions[name] = function

    def addFunctionBody(self, name, body):
        # Add a function body to a declared function.
        if name not in self._functions:
            raise KeyError(f"Undeclared function name '{name}' for model '{self.name}'")
        self._functions[name].addBody(body)
    
    def containsFunction(self, name):
        # Return True if the model contains function, otherwise return False.
        return name in self._functions
    
    def getFunction(self, name):
        # Return the function associated with the name.
        if not self.containsFunction(name):
            raise SyntaxError(f"Unrecognized function '{name}' for model '{self.name}'")
        return self._functions[name]

    def setSystemInformation(self, automataIndices, preSysnActionss):
        # Add system information to the model.
        if self._automataIndices is not None and self._preSyncActionss is not None:
            raise Exception("System information has already been defined and cannot be redefined")
        self._automataIndices = automataIndices
        self._preSyncActionss = preSysnActionss
        self._nonSyncAutomatas = [None] * len(self._automataIndices)

    def addAutomata(self, automata):
        # Add a automaton to the model.
        # System composition metadata must be defined before inserting automata.
        if self._automataIndices is None:
            raise Exception("System information not defined")
        # Resolve automaton name and validate it belongs to declared system automata.
        name = automata.name
        if name not in self._automataIndices:
            raise KeyError(f"Unrecognized automata '{name}' for model '{self.name}'")
        # Place parsed automaton at its fixed index used later by synchronization.
        idx = self._automataIndices[name]
        self._nonSyncAutomatas[idx] = automata

    def getInitStates(self):
        # Return all possible initial states.
        # Output list containing every concrete initial State instance.
        states = []
        # Access synchronized automaton location metadata.
        locs = self._automata.locations
        setOfLocation = self._automata.initLocation
        # Enumerate Cartesian product of all non-transient variable initial instantiations.
        for vars in product(*map(lambda x: x.instantiate(), self._nonTransientVars.values())):
            # Start from fresh transient variable copies for each candidate initial valuation.
            transientVars = deepcopy(self._transientVars)
            # Build non-transient variable map: name -> instantiated variable object.
            nonTransientVars = { var.name: var for var in vars }
            # Build valuation context used to evaluate location transient assignments.
            varGetter = { var.name: var.value for var in vars }
            for loc in setOfLocation:
                if loc not in locs:
                    continue
                # Apply location-level transient assignments at initialization.
                for ref, value in locs[loc].items():
                    transientVars[ref].setValueTo(value.eval(varGetter, self._functions))
            # Materialize one initial state candidate and append to result set.
            states.append(State(transientVars, nonTransientVars, setOfLocation))
        # Return full initial state set (possibly multiple due to variable domains).
        return states

    def synchronize(self):
        # Synchronize automata.
        if self._automata is not None:
            raise Exception("Synchronization is complete")
        print("Start synchronization")
        self._automata = self._synchronizeAutomata()
        print("Synchronization success")
        del self._nonSyncAutomatas
    
    def _synchronizeAutomata(self):
        # Fast path: if there is only one automaton, synchronization is unnecessary.
        if len(self._nonSyncAutomatas) == 1:
            return self._nonSyncAutomatas[0]
        
        # Containers for the synchronized/global automaton being built.
        syncActions = dict()
        syncEdges = list()
        syncInitLoc = set()
        syncLocs = dict()

        # Preprocess each local automaton:
        # - merge initial locations and location assignment maps
        # - index non-silent edges by action label
        # - keep silent edges as-is in the global edge list
        actionMapEdgesList = []
        for automata in self._nonSyncAutomatas:
            syncInitLoc.update(automata.initLocation)
            syncLocs.update(automata.locations)

            actionMapEdges = dict()
            for edge in automata.edges:
                action = edge.action
                if action == "silent-action":
                    # Silent edges are not synchronized; they are directly preserved.
                    syncEdges.append(edge)
                else:
                    # action -> list of candidate edges for that action in this automaton
                    actionMapEdges.setdefault(action, []).append(edge)
            # One action->edges map per automaton, used to build sync combinations.
            actionMapEdgesList.append(actionMapEdges)

        # Build synchronized actions and synchronized edges based on system spec.
        syncActionCounter = 0
        for result, preSyncActions in self._preSyncActionss:
            # Assign stable index to each resulting synchronized action label.
            syncActions[result] = syncActionCounter
            syncActionCounter += 1

            # Collect, per automaton, the edge list participating in this sync rule.
            preSyncEdgesList = []
            for i, preSyncAcition in enumerate(preSyncActions):
                if preSyncAcition is not None:
                    preSyncEdgesList.append(actionMapEdgesList[i][preSyncAcition])

            # Synchronize every Cartesian combination of participating local edges.
            for edgeComb in product(*preSyncEdgesList):
                syncEdges.append(self._synchronizeEdge(edgeComb, result))

        # Replace model action registry with synchronized actions only.
        self._actions = syncActions
        # Store number of synchronized actions.
        self._actionCounter = syncActionCounter
        # Return the final composed automaton.
        return Automata("Main", syncLocs, syncInitLoc, syncEdges)
    
    def _synchronizeEdge(self, edgeComb, action):
        # Aggregated source-location set of the synchronized edge.
        syncSrc = set()
        # Start with True; then conjunct all local guards.
        syncGuard = Expression("bool", True)

        # Collect destination-choices list from each local edge.
        preSyncEdgeDestsList = []
        for edge in edgeComb:
            # Union all local source locations.
            syncSrc.update(edge.source)
            # Global guard is conjunction of participating local guards.
            syncGuard = Expression.reduceExpression("∧", syncGuard, edge.guard)
            # Keep each edge destination list for Cartesian combination later.
            preSyncEdgeDestsList.append(edge.edgeDestinations)
        
        # Build synchronized destinations from every combination of local destinations.
        syncEdgeDests = [
            self._synchronizeEdgeDestination(edgeDestComb) for edgeDestComb in product(*preSyncEdgeDestsList)
        ]
        # Return one synchronized edge with merged source, guard and destinations.
        return Edge(syncSrc, action, syncGuard, syncEdgeDests)
    
    def _synchronizeEdgeDestination(self, edgeDestComb):
        # Aggregated destination-location set for one synchronized destination.
        syncDest = set()
        # Start with neutral probability 1, then multiply local probabilities.
        syncProb = Expression("real", 1.)
        # Combined assignments map (later updates may overwrite same key).
        syncAssngs = dict()

        for edgeDest in edgeDestComb:
            # Union all local destination locations.
            syncDest.update(edgeDest.destination)
            # Multiply local probabilities to get joint probability.
            syncProb = Expression.reduceExpression("*", syncProb, edgeDest.probability)
            # Merge local assignments into synchronized assignment map.
            syncAssngs.update(edgeDest.assignments)
        # Return synchronized edge destination.
        return EdgeDestination(syncDest, syncProb, syncAssngs)

    def addProperty(self, property):
        # Add a property to the model.
        name = property.name
        if name in self._properties:
            raise KeyError(f"Property '{name}' already exists in model '{self.name}'")
        self._properties[name] = property

    def getPropertyNames(self):
        # Return all property names.
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
        # Set of already-expanded State objects (prevents re-expanding same semantic state).
        visitedStates = set()
        # Transition accumulator:
        # (source_tuple, target_tuple, action_label) -> np.array([cumulated_probability, cumulated_reward]).
        transitions = dict()
        # States classified as absorbing (terminal by property or deadlock by dynamics).
        absorbingStates = set()
        # Maximum number of per-state silent-action variants encountered (for final action indexing).
        maxSilentActCnt = 0

        # Cache mapping each discovered State object to its canonical tuple representation.
        stateToTupReprs = dict()
        # Frontier container (LIFO with pop(), so exploration order is depth-first-like).
        queue = deque()
        for initState in initStates:
            # Pre-register tuple form of each initial state for stable indexing downstream.
            stateToTupReprs[initState] = initState.getTupRepr(stateTemplate)
            # Seed exploration frontier with initial states.
            queue.append(initState)
        
        # Global automaton transition relation (edge schema level).
        edges = self._automata.edges
        # Location update dictionary used when materializing successor states.
        locs = self._automata.locations
        # Function environment used by guards/probabilities/assignments expression evaluation.
        funcGetter = self._functions
        while queue:
            # Pop one candidate state from frontier.
            s = queue.pop()
            if s not in visitedStates:
                # Canonical tuple key for current concrete state.
                sTupRepr = s.getTupRepr(stateTemplate)
                # Mark this state as expanded.
                visitedStates.add(s)

                # Property-driven terminal cut: terminal states are made absorbing immediately.
                if terminalStateExpr.eval(s):
                    absorbingStates.add(s)
                    continue

                # Lightweight progress output (number of expanded states).
                print(len(visitedStates), end="\r")

                # Assume deadlock until at least one enabled edge is found.
                deadlock = True
                # Counter to disambiguate multiple silent transitions leaving same state.
                silentActCnt = 0
                for edge in edges:
                    # Enablement test combines location membership + guard evaluation on current state.
                    if not edge.isSatisfied(s.setOfLocation, s, funcGetter):
                        continue

                    # At least one enabled edge exists => current state is not deadlock.
                    deadlock = False
                    src = edge.source
                    act = edge.action
                    # MC-style normalization: collapse all actions into one canonical label.
                    if singleActRequirement:
                        act = "act"
                    # For MDP-like export, silent edges are renamed per-state to keep actions distinct.
                    elif act == "silent-action":
                        act = f"{act}_{silentActCnt}"
                        silentActCnt += 1
                    for edgeDest in edge.edgeDestinations:
                        # --- Automata/edge semantics -> concrete successor state materialization ---
                        # Build s' from current state s by applying one edge destination:
                        # 1) location evolution: remove edge.source locations, add destination locations,
                        # 2) variable evolution: apply destination assignments,
                        # 3) location transient effects: apply location-based transient updates via `locs`.
                        # Result: sPrime contains updated non-transient vars, transient vars, and setOfLocation.
                        sPrime = s.clone(src,
                                         edgeDest.destination,
                                         edgeDest.assignments,
                                         locs,
                                         funcGetter)
                        # Transition probability is evaluated from edge destination expression on source state s.
                        prob = edgeDest.probability.eval(s, funcGetter)
                        # Reward for this transition is evaluated on successor state s'.
                        reward = rewardExpr.eval(sPrime)

                        # Ignore null/negative-probability branches.
                        if prob <= 0.:
                            continue

                        # Reuse existing tuple encoding for s' if already discovered.
                        sPrimeTupRepr = stateToTupReprs.get(sPrime)

                        if sPrimeTupRepr is None:
                            stateToTupReprs[sPrime] = sPrimeTupRepr = sPrime.getTupRepr(stateTemplate)
                        # Aggregate multiple semantic paths landing on same (s, s', a).
                        key = (sTupRepr, sPrimeTupRepr, act)
                        transitions[key] = transitions.get(key, np.array([0., 0.])) + [prob, reward]
                        # Enqueue unexplored successor for later expansion.
                        if sPrime not in visitedStates:
                            queue.append(sPrime)
                        else:
                            del sPrime
                # If no enabled edge was found, state behaves as absorbing.
                if deadlock:
                    absorbingStates.add(s)
                # Track maximum silent action fan-out over all expanded states.
                maxSilentActCnt = max(maxSilentActCnt, silentActCnt)
            else:
                del s
        # Final action dictionary normalization.
        if singleActRequirement:
            # MC export path: single canonical action.
            actions = { "act": 0 }
        else:
            # MDP export path: keep model actions and append generated silent-action variants.
            actions = deepcopy(self._actions)
            actions.update({ f"silent-action_{i}": i + self._actionCounter for i in range(maxSilentActCnt) })
        # Return full explored graph artifacts for downstream MC/MDP data builders.
        return stateToTupReprs, absorbingStates, transitions, actions
        
    def _getStateVarInformation(self):
        
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
        # This exporter is only valid for DTMC models in the base JaniModel class.
        if self._type != "dtmc":
            raise Exception(f"Inconsistent model type '{self._type}'")

        # Enumerate all concrete initial states generated from model initialization rules.
        initStates = self.getInitStates()
        print(f"{len(initStates)} initial states.")

        # Build the fixed state encoding metadata (template, types, and initial values).
        stateTemplate, stateVarTypes, stateVarInitValues = self._getStateVarInformation()

        # Explore the reachable state space from initial states.
        # For MC export:
        # - terminal expression is always False (no custom terminal cut)
        # - reward is constant 0 (not used in pure MC transition matrix)
        # - singleActRequirement=True forces one canonical action label ("act")
        stateToTupReprs, absorbingStates, transitons, actions = self.exploreStateSpace(initStates,
                                                                                       stateTemplate,
                                                                                       Expression("bool", False),
                                                                                       Expression("int", 0),
                                                                                       singleActRequirement=True)
        # Markov chains must expose exactly one abstract action in this representation.
        assert len(actions) == 1
        print(f"{len(stateToTupReprs)} states, "
              f"{len(actions)} actions, "
              f"{len(transitons) + len(absorbingStates)} transitions")
        
        # Build transition dictionary in MC format:
        # source_state_tuple -> { target_state_tuple -> probability }
        transitionDict = {
            sTupRepr: dict() for sTupRepr in stateToTupReprs.values()
        }
        # Ignore action/reward dimensions and keep only transition probabilities.
        for (sTupRepr, sPrimeTupRepr, _), data in transitons.items():
            prob, _ = data
            transitionDict[sTupRepr][sPrimeTupRepr] = prob

        # Package all MC data required by downstream Marmote tooling.
        MCData = {
            # Model identifier used in exported MC metadata.
            "name": self._name,
            # Normalized export type expected by downstream MC consumers.
            "type": "MarkovChain",
            # Set of all reachable states encoded as tuple representations.
            "states": set(stateToTupReprs.values()),
            # Initial state set (as tuples) obtained from model initialization.
            "initial-states": [ stateToTupReprs[s] for s in initStates ],
            # States with no enabled outgoing transition under exploration semantics.
            "absorbing-states": { stateToTupReprs[s] for s in absorbingStates },
            # Action registry (single canonical action for MC export path).
            "actions": actions,
            # Sparse transition map: source_tuple -> { target_tuple -> probability }.
            "transition-dict": transitionDict,
            # Canonical variable/location ordering used to build state tuples.
            "state-template": stateTemplate,
            # Type information for each state variable in the exported representation.
            "state-variable-types": stateVarTypes,
            # Declared initial value metadata for each exported state variable.
            "state-variable-initial-values": stateVarInitValues,
            # Total number of transitions counted (explicit transitions + absorbing self-cases).
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
