from __future__ import annotations

import json


from janiParser.reader.variable import Type
from janiParser.exception import *

import marmote.core as mc
import marmote.markovchain as mmc
import marmote.mdp as mmdp

import numpy as np
import scipy.sparse as ss

_penality = -1e32

class DataMarmote:
    def __init__(self, data):
        if not isinstance(data, dict):
            raise TypeError("The given parameter 'data' must be of type 'dict'")
        for attr in ["name", "type", "states", "actions",
                     "transition-dict", "absorbing-states", "state-template", "state-variable-types"]:
            if attr not in data:
                raise Exception()
        
        self._name = data["name"]
        self._type = data["type"]
         #contient l'ensmebl des etats sous forme de tuple
        self._states = data["states"]
         #contient l'ensemble des etats absorbants sous forme de tuple
        self._absorbingStates = data["absorbing-states"]
        self._actions: dict = data["actions"]
         #contient un dictionnaire imbrique de la forme tuple (otigine):tuples(destinations):probabilités
        self._transitionDict = data["transition-dict"]
        self._stateTemplate = data["state-template"]
        self._stateVarTypes = data["state-variable-types"]

        self._criterion = data.get("criterion")
        self._horizon = data.get("horizon")
        self._gamma = data.get("gamma")

        if self._horizon is None:
            self._horizon = 1
        if self._gamma is None:
            self._gamma = .95
         #contient les tuples des etast initiaux
        self._initStates = data.get("initial-states")
        self._stateVarInitValues = data.get("state-variable-initial-values")
         # on cree un dictionnaire avec comme clé le tuple et un index 
        self._stateTupReprToIdx = { stateTupRepr: idx for idx, stateTupRepr in enumerate(sorted(self._states)) }
         # on inverse la clé avce la valeur et la valeur avec la clé
        self._idxToStateTupRepr = { idx: stateTupRepr for stateTupRepr, idx in self._stateTupReprToIdx.items() }

        self._stateSpace = data.get("state-space")
        self._actionSpace = data.get("action-space")
        self._transitionMatrices = data.get("transition-matrices")
        self._rewardMatrices = data.get("reward-matrices")

        self._isInstantiate = self._stateSpace is not None and \
                              self._actionSpace is not None and \
                              self._transitionMatrices is not None and \
                              self._rewardMatrices is not None

        self._validate()

    def _validate(self):
        if self._type not in ["dtmc", "mdp", "DiscountedMDP", "AverageMDP", "TotalRewardMDP", "FiniteHorizonMDP", "MarkovChain"]:
            raise UnsupportedFeatureError(f"Unsupported type '{self._type}'")

        if self._type == "DiscountedMDP":
            if self._gamma is None:
                raise ValueError("Discounted MDP must define a discounted factor.")
            if self._gamma < 0 or self._gamma >= 1:
                raise ValueError("Discounted factor must have a value between 0 and 1.")

        if self._type == "FiniteHorizonMDP":
            if self._gamma is None:
                raise ValueError("FiniteHorizon MDP must define a discounted factor.")
            if self._gamma < 0 or self._gamma >= 1:
                raise ValueError("Discounted factor must have a value between 0 and 1.")
            if self._horizon is None:
                raise ValueError("FiniteHorizon MDP must define an horizon.")
    
    def getInitStatesIdx(self):
        return [ self.stateToIdx(s) for s in self._initStates ]
    
    def stateToIdx(self, state):
        return self._stateTupReprToIdx.get(state)
    
    def idxToState(self, idx):
        return self._idxToStateTupRepr.get(idx)
    
    @staticmethod
    def fromMarmoteMDP(criterion,
                       type,
                       stateSpace: "marmote.core.MarmoteBox",
                       actionSpace,
                       transitionMatrices: list[mc.SparseMatrix],
                       rewardMatrices,
                       gamma=None,
                       horizon=None):
        """"""
        dims = stateSpace.tot_nb_dims()
        nbStates = stateSpace.Cardinal()
        nbActions = actionSpace.Cardinal()

        states = { tuple(stateSpace.DecodeState(idx)) for idx in range(nbStates)}
        actions = { f"act{idx}": idx for idx in range(nbActions) }

        stateTemplate = { f"var{idx}": idx for idx in range(dims) }
        varNames = list(stateTemplate.keys())
        stateVarTypes = { var: Type("int", (0, stateSpace.CardinalbyDim(idx) - 1)) for idx, var in enumerate(varNames) }
        transitionDict = {
            act: {
                sTupRepr: dict() for sTupRepr in states
            } for act in actions
        }

        isStateActReward = isinstance(rewardMatrices, (mc.SparseMatrix, mc.FullMatrix))
        isSingleTransition = isinstance(transitionMatrices, mc.SparseMatrix)
        for act, actIdx in actions.items():
            transMatrix = transitionMatrices if isSingleTransition else transitionMatrices[actIdx]
            rewardMatrix = rewardMatrices if isStateActReward else rewardMatrices[actIdx]
            for sIdx in range(nbStates):
                s = stateSpace.DecodeState(sIdx)
                for sPrimeIdx in range(nbStates):
                    sPrime = stateSpace.DecodeState(sPrimeIdx)
                    prob = transMatrix.getEntry(sIdx, sPrimeIdx)
                    if prob <= 0.:
                        continue
                    
                    if isStateActReward:
                        reward = rewardMatrix.getEntry(sIdx, actIdx)
                    else:
                        reward = rewardMatrix.getEntry(sIdx, sPrimeIdx)
                    transitionDict[act][tuple(s)][tuple(sPrime)] = [prob, reward]
        # print(transitionDict)
        MDPData = {
            "name": "MDP",
            "type": type,
            "criterion": criterion,
            "states": states,
            "absorbing-states": set(),
            "actions": actions,
            "transition-dict": transitionDict,
            "state-template": stateTemplate,
            "state-variable-types": stateVarTypes,
            "gamma": gamma,
            "horizon": horizon
        }
        return DataMarmote(MDPData)

    def createMDPObject(self, discount, horizonFini):
        """Create an associated Marmote MDP."""
        if not self._isInstantiate:
            n, m = len(self._states), len(self._actions)
            stateSpace = mc.MarmoteInterval(0, n - 1)
            actionSpace = mc.MarmoteInterval(0, m - 1)
            
            transtionDict = self._transitionDict
            absorbingStates = self._absorbingStates
            stateTupReprToIdx = self._stateTupReprToIdx

            penality = _penality if self.isMaximisation() else -_penality

            transitionMatrices = [mc.SparseMatrix(n) for _ in range(m)]
            rewardMatrices = [mc.SparseMatrix(n) for _ in range(m)]
            for act, actIdx in self._actions.items():
                transitionMatrix = transitionMatrices[actIdx]
                rewardMatrix = rewardMatrices[actIdx]

                for sTupRepr, sPrimeMap in transtionDict[act].items():
                    sIdx = stateTupReprToIdx[sTupRepr]
                    row_sum = 0.0

                    if sPrimeMap:
                        for sPrimeTupRepr, data in sPrimeMap.items():
                            prob, reward = data

                            sPrimeIdx = stateTupReprToIdx[sPrimeTupRepr]
                            transitionMatrix.addEntry(sIdx, sPrimeIdx, prob)
                            rewardMatrix.addEntry(sIdx, sPrimeIdx, reward)
                            row_sum += prob

                        if abs(row_sum - 1.0) > 1e-8:
                            if row_sum < 1.0:
                                missing_prob = 1.0 - row_sum
                                transitionMatrix.addEntry(sIdx, sIdx, missing_prob)
                                # rewardMatrix.addEntry(sIdx, sIdx, 0.0)
                            else:
                                raise Exception(
                                    f"Invalid transition probabilities for state {sTupRepr}, action {act}: sum = {row_sum}")
                    else:
                        transitionMatrix.addEntry(sIdx, sIdx, 1.)
                        if sTupRepr not in absorbingStates:
                            rewardMatrix.addEntry(sIdx, sIdx, penality)

            print("Build success - Transition and Reward matrices")
            self._stateSpace = stateSpace
            self._actionSpace = actionSpace
            self._transitionMatrices = transitionMatrices
            self._rewardMatrices = rewardMatrices
            self._isInstantiate = True

        MDPType = self._type
        if discount:
            MDPType = "DiscountedMDP"
        if horizonFini:
            MDPType = "FiniteHorizonMDP"
        if MDPType == "DiscountedMDP":
            return mmdp.DiscountedMDP(self._criterion,
                                      self._stateSpace,
                                      self._actionSpace,
                                      self._transitionMatrices,
                                      self._rewardMatrices,
                                      self._gamma)
        if MDPType == "AverageMDP":
            return mmdp.AverageMDP(self._criterion,
                                   self._stateSpace,
                                   self._actionSpace,
                                   self._transitionMatrices,
                                   self._rewardMatrices)
        if MDPType == "TotalRewardMDP":
            return mmdp.TotalRewardMDP(self._criterion,
                                       self._stateSpace,
                                       self._actionSpace,
                                       self._transitionMatrices,
                                       self._rewardMatrices)
        if MDPType == "FiniteHorizonMDP":
            return mmdp.FiniteHorizonMDP(self._criterion,
                                         self._stateSpace,
                                         self._actionSpace,
                                         self._transitionMatrices,
                                         self._rewardMatrices,
                                         self._horizon,
                                         self._gamma)

   #def createMCObject(self):
        #transitionDict = self._transitionDict
        #n = len(self._states)
        #stateTupReprToIdx = self._stateTupReprToIdx
        #transitionMatrix = mc.SparseMatrix(n)

        #for sTupRepr, sPrimeMap in transitionDict.items():
           # sIdx = stateTupReprToIdx[sTupRepr]
            #row_sum = 0
           # for sPrimeTupRepr, data in sPrimeMap.items():
                #sPrimeIdx = stateTupReprToIdx[sPrimeTupRepr]
                #transitionMatrix.addEntry(sIdx, sPrimeIdx, data)
                #row_sum += data
            #if abs(row_sum - 1.0) > 1e-8:
                #if row_sum < 1.0:
                #    missing_prob = 1.0 - row_sum
                #    transitionMatrix.addEntry(sIdx, sIdx, missing_prob)
                #else:
                #    raise Exception(f"Invalid transition probabilities for state {sTupRepr}: sum = {row_sum}")

        #transitionMatrix.set_type(mc.DISCRETE)
        #self._transitionMatrices = transitionMatrix
        #return mmc.MarkovChain(transitionMatrix)

    def createMCObject(self, init_mode="uniform"):
   
        import random # Nécessaire pour le mode random
        
        # 1. Préparation des dimensions
        n = len(self._states)
        stateTupReprToIdx = self._stateTupReprToIdx
        
        # 2. Création de l'Espace d'États
        stateSpace = mc.MarmoteInterval(int(0), int(n - 1))
        # 3. Construction de la Structure de Transition (inchangé)
        transitionMatrix = mc.SparseMatrix(stateSpace)
        transitionDict = self._transitionDict

        for sTupRepr, sPrimeMap in transitionDict.items():
            sIdx = stateTupReprToIdx[sTupRepr]
            row_sum = 0
            for sPrimeTupRepr, data in sPrimeMap.items():
                sPrimeIdx = stateTupReprToIdx[sPrimeTupRepr]
                transitionMatrix.addEntry(sIdx, sPrimeIdx, data)
                row_sum += data
            
            if abs(row_sum - 1.0) > 1e-8:
                if row_sum < 1.0:
                    missing_prob = 1.0 - row_sum
                    transitionMatrix.addEntry(sIdx, sIdx, missing_prob)
                else:
                    raise Exception(f"Somme des probabilités invalide : {row_sum}")

        transitionMatrix.set_type(mc.DISCRETE)

        # 4. Création de la Distribution Initiale selon le MODE
        init_probas = [0.0] * n

        init_indices = self.getInitStatesIdx() # Les vrais index du JANI

        if init_mode == "first":
            idx = init_indices[0] if init_indices else 0
            init_probas[idx] = 1.0

        elif init_mode == "uniform":
            if init_indices:
                prob = 1.0 / len(init_indices)
                for idx in init_indices:
                    init_probas[idx] = prob
            else:
                init_probas[0] = 1.0

        elif init_mode == "random":
            if init_indices:
                # Aléatoire UNIQUEMENT sur les vrais états initiaux
                weights = [random.random() for _ in range(len(init_indices))]
                total = sum(weights)
                for i, idx in enumerate(init_indices):
                    init_probas[idx] = weights[i] / total
            else:
                init_probas[0] = 1.0
                
        elif init_mode == "rangeall":
            # Distribution pondérée (donne plus de poids aux premiers index)
            # On crée une suite décroissante : n, n-1, n-2...
            weights = [float(n - i) for i in range(n)]
            total = sum(weights)
            init_probas = [w / total for w in weights]

        # Création de l'objet DiscreteDistribution
        init_dist = mc.DiscreteDistribution(stateSpace, init_probas)

        # Après création de init_dist
        self._initDistribution = init_dist

        # 5. Assemblage final
        m_chain = mmc.MarkovChain(transitionMatrix)
        m_chain.set_init_distribution(init_dist)

        self._transitionMatrices = transitionMatrix
        self._stateSpace = stateSpace
        
        return m_chain

    def createMarmoteObject(self, discount=False, horizonFini=False, init_mode="uniform"):
        """Create an associated Marmote instance."""
        if self._type in ["dtmc", "MarkovChain"]:
            # On passe le mode à la fonction spécifique
            return self.createMCObject(init_mode=init_mode)
        return self.createMDPObject(discount, horizonFini)
    
    def getInitDistribution(self):
        return getattr(self, "_initDistribution", None)

    def buildTransitionRewardForMDPToolbox(self) -> tuple[list[ss.csc_matrix], np.ndarray]:
        """Build the transition and reward matrices compatible with MDPToolbox.
        
        Returns:
            out (tuple):
                * A list of csr sparse transition matrix: [action, current state, next state].
                * A 2D numpy array which represents the reward matrix: [state, action].
        """
        # n: the cardinality of the state space.
        n = len(self._states)

        # m: the cardinality of the action space.
        m = len(self._actions)

        # Save theses variables as local variables to slightly accelerate memory access.
        # Format: { action: { current state: { next state: [ probability, reward ] } } }
        transitionDict: dict[str, dict[tuple, dict[tuple, np.ndarray]]] = self._transitionDict

        # A set of absorbing states defined in the model.
        absorbingStates: set[tuple] = self._absorbingStates

        # A dictionary that maps a unique index to each state.
        stateTupReprToIdx: dict[tuple, int] = self._stateTupReprToIdx

        isMaximisation = self.isMaximisation()

        # Penality applied when an unauthorized action is performed in a state. 
        penality = _penality

        transitionMatrices = []
        rewardMatrix = np.zeros((n, m), dtype=np.float64)
        for act, actIdx in self._actions.items():
            transitionMatrix = ss.lil_matrix((n, n), dtype=np.float64)

            for sTupRepr, sPrimeMap in transitionDict[act].items():
                sIdx = stateTupReprToIdx[sTupRepr]

                rewardSum, probSum = 0., 0.
                # If the state has defined transitions.
                if sPrimeMap:
                    for sPrimeTupRepr, data in sPrimeMap.items():
                        prob, reward = data
                        sPrimeIdx = stateTupReprToIdx[sPrimeTupRepr]

                        # Assign the transition probability.
                        transitionMatrix[sIdx, sPrimeIdx] = prob

                        # Accumulate expected reward and probability sum.
                        rewardSum += (prob * reward)
                        probSum += prob
                    
                    # Check if the sum of transition probabilities is valid.
                    if abs(probSum - 1.) > 1e-10:
                        if 0. < probSum < 1.:
                            transitionMatrix[sIdx, sIdx] = 1. - probSum
                        else:
                            raise ValueError(f"Invalid transition probabilities for state {sTupRepr} and action {act}: total = {probSum}")
                    
                    rewardMatrix[sIdx, actIdx] = rewardSum if isMaximisation else -rewardSum
                else:
                    transitionMatrix[sIdx, sIdx] = 1.

                    # If the current state is not in set of absorbing states,
                    # then the current state is not authorized to perform the 'act' action.
                    if sTupRepr not in absorbingStates:
                        rewardMatrix[sIdx, actIdx] = penality
            transitionMatrices.append(transitionMatrix.tocsr())
        print("Build success - Transition and Reward matrices")
        return transitionMatrices, rewardMatrix

    @staticmethod
    def _build_expression(variables, values):
        def recurse(items):
            if len(items) == 1:
                var, val = items[0]
                return {"op": "=", "left": var, "right": int(val)}
            else:
                mid = len(items) // 2
                left_expr = recurse(items[:mid])
                right_expr = recurse(items[mid:])
                return {
                    "op": "∧",
                    "left": left_expr,
                    "right": right_expr
                }

        variable_value_pairs = list(zip(variables, values))
        expression = {
            "exp": recurse(variable_value_pairs)
        }
        return expression

    def _createJaniRModelStruct(self):
        stateTemplate = self._stateTemplate
        stateVarTypes = self._stateVarTypes
        stateVarInitValues = self._stateVarInitValues
        transtionDict = self._transitionDict

        modelStruct = dict()
        modelStruct["name"] = self._name

        isMCModel = self._type == "MarkovChain"
        modelStruct["type"] = self._type
        
        if self._criterion is not None:
            modelStruct["criterion"] = self._criterion
        
        if self._gamma is not None:
            modelStruct["gamma"] = self._gamma

        if self._horizon is not None:
            modelStruct["horizon"] = self._horizon


        vars = [None] * len(stateTemplate)
        for var, idx in stateTemplate.items():
            item = dict()
            item["name"] = f"var{idx}"
            type = stateVarTypes[var]
            if type.hasBounds():
                lowerBound, upperBound = type.bounds
                item["type"] = {
                    "kind": "bounded",
                    "base": type.type,
                    "lower-bound": lowerBound,
                    "upper-bound": upperBound
                }
            else:
                item["type"] = type.type
            if stateVarInitValues is not None and stateVarInitValues[var] is not None:
                item["initial-value"] = stateVarInitValues[var]
            vars[idx] = item
        modelStruct["variables"] = vars


        actions = [None] * len(self._actions)
        for idx in self._actions.values():
            actions[idx] = { "name": f"act{idx}" }
        modelStruct["actions"] = actions


        automata = dict()
        automata["name"] = "main-automata"
        automata["locations"] = [{ "name": "loc" }]
        automata["initial-locations"] = ["loc"]
        edges = list()
        nbVars = len(vars)
        varNames = [f"var{idx}" for idx in range(nbVars)]


        for act, idx in self._actions.items():
            action = f"act{idx}"
            items = transtionDict.items() if isMCModel else transtionDict[act].items()
            for sTupRepr, sPrimeMap in items:
                destinations = list()
                if not sPrimeMap:
                    continue
                for sPrimeTupRepr, data in sPrimeMap.items():
                    dest = {
                        "location": "loc",
                        "assignments": [
                            {
                                "ref": varNames[idx],
                                "value": int(sPrimeTupRepr[idx])
                            } for idx in range(nbVars)
                        ]
                    }
                    if isMCModel:
                        dest["probability"] = { "exp": data }
                    else:
                        prob, reward = data
                        dest["probability"] = { "exp": prob }
                        dest["reward"] = { "exp": reward }
                    destinations.append(dest)
                item = {
                    "location": "loc",
                    "action": action,
                    "guard": self._build_expression(varNames, sTupRepr),
                    "destinations": destinations
                }
                edges.append(item)
        automata["edges"] = edges
        modelStruct["automata"] = [automata]


        modelStruct["system"] = {
            "elements": [
                { "automaton": "main-automata" }
            ]
        }

        return modelStruct

    def saveAsJaniRFile(self, filename):
        modelStruct = self._createJaniRModelStruct()
        print("Build success - JaniR model")
        with open(filename, "w", encoding="utf-8-sig") as file:
            file.write(json.dumps(modelStruct, indent=4, ensure_ascii=False))
        print(f"Save success - saved file '{filename}'")

    def nbStates(self) -> int:
        """Return the number of states."""
        return len(self._states)
    
    def nbActions(self) -> int:
        """Return the number of actions."""
        return len(self._actions)
    
    def setMDPTypeTo(self, mdpType: str, discount: float=.95, horizon: int=1) -> None:
        """Set MDP model type and associated parameters.
        
        Parameters:
            mdpType (str): The MDP model type.
            
            discount (float): The discount facotor used in 'DiscountedMDP' and 'FiniteHorizonMDP'.

            horizon (int): The horizon length used in 'FiniteHorizonMDP'.
        """
        # Check input argument: 'mdpType'.
        if self._type == "MarkovChain":
            raise Exception("Current model is not a MDP model")
        if mdpType not in ["DiscountedMDP", "AverageMDP", "TotalRewardMDP", "FiniteHorizonMDP"]:
            raise ValueError(f"Unsupported MDP model type: {mdpType}")
        
        # Check input argument: 'discount'.
        if not (0 < discount <= 1):
            raise ValueError("Discount factor must be a float in the range (0, 1]")
        
        # Check input argument: "horizon"
        if horizon < 0:
            raise ValueError("Horizon length must be a non-negative integer")
        
        self._type = mdpType
        self._gamma = discount
        self._horizon = horizon

    def isMaximisation(self) -> bool:
        """Return True if the optimization criterion is 'max', otherwise False."""
        return self._criterion == "max"