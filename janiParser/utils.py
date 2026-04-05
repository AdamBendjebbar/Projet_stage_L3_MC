from janiParser.reader.reader import JaniReader
from janiParser.dataMarmote import DataMarmote

import pickle
import os
from time import time as _timer

import marmote.mdp as mmdp
import mdptoolbox

def saveMDPModelsFromQComp(fullModelName: str, modelParams: dict={}, replace: bool=False) -> list[str]:
    """Download the specified `.jani` file available online at `https://qcomp.org/benchmarks/`,
    buids and saves one or more DataMarmote objects as `.pkl` files.  
    Each property defined in the target file results in a distinct `.pkl` file containing
    the corresponding DataMarmote object.

    Parameters:
        fullModelName (str): Full file name of the JANI model.
        
        modelParams (dict): Optional parameter used to configure a scalable model. It defaults to an empty dictionary.

        replace (bool): If True, all corresponding `.pkl` files will be overwritten.

    Returns:
        out (list[str]): List of paths to the saved `.pkl` files (one per property).

    Examples:
    >>> import janiParser
    >>> janiParser.saveMDPModelsFromQComp("consensus.2.jani", modelParams={ "K": 10 }, replace=True)
    ['pklFiles/consensus/consensus.2.c1.pkl', 'pklFiles/consensus/consensus.2.c2.pkl', 'pklFiles/consensus/consensus.2.disagree.pkl', 'pklFiles/consensus/consensus.2.steps_max.pkl', 'pklFiles/consensus/consensus.2.steps_min.pkl']
    """
    modelClassName = fullModelName.split(".")[0]
    outputDir = f"pklFiles/{modelClassName}"
    os.makedirs(outputDir, exist_ok=True)

    url = f"https://qcomp.org/benchmarks/mdp/{modelClassName}/{fullModelName}"
    model = JaniReader(url, modelParams=modelParams, isLocalPath=False).build()

    savedPaths = []
    for prop in model.getPropertyNames():
        savedName = fullModelName.replace(".jani", f".{prop}.pkl")
        savedPath = f"{outputDir}/{savedName}"

        savedPaths.append(savedPath)
        if os.path.exists(savedPath) and not replace:
            continue

        mdpData = model.getMDPData(prop)
        dataMarmote = DataMarmote(mdpData)

        with open(savedPath, "wb") as file:
            pickle.dump(dataMarmote, file, pickle.HIGHEST_PROTOCOL)
        del mdpData
        del dataMarmote
    return savedPaths

def loadDataMarmoteFromPklFile(path: str) -> DataMarmote:
    """Load a DataMarmote object from a `.pkl` file.
    
    Parameters:
        path (str): Path to the target `.pkl` file.
    
    Returns:
        out (DataMarmote): DataMarmote instance.
    """
    if not os.path.exists(path):
        raise Exception(f"Given path: {path} does not exist")
    with open(path, "rb") as file:
        dataMarmote = pickle.load(file)
    return dataMarmote


def _benchmarkMarmoteSolver(obj: mmdp.DiscountedMDP | mmdp.TotalRewardMDP | mmdp.AverageMDP,
                            resolMethodName: str,
                            reps: int,
                            _,
                            epsilon: float,
                            maxIter: int,
                            maxInIter: int) -> float:
    """Benchmark a specified resolution method using the `Marmote` library.
    
    Parameters:
        obj: Marmote instance
    
        resolMethods (str): Name of the resolution method.

        reps (int): Number of repetitions.

        epsilon (float): Convergence threshold.

        maxIter (int): Number of maximum iterations.

        maxInIter (int): Number of maximum inner iterations (for modified policy methods).
    
    Returns:
        out (float): Average runtime (in seconds).
    """
    supportedMethodMap = {
        "ValueIteration": lambda: obj.ValueIteration(epsilon, maxIter),
        "RelativeValueIteration": lambda: obj.RelativeValueIteration(epsilon, maxIter),
        "ValueIterationGS": lambda: obj.ValueIterationGS(epsilon, maxIter),

        # We use the same convergence threshold: epsilon for the inner loop.
        "PolicyIterationModified": lambda: obj.PolicyIterationModified(epsilon, maxIter, epsilon, maxInIter),
        "PolicyIterationModifiedGS": lambda: obj.PolicyIterationModifiedGS(epsilon, maxIter, epsilon, maxInIter)
    }
    
    # Determine the resolution method.
    if resolMethodName not in supportedMethodMap:
        raise ValueError(f"Marmote library class: {obj.className()} does not support resolution method: {resolMethodName}")
    method = supportedMethodMap[resolMethodName]

    # Benchmarking.
    totalTime = 0.
    for _ in range(reps):
        startTime = _timer()
        method()
        totalTime += _timer() - startTime
    return totalTime / reps

def _benchmarkMDPToolboxSolver(obj: tuple,
                               resolMethodName: str,
                               reps: int,
                               discount: float,
                               epsilon: float,
                               maxIter: int,
                               maxInIter: int) -> float:
    """Benchmark a specified resolution method using the `MDPToolbox` library.
    
    Parameters:
        obj: Tuple of the transitions and reward matrices.
    
        resolMethods (str): Name of the resolution method.

        reps (int): Number of repetitions.

        discount (float): Discount factor.

        epsilon (float): Convergence threshold.

        maxIter (int): Number of maximum iterations.

        maxInIter (int): Number of maximum inner iterations (for modified policy methods).
    
    Returns:
        out (float): Average runtime (in seconds).
    """
    transitions, reward = obj
    supportedMethodConstructorMap = {
        "ValueIteration": lambda: mdptoolbox.mdp.ValueIteration(transitions, reward, discount, epsilon=epsilon, max_iter=maxIter),
        "RelativeValueIteration": lambda: mdptoolbox.mdp.RelativeValueIteration(transitions, reward, epsilon=epsilon, max_iter=maxIter),
        "ValueIterationGS": lambda: mdptoolbox.mdp.ValueIterationGS(transitions, reward, discount, epsilon=epsilon, max_iter=maxIter),
        "PolicyIteration": lambda: mdptoolbox.mdp.PolicyIteration(transitions, reward, discount, max_iter=maxIter),
        "PolicyIterationModified": lambda: mdptoolbox.mdp.PolicyIterationModified(transitions, reward, discount, epsilon=epsilon, max_iter=maxInIter)
    }
    
    # Determine the constructor function of the specified resolution method.
    if resolMethodName not in supportedMethodConstructorMap:
        raise ValueError(f"MDPtoolbox library does not support resolution method: {resolMethodName}")
    methodConstructor = supportedMethodConstructorMap[resolMethodName]

    # Benchmarking.
    totalTime = 0.
    for _ in range(reps):
        method = methodConstructor()
        startTime = _timer()
        method.run()
        totalTime += _timer() - startTime
    return totalTime / reps


# Map benchmark source name to the corresponding builder.
_recognizedBenchmarkSrcMap = {
    "QComp": saveMDPModelsFromQComp
}

# Map solver name to the corresponding benchmarking function.
_supportedSolverMap = {
    "Marmote": _benchmarkMarmoteSolver,
    "MDPToolbox": _benchmarkMDPToolboxSolver
}

def benchmarkJaniMDPModel(fullModelName: str,
                          solverName: str,
                          srcName: str="QComp",
                          modelParams: dict={},
                          replace: bool=False,
                          resolMethods: list[str]=["ValueIteration", "PolicyIterationModified"],
                          reps: int=20,
                          discount: float=None,
                          epsilon: float=1e-3,
                          maxIter: int=1000,
                          maxInIter: int=10) -> list[dict[str, str | int | float | None]]:
    """Build MDP models from a `.jani` file and benchmark them using specific solver resolution methods.
    
    Parameters:
        fullModelName (str): Full file name of the JANI model.

        solverName (str): Name of the specific solver.

        srcName (str): Name of the benchmark source.

        modelParams (dict): Optional parameter used to configure a scalable model. It defaults to an empty dictionary.

        replace (bool): If True, all corresponding `.pkl` files will be overwritten.

        resolMethods (list[str]): List of resolution methods.

        reps (int): Number of repetitions.

        discount (float): Discount factor.

        epsilon (float): Convergence threshold.

        maxIter (int): Number of maximum iterations.

        maxInIter (int): Number of maximum inner iterations (for modified policy methods).
    
    Returns:
        out(list[dict]): List of dictionaries containing benchmarking results.
    
    Examples:
    >>> import janiParser
    >>> _fullModelName = "consensus.2.jani"
    >>> _modelParams = { "K": 10 }
    >>> _resolMethods = ["ValueIteration", "PolicyIterationModified"]
    >>> _discount = .95
    >>> janiParser.benchmarkJaniMDPModel(_fullModelName,
    >>>                                  solverName="Marmote",
    >>>                                  modelParams=_modelParams,
    >>>                                  replace=True,
    >>>                                  resolMethods=_resolMethods,
    >>>                                  discount=_discount)
    >>> # We slightly adjust the output manually to improve readability.
    [
        {
            'name': 'consensus.2.c1',
            'number-of-states': 1296,
            'number-of-actions': 3,
            'ValueIteration': 0.002238965034484863,
            'PolicyIterationModified': 0.01139671802520752
        },
        {
            'name': 'consensus.2.c2',
            'number-of-states': 1296,
            'number-of-actions': 3,
            'ValueIteration': 0.0021697521209716798,
            'PolicyIterationModified': 0.011333894729614259
        },
        {
            'name': 'consensus.2.disagree',
            'number-of-states': 1296,
            'number-of-actions': 3,
            'ValueIteration': 0.0021898865699768065,
            'PolicyIterationModified': 0.011264896392822266
        },
        {
            'name': 'consensus.2.steps_max',
            'number-of-states': 1296,
            'number-of-actions': 3,
            'ValueIteration': 0.004920089244842529,
            'PolicyIterationModified': 0.015717458724975587
        },
        {
            'name': 'consensus.2.steps_min',
            'number-of-states': 1296,
            'number-of-actions': 3,
            'ValueIteration': 0.004900205135345459,
            'PolicyIterationModified': 0.01584293842315674
        }
    ]
    >>> janiParser.benchmarkJaniMDPModel(_fullModelName,
    >>>                                  solverName="MDPToolbox",
    >>>                                  modelParams=_modelParams,
    >>>                                  resolMethods=_resolMethods,
    >>>                                  discount=_discount)
    [
        {
            'name': 'consensus.2.c1',
            'number-of-states': 1296,
            'number-of-actions': 3,
            'ValueIteration': 0.003207528591156006,
            'PolicyIterationModified': 4.9233436584472656e-05
        },
        {
            'name': 'consensus.2.c2',
            'number-of-states': 1296,
            'number-of-actions': 3,
            'ValueIteration': 0.0031926989555358886,
            'PolicyIterationModified': 5.1593780517578126e-05
        },
        {
            'name': 'consensus.2.disagree',
            'number-of-states': 1296,
            'number-of-actions': 3,
            'ValueIteration': 0.00315701961517334,
            'PolicyIterationModified': 5.010366439819336e-05
        },
        {
            'name': 'consensus.2.steps_max',
            'number-of-states': 1296,
            'number-of-actions': 3,
            'ValueIteration': 0.0055817246437072756,
            'PolicyIterationModified': 4.755258560180664e-05
        },
        {
            'name': 'consensus.2.steps_min',
            'number-of-states': 1296,
            'number-of-actions': 3,
            'ValueIteration': 0.005490529537200928,
            'PolicyIterationModified': 4.876852035522461e-05
        }
    ]
    """
    # Determine the benchmarking function.
    if solverName not in _supportedSolverMap:
        raise ValueError(f"Unsupported solver: {solverName}")
    runner = _supportedSolverMap[solverName]

    # Determine the builder.
    if srcName not in _recognizedBenchmarkSrcMap:
        raise ValueError(f"Unrecognized source: {srcName}")
    builder = _recognizedBenchmarkSrcMap[srcName]

    # Determine the MDP type based on the gievn discount factor.
    if discount is None:
        if solverName == "MDPToolbox":
            raise Exception("MDPToolbox Solver requires a discount factor for resolution")
        mdpType = None
    elif 0 < discount < 1:
        mdpType = "DiscountedMDP"
    elif discount == 1:
        mdpType = "TotalRewardMDP"
    elif discount == -1:
        mdpType = "AverageMDP"
        discount = 1
    else:
        raise Exception(f"Invalid discount factor value: {discount}")
    
    res = []
    savedPaths = builder(fullModelName, modelParams=modelParams, replace=replace)
    for path in savedPaths:
        dataMarmote = loadDataMarmoteFromPklFile(path)

        if solverName == "Marmote":
            # Forced reset of mdp type.
            if mdpType is not None:
                dataMarmote.setMDPTypeTo(mdpType, discount=discount)
            obj = dataMarmote.createMarmoteObject()
        elif solverName == "MDPToolbox":
            obj = dataMarmote.buildTransitionRewardForMDPToolbox()
        
        data = {
            # [Input]   "pklFiles/consensus/consensus.2.c1.pkl"
            # [Output]  "consensus.2.c1"
            "name": os.path.splitext(path.split("/")[-1])[0],
            "number-of-states": dataMarmote.nbStates(),
            "number-of-actions": dataMarmote.nbActions()
        }
        for method in resolMethods:
            try:
                data[method] = runner(obj, method, reps, discount, epsilon, maxIter, maxInIter)
            except Exception as e:
                print(f"Warning: {e}")
                data[method] = None
                continue
        res.append(data)
    return res
