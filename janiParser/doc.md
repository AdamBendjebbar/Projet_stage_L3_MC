# Documentation

## `janiParser.dataMarmote.DataMarmote`
### Constructor
**`DataMarmote(data: dict)`**

**Parameters**:

* **data**: Dictionary that contains at least the following information:

    * **name**: Name of the model.

    * **type**: Model type. Supported types include: `DiscountedMDP`, `AverageMDP`, `TotalRewardMDP`, `FiniteHorizonMDP` and `MarkovChain`.

    * **states**: Set of states, where each state is represented by a tuple to reduce memory usage.

    * **actions**: Set of actions available.

    * **transition-dict**: Dictionary that contains transition probabilities and rewards. Its structure depends on the model types:  
        * For MDPs: `[action][state][next state]` $\rightarrow$ `NumPy.2DArray[probability, reward]`
        
        * For MCs: `[state][next state]` $\rightarrow$ `probability`
    
    * **absorbing-states**: Set of absorbing states (i.e., states with no outgoing transitions).

    * **state-template**: Template that defines the encoding of states. It is a dictionary that maps each state variable to a unique index, indicating its position in the tuple representation. This is typically used to interpret tuple-encoded states and to ensure a stable state variable order.

    * **state-variable-types**: Dictionary that maps each state variable to its type definition. This is typically used when converting to a `.janir` file.

    * **initial-states**: List of initial states, where each state is represented by a tuple.

    * **state-variable-initial-values**: Dictionary that maps state variables to their initial values.

    * **criterion**: Optimization criterion used for the model. **If the model type is `MarkovChain`, then this field is not required.**

### Methods
* **`createMarmoteObject()`**

    **Construct and return a Marmote object using the data given during class construction.**

* **`buildTransitionRewardForMDPToolbox()`**

    **Build the transition matrices as a list of SciPy sparse matrices and the reward matrix as a full matrix (NumPy 2DArray). These parameters are then passed to `MDPToolbox` to solve the `MDP` problem.**

* **`saveAsJaniRFile(path: str)`**

    **Convert the current model to a `.janir` file and save it to the specified path.**

    **Parameters**:

    * **path**: File path where the generated `.janir` file will be saved.

* **`getInitStatesIdx()`**

    **Return the list of initial states.**

* **`stateToIdx(state: tuple)`**

    **Return the unique index corresponding to the specified state.**

    **Parameters**:

    * **state**: Tuple representation of a specific state.

* **`ìdxToState(idx: int)`**

    **Return the state corresponding to the index.**

    **Parameters**:

    * **idx**: Integer that represents the index of the constructed matrices.

* **`nbStates()`**

    **Return the number of states.**

* **`nbActions()`**

    **Return the number of actions.**

* **`setMDPTypeTo(mdpType: str, discount: float=.95, horizon: int=1)`**

    **Set MDP model type and associated parameters.**

    **Parameters**:

    * **mdpType**: The MDP model type.
            
    * **discount**: The discount facotor used in `DiscountedMDP` and `FiniteHorizonMDP`.

    * **horizon**: The horizon length used in `FiniteHorizonMDP`.

## `janiParser.reader.reader.JaniReader`
### Constructor
`JaniReader(path: str, modelParams: dict={}, isLocalPath: bool=True)`

**Parameters**:

* **path**: Path to the `.jani` file, which can be a local path or a URL path.

* **modelParams**: Optional parameter used to configure a scalable model.

* **isLocalPath**: Boolean that indicates whether the given path is a local file path or not.

**Examples**:
```python
import janiParser

# Example of using a local file path.
reader = janiParser.JaniReader("./benchmarks/prism2jani/consensus.2.v1.jani", modelParams={ "K": 10 })
```

```python
# Example of using a URL path.
reader = janiParser.JaniReader("https://qcomp.org/benchmarks/mdp/consensus/consensus.2.jani", modelParams={ "K": 10 }, isLocalPath=False)
```

### Methods
* **`build()`**

    **Load and parse the target file specified during class construction to build a JANI model representation. It returns an instance of class `JaniModel`.**

## `janiParser.reader.reader.JaniRReader`
### Constructor
`Bases: janiParser.reader.reader.JaniReader`

`JaniRReader(path: str, modelParams: dict={})`

**Parameters**:

* **path**: Path to the `.janir` file. It accepts only the local file path.

* **modelParams**: Optional parameter used to configure a scalable model.

### Methods
* **`build()`**

    **Load and parse the target file specified during class construction to build a JANIR model representation. It returns an instance of class `JaniRModel`.**

## `janiParser.reader.model.JaniModel`
### Constructor
`JaniModel(name: str, type: str)`

**Parameters**:

* **name**: Name of the model.

* **type**: Type of the model specified according to the JANI standard. **In this project, only `dtmc` and `mdp` types are supported.**

### Methods
* **`getMDPData(name: str)`**

    **Starting from the initial states, it explores all reachale states under the specified property and returns a dictionary with all necessary data to build a corresponding Marmote instance (`MDPs`). See the `DataMarmote` class documentation above for more details on the returned data.**

    **Parameters**:

    * **name**: Name of the property to be modeled.

* **`getMCData()`**

    **This method is similar to `getMDPData`. It explores all reachable states and returns a dictionary with all necessary data to build a corresponding Marmote instance (`MCs`). See the `DataMarmote` class documentation above for more details on the returned data.**

## `janiParser.reader.model.JaniModel`
### Constructor
`Bases: janiParser.reader.model.JaniModel`

`JaniRModel(name: str, type: str, criterion: str, gamma: float=None, horizon: int=None)`

**Parameters**:

* **name**: Name of the model.

* **type**: Type of the model. Supported types include: `DiscountedMDP`, `AverageMDP`, `TotalRewardMDP`, `FiniteHorizonMDP` and `MarkovChain`.

* **criterion**: Optimization criterion used for the model (`"max"` or `"min"`).

* **gamma**: Discount factor used in discounted and finite horizon models.

* **horizon**: Horizon length used in finite horizon models.

### Methods
* **`getMDPData()`**

    **This method takes no arguments. Starting from the initial states, it explores all reachale states under the specified property and returns a dictionary with all necessary data to build a corresponding Marmote instance (`MDPs`). See the `DataMarmote` class documentation above for more details on the returned data.**

* **`getMCData()`**

    **This method is similar to `getMDPData`. It explores all reachable states and returns a dictionary with all necessary data to build a corresponding Marmote instance (`MCs`). See the `DataMarmote` class documentation above for more details on the returned data.**
