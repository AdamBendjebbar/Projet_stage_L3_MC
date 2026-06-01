# pAI2D_mdp_languages

Small project for parsing JANI models, building Markov chains/MDPs, and running analysis workflows with Marmote (transition diagnostics and stationary distribution computations).

## Environment setup (Conda + Marmote)

Create the environment:

```bash
conda create --name marmote_env -c marmote -c conda-forge marmote=1.3.1 --yes
```

Activate it:

```bash
conda activate marmote_env
```

Install the additional packages:

```bash
conda install -c conda-forge numpy scipy typing_extensions requests tqdm --yes
```

## Run `test_final.py`

From the project root:

```bash
python3 test_final.py
```

## What `test_final.py` does

`test_final.py` runs an interactive analysis pipeline:

1. Reads a JANI benchmark model with selected constants.
2. Builds the internal model representation through `JaniReader`.
3. Extracts MC data (`getMCData`) and creates a Marmote Markov chain object.
4. Runs transition matrix diagnostics (`Diagnose()`).
5. Computes stationary distributions with:
   - `StationaryDistributionPower(...)`
   - `StationaryDistribution()`
6. Prints the most probable equilibrium states.

This script is intended to validate model extraction quality and basic stochastic behavior analysis.
