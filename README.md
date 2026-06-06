# L3 Internship Project — Markov Chain Modelling and Description

This project is inspired by the work of **Zeyu TAO** and **Jiahua LI**, students in the ANDROIDE Master's program at Sorbonne University. Their original repository is available here:
[https://github.com/bellkilo/pAI2D_mdp_languages](https://github.com/bellkilo/pAI2D_mdp_languages)

This tool parses JANI model files to build Markov chains and analyse them using the **Marmote** library. The pipeline reads a JANI file, explores the state space, builds the transition matrix, and computes the stationary distribution.

---

## Environment Setup (Conda + Marmote)

Create the environment:

```bash
conda create --name marmote_env -c marmote -c conda-forge marmote=1.3.1 --yes
```

Activate it:

```bash
conda activate marmote_env
```

Install the required dependencies:

```bash
conda install -c conda-forge numpy scipy typing_extensions requests tqdm --yes
```

---

## Running `test_final.py`

From the project root:

```bash
python3 test_final.py
```

---

## What `test_final.py` does

`test_final.py` runs an interactive analysis pipeline:

1. Prompts the user to enter the BRP model constants (`N` and `MAX`) and the initialisation mode (`first`, `uniform`, `random`).
2. Reads `brp.jani` and builds the internal model representation via `JaniReader`.
3. Extracts Markov chain data (`getMCData`) and creates the Marmote object via `DataMarmote`.
4. Runs the transition matrix diagnostic (`Diagnose()`).
5. Computes the stationary distribution using two independent methods:
   - `StationaryDistributionPower(...)`
   - `StationaryDistribution()`
6. Prints the most probable equilibrium states.
