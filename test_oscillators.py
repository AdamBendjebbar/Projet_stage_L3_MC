import os
import sys

sys.path.append(os.getcwd())

from janiParser.reader.reader import JaniReader
from janiParser.dataMarmote import DataMarmote


def run_oscillators_analysis():
    print("\n" + "=" * 70)
    print(" OSCILLATORS DTMC ANALYSIS")
    print("=" * 70)

    # Constants dictionary for oscillators.3-6-0.1-1.v1
    params = {
        "mu": 0.1,
        "lambda": 1.0,
    }

    path = "./benchmarks/mcJani/oscillators.3-6-0.1-1.v1.jani"
    init_mode_chosen = "uniform"

    try:
        print(f"\nReading file {os.path.basename(path)} with constants {params}...")
        reader = JaniReader(path, modelParams=params)
        model = reader.build()

        init_states = model.getInitStates()
        print(f"\n--- INSPECTION ({len(init_states)} initial state(s)) ---")
        for i, state in enumerate(init_states):
            print(f"\nState #{i}")
            print(f"  Locations: {list(state.setOfLocation)}")
            if hasattr(state, "_nonTransientVars"):
                for name, var in state._nonTransientVars.items():
                    print(f"  {name} = {var.value}")

        print("\nExtracting MC data...")
        mc_data = model.getMCData()
        dm = DataMarmote(mc_data)

        print(f"\nCreating Markov chain object (init_mode='{init_mode_chosen}')...")
        obj_marmote = dm.createMCObject(init_mode=init_mode_chosen)

        print("\n--- TRANSITION MATRIX DIAGNOSTIC ---")
        transition_matrix = obj_marmote.generator()
        transition_matrix.Diagnose()

        print("\n--- STATIONARY PROBABILITIES (POWER) ---")
        init_dist = dm.getInitDistribution()
        nb_states = dm.nbStates()

        stationary_power = obj_marmote.StationaryDistributionPower(100000000, 1e-8, init_dist, False)

        found_power = 0
        for i in range(nb_states):
            proba = stationary_power.getProbaByIndex(i)
            if proba > 1e-6:
                print(f"  ▶ State #{i}: {proba * 100:.4f}%")
                found_power += 1
        if found_power == 0:
            print("  No state above threshold.")

        print("\n--- STATIONARY PROBABILITIES (CLASSIC) ---")
        pi_marmote = obj_marmote.StationaryDistribution()

        found_classic = 0
        for i in range(nb_states):
            proba = pi_marmote.getProbaByIndex(i)
            if proba > 1e-6:
                print(f"  State #{i}: {proba * 100:.4f}%")
                found_classic += 1
        if found_classic == 0:
            print("  No state above threshold.")

        print("\nAnalysis completed successfully.")

    except Exception as e:
        print(f"\nExecution error: {e}")


if __name__ == "__main__":
    run_oscillators_analysis()
