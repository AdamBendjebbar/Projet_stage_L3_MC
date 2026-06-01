import sys
import os
import numpy as np
import marmote.core as mc
import sys

import subprocess
import json

# Path configuration
sys.path.append(os.getcwd())
from janiParser.reader.reader import JaniReader
from janiParser.dataMarmote import DataMarmote
from janiParser.utils import benchmarkJaniMCModel

 # Iterative method to study the convergence of pi toward the equilibrium state,
 # ensuring the mathematical model extracted from JANI is stable.



def run_analysis():
    print("\n" + "="*60)
    print(" INTERACTIVE MANUAL ANALYSIS OF A MARKOV CHAIN")
    print("="*60)
    
    # --- 1. BRP parameter configuration ---
    print("\n  Configure constants for the BRP model:")
    
    while True:
        try:
            val_n = input(" Value of N (e.g.: 4) [Press Enter for 4]: ").strip()
            
            n_val = int(val_n)
            if n_val > 0: break
            print(" Error: N must be strictly greater than 0.")
        except ValueError:
            print(" Invalid input. Please enter an integer.")

    while True:
        try:
            val_max = input(" Value of MAX (e.g.: 3) [Press Enter for 3]: ").strip()
            
            max_val = int(val_max)
            if max_val > 0: break
            print(" Error: MAX must be strictly greater than 0.")
        except ValueError:
            print(" Invalid input. Please enter an integer.")

    params = {"N": n_val, "MAX": max_val}
    path = "./benchmarks/mcJani/brp.jani"

    # --- 2. Initialization mode input ---
    modes_valides = ["first", "uniform", "random"]
    while True:
        print("\n Chain creation modes (Initial Distribution):")
        print("   Options: first, uniform, random")
        choice = input(" Enter selected mode: ").strip().lower()
        
        if choice in modes_valides:
            init_mode_chosen = choice
            break
        else:
            print(f" Error: '{choice}' is not a valid mode.")

    # --- 3. Run analysis ---
    try:
        print(f"\n Reading file {os.path.basename(path)} with parameters {params}...")
        reader = JaniReader(path, modelParams=params)
        model = reader.build()
        
        init_states = model.getInitStates()
        print(f"\n--- INSPECTION ({len(init_states)} initial state(s)) ---")
        
        for i, state in enumerate(init_states):
            print(f"\nState #{i}")
            print(f"  Locations: {list(state.setOfLocation)}")
            if hasattr(state, '_nonTransientVars'):
                for name, var in state._nonTransientVars.items():
                    print(f"  {name} = {var.value}")

        mc_data = model.getMCData() 
        dm = DataMarmote(mc_data)
        
        print(f"\n Extraction succeeded. Creating matrix (Mode: '{init_mode_chosen}')...")
        obj_marmote = dm.createMCObject(init_mode=init_mode_chosen)

        print("\n--- TRANSITION MATRIX DIAGNOSTIC ---")
        transition_matrix = obj_marmote.generator()
        transition_matrix.Diagnose()
  
        # ==========================================
        # COMPUTATION AND EXTRACTION OF STATIONARY STATES
        # ==========================================
        print("\n--- COMPUTING STATIONARY PROBABILITIES ---")
        init_dist = dm.getInitDistribution()
        nb_etats = dm.nbStates()
       
        # 1. Power method
        print("\n Launching 'Power' method...")
        stationary_power = obj_marmote.StationaryDistributionPower(100000000, 1e-8, init_dist, False)
        print(" Computation completed. Equilibrium state analysis:")
        
        trouve_power = 0
        for i in range(nb_etats):
            try:
                proba = stationary_power.getProbaByIndex(i)
            except AttributeError:
                proba = stationary_power.getProbaByIndex(i)
                
            if proba > 0.000001: 
                print(f"   ▶ State #{i} concentrates {proba * 100:.4f}% of the probability")
                trouve_power += 1
                
        if trouve_power == 0:
            print("    Probability is too diluted to stand out.")

        # 2. Classic iterative method
        print("\n Launching 'Classic' method...")
        pi_marmote = obj_marmote.StationaryDistribution()
        print(" Computation completed. Equilibrium state analysis:")
        
        trouve_classique = 0
        for i in range(nb_etats):
            try:
                proba = pi_marmote.getProbaByIndex(i)
            except AttributeError:
                proba = pi_marmote.getProbaByIndex(i)
                
            if proba > 0.000001: 
                print(f"   State #{i} concentrates {proba * 100:.4f}% of the probability")
                trouve_classique += 1
                
        if trouve_classique == 0:
            print("    Probability is too diluted to stand out.")
       
        
    except Exception as e:
        print(f"\n Execution error: {e}")
# Path configuration
sys.path.append(os.getcwd())
from janiParser.utils import benchmarkJaniMCModel


    

if __name__ == "__main__":
    run_analysis()
    