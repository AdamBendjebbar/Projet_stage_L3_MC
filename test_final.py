import sys
import os
import numpy as np
import marmote.core as mc
import sys


#Configuration du chemin
sys.path.append(os.getcwd())
from janiParser.reader.reader import JaniReader
from janiParser.dataMarmote import DataMarmote

 #methode iterative pour etudier la convergence de pi vers létat d'équilibre, garatissant que le modele mathématique extrait de jani est stable

def test_diagnose_on_new_matrix():
    print("\n--- TEST DIAGNOSTIC SUR MATRICE VIERGE ---")
    
    # 1. Création d'un espace d'états simple (2 états : 0 et 1)
    test_space = mc.MarmoteInterval(0, 1)
    
    # 2. Création d'une SparseMatrix (qui hérite de TransitionStructure)
    test_matrix = mc.SparseMatrix(test_space)
    test_matrix.set_type(mc.DISCRETE)
    
    # 3. On ajoute une transition pour ne pas avoir une matrice totalement vide
    # État 0 -> État 1 avec proba 1.0
    test_matrix.addEntry(0, 1, 1.0)
    # État 1 -> État 1 avec proba 1.0 (auto-boucle pour être stochastique)
    test_matrix.addEntry(1, 1, 1.0)
    
    print(f"Matrice de test créée. Type : {type(test_matrix)}")
    
    try:
        # Tentative de diagnostic
        print("Tentative de test_matrix.Diagnose()...")
        test_matrix.Diagnose() 
        print("Succès du Diagnose sans arguments.")
    except Exception as e:
        print(f"Échec Diagnose() sans arguments : {e}")
        
    try:
        # Tentative avec le mode vide (ce que tu as essayé)
        print("\nTentative de test_matrix.Diagnose(\"\")...")
        test_matrix.Diagnose("")
    except Exception as e:
        print(f"Échec Diagnose(\"\") : {e}")

def run_analysis():
    print("\n--- ANALYSE DE CHAÎNE DE MARKOV (JANI -> MARMOTE) ---")
    path = "./benchmarks/mcJani/brp.jani"
    params = {"N": 4, "MAX": 3}
    path1="./benchmarks/mcJani/bluetooth.v1.jani"
    path2="./benchmarks/mcJani/haddad-monmege.v1.jani"
    params1 = {
    
         "N": 4,
        "MAX": 3,
        "phase": 2,
        "maxr":    4,
         "mrep": 4,
         "mrec": 4,  # La valeur qui manquait !
         "k": 1,
         "T": 0
    }
    params2 = {
    
         "N": 4,
        "p": 0.5,
        "q": 0.5
    }


    try:
        #  instanciation d'un objet JaniReader
        reader = JaniReader(path2, modelParams=params2)
        #  retourne un objet JaniModel
        model = reader.build()
        
        # Inspection des états initiaux

        # Retourne une liste d'objets State, ce sont les etats initiaux
        init_states = model.getInitStates()
        print(f"\n--- INSPECTION ({len(init_states)} état(s) initial/aux) ---")
        #  enumere les etat et les associe a un indice i
        for i, state in enumerate(init_states):
            #  affoche l'etat i
            print(f"\nÉtat #{i}")
            #  affiche la loc
            print(f"Locations: {list(state.setOfLocation)}")
            if hasattr(state, '_nonTransientVars'):
                for name, var in state._nonTransientVars.items():
                    print(f"  {name} = {var.value}")

        # Conversion
        # Renvoie un dictionnaire qui va permettre d'initialiser l'objet DataMarmot
        mdp_data = model.getMCData() 

        # creation de l'objet DataMarmott contenant listes de tuples etats initiaux, liste de tuple etats, dict transition et initialise 2 dict pour passer de tuple a index et de index a tuple
        dm = DataMarmote(mdp_data)

        # Cree un objet MarkovChain de la bibliothèque Marmotte contentant la matrice de transition creuse et la distribution initial (qui depend du second parametre de la focntion)
        obj_marmote = dm.createMarmoteObject(init_mode="first")
        
        print("\n--- DIAGNOSTIC DE LA MATRICE DE TRANSITION ---")
    
    # On récupère la matrice que Marmote a construite
        
        
        #matrix = dm.get_transition_matrix()

# 2. CONFIGURATION DE L'AFFICHAGE (Optionnel, comme dans ton exemple)

       
# 3. L'APPEL CORRECT : 
# Selon la documentation Marmote, "cerr" ou "-" sont les descripteurs 
# internes pour le flux de sortie standard C++.
       
        
        # Analyse
        #  contient un pointeur vers la matrice de transition contenue dasn l'objet markov chain obj_marmot
        # SSimulation de la Chaine, rencoie un SimulationResult
        diag= obj_marmote.SimulateChainDT( 10, False, True, False )
        # Diagnostique de la chaine
        print(diag.Diagnose())

        init_dist = dm.getInitDistribution()
      
       
         # stationary = obj_marmote.StationaryDistributionRLGL(100000000,  1e-8,       init_dist,   False,       0.5,        "thresh_1"   )
        stationary = obj_marmote.StationaryDistributionPower(100000000,  1e-8,       init_dist,   False  )
        print(f"Proba stationnaires selon RLGL:{stationary}")
         #  calcul de la distribution stationnaire
        pi_marmote = obj_marmote.StationaryDistribution()
        print(f"Probabilités stationnaires (extraits) : {pi_marmote}")

        # Omet toutes les transitions entre états dont la probabilité est null et n'affiche que les transition sous la forme: etat source, etat dest, proba
        # print(gen.toString())

        #  format numpy interpretable par exec
        #   matrix_str = gen.toString(mc.FORMAT_NUMPY)
        

        #  affiche pour observer la matrice de transition cette fois ci ca affiche toute la matrice y compris les transitions dont la proba est null
        
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    #test_diagnose_on_new_matrix()
    run_analysis()