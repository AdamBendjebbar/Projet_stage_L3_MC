#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Marmote and MarmoteMDP and pyMarmoteMDP are free softwares: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#Marmote is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY  without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with MarmoteMDP. If not, see <http://www.gnu.org/licenses/>.

#Copyright 2024 Emmanuel Hyon, Alain Jean-Marie



"""
 * @brief python code to implement an admission control discounted MDP 
 * @author Hyon,
 * @date Oct 2024
 * @version 1.0
 *
 * Example of the admission control described in Busic Bollati et al 
 
 * Description is in the Research Report INRIA 
"""


# import the library
import math as mt
import marmote.core as mc 
import marmote.mdp as md
import numpy as np


class Problem :
	""" definition of class that stores all the variables of the problem  """ 
	
	def __init__(self) :
		self.S = 5                          # Size of the system
		self.lamda = 1			    		# arrival rate
		self.J = 5                          # Number of Customer Classes (index of classes is from 1 to J
		self.p = [0.2,0.3,0.25,0.15,0.10]   # Probability of Customer Classes 
		self.C = [5.0,2.0,1.0,0.5,0.25]     # Rejection Cost
		self.K = 3                          # Number of phasis of hyperexponential
		self.alpha = [0.5,0.2499,0.2501]    # Proba of phasis of the Hyper Expo
		self.mu = [2.0,0.999,1.001]         # Rate of phasis of the Hyper Expo
		self.holdingCost = 0.2              # holding cost


	def __str__(self) :
		s= "## Optimal Admission control\n"
		s+="# Size of the system                = " + str(self.S)+ "\n"
		s+="# Arrival rate                      = " + str(self.lamda)+ "\n" 
		s+="# Number of customer classes        = " + str(self.J)+ "\n"
		s+="# Probability of customer classes   = " + str(self.p)+ "\n"
		s+="# Rejection costs                   = " + str(self.C)+ "\n"
		s+="# Number of phasis                  = " + str(self.K)+ "\n"
		s+="# Phasis probabilities (alpha)      = " + str(self.alpha)+ "\n"						
		s+="# Phasis rates                      = " + str(self.mu)+ "\n"
		s+="# Holding cost                      = " + str(self.holdingCost)+ "\n"
		s+="#"+ "\n"

		s+="#############################" + "\n" 

		return s
			 

# create the default problem
def_prob=Problem()

print("Model printing \n") 
print(str(def_prob))

# Create the elements of the MDP 
# Create the MDP action space
action_space = mc.MarmoteInterval(0,1)
# Create the State Space 
# State Space is a box with 3 dimensions (nb customer in the system,type
# of customer that arrives, phasis of the hyperexpo)
# be carefull that the value of $K$ is from 0 to K-1 
# k=0 in the code means k=1 in the paper
# Now we build the objects
# create dims an array of integers of size 3 given the size of each of the dimension
dims = np.array([def_prob.S+1, def_prob.J+1, def_prob.K])

# create the box of two dimension
state_space = mc.MarmoteBox(dims)
    
#check stateSpace
print("Check the state Space ", str(state_space)," Check the size ",state_space.Cardinal())    
    
print("##\nBegining MDP building\n")
    
# fill in cost matrix  
reward  = mc.FullMatrix(state_space.Cardinal(), action_space.Cardinal())
    
# Creating instantaneous reward matrix
# to do that create a state etat and browse the state space with iterator
# First create the array that stores the state
# indeed a state stores the multidimensionnal state
# the state will be [0,0,0]
etat=np.array([0,0,0])

state_space.FirstState(etat) 
for l in range(state_space.Cardinal()) : 
	# computing the index of the state
	indexO = state_space.Index(etat)
	# Get the number of customer
	x = etat[0]
	# Get the customer class
	j = etat[1]
	# Get the phasis
	k = etat[2]
	# compute the cost for action 0
	cost = 0.0
	if (j > 0) :
		cost += def_prob.C[j-1] #rejection
	cost = cost + (def_prob.S-x) * def_prob.holdingCost
	reward.setEntry(indexO,0,cost)
	# compute the cost for action 1
	costa1=0.0
	if (j > 0):
		if (x==def_prob.S) :
			costa1 += def_prob.C[j-1] #mandatory rejection
			costa1 += (def_prob.S-(x)) * def_prob.holdingCost 
		else :
			costa1 = costa1 + (def_prob.S-(x+1)) * def_prob.holdingCost
	else:
		costa1 = 0.0 + (def_prob.S-x) * def_prob.holdingCost
	reward.setEntry(indexO,1,costa1)
	print("Etat [",x,j,k,"] costs action reject",cost," cost action accept",costa1)
	state_space.NextState(etat)

print("\n###Check reward matrix\n")
print(str(reward))

#also create a variable to store the state after transition
sortie = np.array([0,0,0])

#create transition matrix associated with action 0 (rejection)
P0  = mc.SparseMatrix(state_space.Cardinal())
# fill in matrix
state_space.FirstState(etat) 
for l in range(state_space.Cardinal()) : 
	# computing the index of the input state
	index_input = state_space.Index(etat)
	# Get the number of customer
	x = etat[0]
	# Get the customer class
	j = etat[1]
	# Get the phasis
	k=etat[2]
	print("etat",x,j,k,"index entree",index_input)
	if (def_prob.S == x) :
		# x==S
		# arrivals
		for jprime in range(def_prob.J):
			sortie[0] = x
			sortie[1] = jprime+1
			sortie[2] = k
			probability = def_prob.lamda * def_prob.p[jprime]
			index_output = state_space.Index(sortie)
			print("Sortie: x=",x,"jprime=",jprime,"def_prob.p=",def_prob.p[jprime],"sortie", index_output,"proba",probability)
			P0.setEntry(index_input,index_output,probability)
		# departure
		for kprime in range(def_prob.K):
			sortie[0] = x-1
			sortie[1] = 0
			sortie[2] = kprime
			probability=def_prob.mu[k] * def_prob.alpha[kprime]
			index_output = state_space.Index(sortie)
			print("Sortie: x=",x-1,"kprime=",kprime,"def_prob.a=",def_prob.alpha[kprime],"sortie", index_output,"proba",probability)
			P0.setEntry(index_input,index_output,probability)
	elif (x > 1) :
		# 1 < x < S  
		# arrivals
		for jprime in range(def_prob.J):
			sortie[0] = x
			sortie[1] = jprime+1
			sortie[2] = k
			probability = def_prob.lamda * def_prob.p[jprime]
			index_output = state_space.Index(sortie)
			print("Sortie: x=",x,"jprime=",jprime,"def_prob.p=",def_prob.p[jprime],"sortie", index_output,"proba",probability)
			P0.setEntry(index_input,index_output,probability)
		# departure
		for kprime in range(def_prob.K):
			sortie[0] = x-1
			sortie[1] = 0
			sortie[2] = kprime
			probability = def_prob.mu[k] * def_prob.alpha[kprime] 
			index_output = state_space.Index(sortie)
			print("Sortie: x=",x-1,"kprime=",kprime,"def_prob.a=",def_prob.alpha[kprime],"sortie", index_output,"proba",probability)
			P0.setEntry(index_input,index_output,probability)
	elif (x==1)  :
		# x==1
		# arrivals
		for jprime in range(def_prob.J):
			sortie[0] = x
			sortie[1] = jprime+1
			sortie[2] = k
			probability = def_prob.lamda * def_prob.p[jprime]
			index_output = state_space.Index(sortie)
			print("Sortie: x=",x,"jprime=",jprime,"def_prob.p=",def_prob.p[jprime],"sortie", index_output,"proba",probability)
			P0.setEntry(index_input,index_output,probability)
		#departure
		sortie[0] = 0
		sortie[1] = 0
		sortie[2] = 0
		probability = def_prob.mu[k]  
		index_output = state_space.Index(sortie)
		print("Sortie: x=",x-1,"k=",k,"     def_prob.m=",def_prob.mu[k],"sortie", index_output,"proba",probability)
		P0.setEntry(index_input,index_output,probability)
	else  :
		# x == 0 
		# arrivals
		for jprime in range(def_prob.J):
			sortie[0] = 0
			sortie[1] = jprime+1
			sortie[2] = 0
			probability = def_prob.lamda * def_prob.p[jprime]
			index_output = state_space.Index(sortie)
			print("Sortie: x=",x,"jprime=",jprime,"def_prob.p=",def_prob.p[jprime],"sortie", index_output,"proba",probability)
			P0.setEntry(index_input,index_output,probability)
	# end fill in state		
	state_space.NextState(etat)
	
####################################
#create transition matrix associated with action 1 (accept)
P1  = mc.SparseMatrix(state_space.Cardinal())
# fill in matrix
state_space.FirstState(etat) 
for l in range(state_space.Cardinal()) : 
	# computing the index of the input state
	index_input = state_space.Index(etat)
	# Get the number of customer
	x = etat[0]
	# Get the customer class
	j = etat[1]
	# Get the phasis
	k=etat[2]
	print("Etat",x,j,k,"index entree",index_input)
	if (def_prob.S == x) :
		# x==S
		# arrivals
		for jprime in range(def_prob.J):
			sortie[0] = x
			sortie[1] = jprime+1
			sortie[2] = k
			probability = def_prob.lamda * def_prob.p[jprime]
			index_output = state_space.Index(sortie)
			print("Sortie a1 x=",x,"jprime=",jprime,"def_prob.p=",def_prob.p[jprime],"sortie", index_output,"proba",probability)
			P1.setEntry(index_input,index_output,probability)
		# departure
		for kprime in range(def_prob.K):
			sortie[0] = x-1
			sortie[1] = 0
			sortie[2] = kprime
			probability=def_prob.mu[k] * def_prob.alpha[kprime]
			index_output = state_space.Index(sortie)
			print("Sortie a1 x=",x,"kprime=",kprime,"def_prob.a=",def_prob.alpha[kprime],"sortie", index_output,"proba",probability)
			P1.setEntry(index_input,index_output,probability)
	elif (x > 1) :
		# 1 < x < S  
		if (j >0) :
			# arrivals
			for jprime in range(def_prob.J):
				sortie[0] = x+1
				sortie[1] = jprime+1
				sortie[2] = k
				probability = def_prob.lamda * def_prob.p[jprime]
				index_output = state_space.Index(sortie)
				print("Sortie a1 x=",x,"jprime=",jprime,"def_prob.p=",def_prob.p[jprime],"sortie", index_output,"proba",probability)
				P1.setEntry(index_input,index_output,probability)
			# departure
			for kprime in range(def_prob.K):
				sortie[0] = x
				sortie[1] = 0
				sortie[2] = kprime
				probability = def_prob.mu[k] * def_prob.alpha[kprime] 
				index_output = state_space.Index(sortie)
				print("Sortie a1 x=",x,"kprime=",kprime,"def_prob.a=",def_prob.alpha[kprime],"sortie", index_output,"proba",probability)
				P1.setEntry(index_input,index_output,probability)
		else:
			# arrivals
			for jprime in range(def_prob.J):
				sortie[0] = x
				sortie[1] = jprime+1
				sortie[2] = k
				probability = def_prob.lamda * def_prob.p[jprime]
				index_output = state_space.Index(sortie)
				print("Sortie a1 x=",x,"jprime=",jprime,"def_prob.p=",def_prob.p[jprime],"sortie", index_output,"proba",probability)
				P1.setEntry(index_input,index_output,probability)
			# departure
			for kprime in range(def_prob.K):
				sortie[0] = x-1
				sortie[1] = 0
				sortie[2] = kprime
				probability = def_prob.mu[k] * def_prob.alpha[kprime] 
				index_output = state_space.Index(sortie)
				print("Sortie a1 x=",x,"kprime=",kprime,"def_prob.a=",def_prob.alpha[kprime],"sortie", index_output,"proba",probability)
				P1.setEntry(index_input,index_output,probability)
	elif (x==1)  :
		# x==1
		if (j >0) :
			# arrivals
			for jprime in range(def_prob.J):
				sortie[0] = x+1
				sortie[1] = jprime+1
				sortie[2] = k
				probability = def_prob.lamda * def_prob.p[jprime]
				index_output = state_space.Index(sortie)
				print("Sortie a1 x=",x,"jprime=",jprime,"def_prob.p=",def_prob.p[jprime],"sortie", index_output,"proba",probability)
				P1.setEntry(index_input,index_output,probability)
			#departure
			for kprime in range(def_prob.K):
				sortie[0] = x
				sortie[1] = 0
				sortie[2] = kprime
				probability = def_prob.mu[k] * def_prob.alpha[kprime] 
				index_output = state_space.Index(sortie)
				print("Sortie a1 x=",x,"kprime=",kprime,"def_prob.a=",def_prob.alpha[kprime],"sortie", index_output,"proba",probability)
				P1.setEntry(index_input,index_output,probability)
		else :
			#j==0
			# arrivals
			for jprime in range(def_prob.J):
				sortie[0] = x
				sortie[1] = jprime+1
				sortie[2] = k
				probability = def_prob.lamda * def_prob.p[jprime] 
				index_output = state_space.Index(sortie)
				print("Sortie a1 x=",x,"jprime=",jprime,"def_prob.p=",def_prob.p[jprime],"sortie", index_output,"proba",probability)
				P1.setEntry(index_input,index_output,probability)
			#departure
			sortie[0] = 0
			sortie[1] = 0
			sortie[2] = 0
			probability = def_prob.mu[k] 
			index_output = state_space.Index(sortie)
			print("Sortie a1 x=",x,"k=",k,"     def_prob.m=",def_prob.mu[k],"sortie", index_output,"proba",probability)
			P1.setEntry(index_input,index_output,probability)
	else  :
		# x == 0 
		if (j > 0) :
			# arrivals
			#Here I have to choose the service and then 
			for jprime in range(def_prob.J):
				for kprime in range(def_prob.K):	
					sortie[0] = 1
					sortie[1] = jprime+1
					sortie[2] = kprime 
					probability = def_prob.lamda * def_prob.p[jprime] * def_prob.alpha[kprime]
					index_output = state_space.Index(sortie)
					print("Sortie a1 x=",x,"jprime=",jprime,"def_prob.p=",def_prob.p[jprime],"k",kprime,"sortie", index_output,"proba",probability)
					P1.setEntry(index_input,index_output,probability)
			### departure
			for kprime in range(def_prob.K):
				sortie[0] = 0
				sortie[1] = 0
				sortie[2] = 0
				probability = def_prob.mu[kprime] * def_prob.alpha[kprime] 
				index_output = state_space.Index(sortie)
				print("Sortie a1 x=",x,"k=",kprime,"def_prob.a=",def_prob.alpha[kprime],"sortie", index_output,"proba",probability)
				P1.setEntry(index_input,index_output,probability)
		else :
			# arrivals
			for jprime in range(def_prob.J):
				sortie[0] = 0
				sortie[1] = jprime+1
				sortie[2] = 0
				probability = def_prob.lamda * def_prob.p[jprime]
				index_output = state_space.Index(sortie)
				print("Sortie a1 x=",x,"jprime=",jprime,"def_prob.p=",def_prob.p[jprime],"sortie", index_output,"proba",probability)
				P1.setEntry(index_input,index_output,probability)
			### departure
			sortie[0] = 0
			sortie[1] = 0
			sortie[2] = 0
			probability = 0
			index_output = state_space.Index(sortie)
			print("Sortie a1 x=",x,"sortie", index_output,"proba",probability)
			P1.setEntry(index_input,index_output,probability)
	# end fill in state		
	state_space.NextState(etat)
    
    
#Create a list of Transition Structure
transl = [P0,P1]

#create MDP object
print("\n##Begining MDP building")
critere = "min"
theta = 0.99 #See later how to compute  the appropriate value of theta to get the good discount factor
ctmdp_to_solve = md.ContinuousTimeDiscountedMDP(critere, state_space, action_space, transl,reward,theta)
print("##End of building MDP")

# %%
import json
import os
import marmote.core as mc
import marmote.mdp as md
import numpy as np


def build_expression(variables, values):
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


def create_jani_model(model, criterion, stateSpace, transitions, actionSpace=None, reward=None):
	num_states = stateSpace.Cardinal()
	if actionSpace is not None:
		num_actions = actionSpace.Cardinal()
	else:
		num_actions = 1

	dims = stateSpace.tot_nb_dims()
	variable_names = [f"x{i + 1}" for i in range(dims)]
	variables = []
	for name in variable_names:
		variable_dict = {
			"name": name,
			"type": {
				"kind": "bounded",
				"base": "int",
				"lower-bound": 0,
				"upper-bound": stateSpace.CardinalbyDim(0) - 1
			},
			"initial-value": 0
		}
		variables.append(variable_dict)

	model = {
		"jani-version": 1,
		"name": "MDP Model",
		"Type": model.className(),
		"criterion": criterion,
		"features": ["rewards"],
		"variables": variables,
		"actions": [
			{"name": f"action{i}"} for i in range(num_actions)
		],
		"automata": [{
			"name": "MDPProcess",
			"locations": [{"name": "loc0"}],
			"initial-locations": ["loc0"],
			"edges": []
		}],
		"system": {
			"elements": [
				{
					"automaton": "MDPProcess"
				}]
		}
	}

	automaton = model["automata"][0]

	for a in range(num_actions):
		for i in range(num_states):
			stateIn = stateSpace.DecodeState(i)
			destinations = []
			for j in range(num_states):
				stateOut = stateSpace.DecodeState(j)
				if isinstance(transitions, list):
					prob = transitions[a].getEntry(i, j)
				else:
					prob = transitions.getEntry(i, j)
				if prob > 0:
					if reward is not None:
						reward_value = reward.getEntry(i, a)
					else:
						reward_value = 0
					destinations.append({
						"location": "loc0",
						"probability": {
							"exp": prob
						},
						"assignments": [
							{
								"ref": variable_names[n],
								"value": int(stateOut[n])
							} for n in range(dims)
						],
						"rewards": {
							"exp": reward_value
						}
					})
			automaton["edges"].append({
				"location": "loc0",
				"action": f"act{a}",
				"guard":
					build_expression(variable_names, stateIn)
				,
				"destinations": destinations
			})

	return model


def save_jani_model_to_file(model, filename):
	filename = f"{filename}.janiR"
	counter = 1
	while os.path.exists(filename):
		filename = f"{filename.split('.')[0]}_{counter}.janiR"
		counter += 1

	jani_content = json.dumps(model, indent=2)

	with open(filename, 'w') as file:
		file.write(jani_content)
	print(f"Model saved as {filename}")


model = create_jani_model(ctmdp_to_solve, critere, state_space, transl, action_space, reward)
save_jani_model_to_file(model, "jouet14")
