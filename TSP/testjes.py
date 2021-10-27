import pyomo.environ as pyEnv
import numpy as np
from scipy.spatial import distance
from scipy.stats import uniform
from miplearn import LearningSolver
from miplearn import Instance
from random import randint

class TSP_instance(Instance):
    """
    Instances to be solved by LearningSolver must derive from the abstract class miplearn.Instance
    The following three abstract methods must be implemented:
        
        instance.to_model(), which returns a concrete Pyomo model corresponding to the instance;
        
        instance.get_instance_features(), which returns a 1-dimensional Numpy array of (numerical) features 
        describing the entire instance;
        
        instance.get_variable_features(var_name, index), which returns a 1-dimensional array of (numerical) features 
        describing a particular decision variable.
    """
    
    def __init__(self,amountCities,cost_matrix,cities):
        self.n = amountCities
        self.cost_matrix = cost_matrix
        self.cities = cities
        
    def to_model(self):
        
        ##-------------------------MODEL INITIALIZATION AND ITS PARAMETERS--------------------##
        
        #Model
        model = pyEnv.ConcreteModel()
        
        #Indexes for the cities
        model.M = pyEnv.RangeSet(self.n)                
        model.N = pyEnv.RangeSet(self.n)
        
        #Index for the dummy variable u
        model.U = pyEnv.RangeSet(2,self.n)
        
        #Decision variables xij
        model.x=pyEnv.Var(model.N,model.M, within=pyEnv.Binary)
        
        #Dummy variable ui
        model.u=pyEnv.Var(model.N, within=pyEnv.NonNegativeIntegers,bounds=(0,self.n-1))
        
        #Cost Matrix cij
        model.c = pyEnv.Param(model.N, model.M,initialize=lambda model, i, j: self.cost_matrix[i-1][j-1])
        
        ##-------------------------OBJECTIVE FUNCTION AND CONSTRAINTS--------------------##
        
        def obj_func(model):
            return sum(model.x[i,j] * model.c[i,j] for i in model.N for j in model.M)
        
        model.objective = pyEnv.Objective(rule=obj_func,sense=pyEnv.minimize)
        
        ##------------------------------------------------------##
        #Only 1 leaves each city
        
        def rule_const1(model,M):
            return sum(model.x[i,M] for i in model.N if i!=M ) == 1
        
        model.const1 = pyEnv.Constraint(model.M,rule=rule_const1)
        ##------------------------------------------------------##
        #Only 1 enters each city
        
        def rule_const2(model,N):
            return sum(model.x[N,j] for j in model.M if j!=N) == 1
        
        model.rest2 = pyEnv.Constraint(model.N,rule=rule_const2)
        ##------------------------------------------------------##
        #Only a single tour covering all cities
        
        def rule_const3(model,i,j):
            if i!=j: 
                return model.u[i] - model.u[j] + model.x[i,j] * self.n <= self.n-1
            else:
                
                return model.u[i] - model.u[i] == 0 
            
        model.rest3 = pyEnv.Constraint(model.U,model.N,rule=rule_const3)
        
        return model
    
    def get_instance_features(self):
        return np.array([1])
    
    def get_variable_features(self, var, index):
        return np.array([1])
    
    def get_cities(self):
        return self.cities


def instance_generator(amount_instances,amount_cities):
    instances = []
    cities = generate_same_cities(amount_cities)
    for i in range(amount_instances):
        new_cities = generate_slightly_different_cities(cities)
        cost_matrix = compute_distances(new_cities)
        print(f"\n {cost_matrix}")
        instance = TSP_instance(amount_cities,cost_matrix,new_cities)
        instances.append(instance)
    print([i.get_cities() for i in instances])
    return instances


def random_instance_generator(amount_instances,amount_cities):
    instances = []
    for i in range(amount_instances):
        new_cities = generate_random_cities(amount_cities)
        cost_matrix = compute_distances(new_cities)
        print(f"\n {cost_matrix}")
        instance = TSP_instance(amount_cities,cost_matrix,new_cities)
        instances.append(instance)
    print([i.get_cities() for i in instances])
    
    return instances

def generate_same_cities(amount):
    """
    Parameters
    ----------
    amount : int
        amount of cities you would like to generate.

    Returns
    -------
    cities : 2D np.array([[x1,y1],
                       [x2,y2],
                       ...])
        List of integer coordinates of all cities.

    """
    np.random.seed(0) #generate the same amount of cities everytime!
    x = uniform(loc = 0, scale = 10000) #x is uniform verdeeld in domein [0,1000]
    y = uniform(loc = 0, scale = 10000)
    cities = np.array([(round(x.rvs(),3),round(y.rvs(),3)) for i in range(amount)])
    return cities

def generate_random_cities(amount):
    """
    Parameters
    ----------
    amount : int
        amount of cities you would like to generate.

    Returns
    -------
    cities : 2D np.array([[x1,y1],
                       [x2,y2],
                       ...])
        List of integer coordinates of all cities.

    """
    x = uniform(loc = 0, scale = 10000) #x is uniform verdeeld in domein [0,1000]
    y = uniform(loc = 0, scale = 10000)
    cities = np.array([(round(x.rvs(),3),round(y.rvs(),3)) for i in range(amount)])
    return cities

def generate_slightly_different_cities(cities):
    alfa_distr = uniform(-2,4) #uniforme distributie van -2 tot 2
    new_cities = np.array([0]*2)
    for i in cities:
        row = [(j+alfa_distr.rvs()) for j in i]
        row_array = np.array(row) #needs to be an np.array
        #np.append(new_cities,row_array)
        new_cities = np.vstack([new_cities, row_array])
    
    new_cities = np.delete(new_cities,0,0)

    return new_cities

##-------------------------COMPUTE EUCLIDIAN DISTANCES = cost_matrix--------------------##

def compute_distances(cities):
    """
    Parameters
    ----------
    cities : int
        List of cities.

    Returns
    -------
    list : 2D List of integers 
        all the distances between cities, from city i to city j. (i=row, j=column)

    """
    distance_matrix = np.array([0]*len(cities))
    for i in cities:
        row = [round(distance.euclidean(i, j),3) for j in cities]
        row_array = np.array(row) #needs to be an np.array
        np.append(distance_matrix,row_array)
        distance_matrix = np.vstack([distance_matrix, row_array])
    
    distance_matrix = np.delete(distance_matrix,0,0)
    #print(distance_matrix)
    list = distance_matrix.tolist()
    return list

instance_generator(5,3)
random_instance_generator(5,3)
