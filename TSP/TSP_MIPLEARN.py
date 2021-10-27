import pyomo.environ as pyEnv
import numpy as np
from scipy.spatial import distance
from scipy.stats import uniform
from miplearn import LearningSolver
from miplearn import Instance
from random import randint

##-------------------------Class for generating one TSP_Instance--------------------##

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

##-------------------------INSTANCE GENERATOR--------------------##

def instance_generator(amount_instances,amount_cities):
    instances = []
    cities = generate_same_cities(amount_cities)
    for i in range(amount_instances):
        new_cities = generate_slightly_different_cities(cities)
        cost_matrix = compute_distances(new_cities)
        instance = TSP_instance(amount_cities,cost_matrix,new_cities)
        instances.append(instance)
    #print([i.get_cities() for i in instances])
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

##-------------------------CITIES GENERATOR--------------------##
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


##-------------------------MIPLearn Solver--------------------##
import pickle
def training():
    
    ##-------------------------TRAINING DATA--------------------##
    training_instances = instance_generator(50,20) #genereer 100 keer 25steden die lichtjes anders zijn
    
    ##-------------------------SOLVE ALL TRAINING INSATNCES--------------------##
    solver = LearningSolver(solver="gurobi")
    '''
    Learning Solver = a learning-enhanced MIP solver which uses information from previously solved instances to accelerate the solution of new instances
    '''
    
    for instance in training_instances:
        solver.solve(instance)
    
    ##-------------------------LEARN FROM TRAINING INSTANCES--------------------##
    print("Training... please be patient")
    solver.fit(training_instances)
    print("Training is complete!")
    pickle.dump(solver,open("solver.pickle","wb"))

training() #has to be done before calling miplearnSolver()!!


def miplearnSolver():
    
    ##-------------------------SOLVE TEST INSTANCE--------------------##
    
    ##-------------------------TEST DATA--------------------##
    #test_instances = instance_generator(1,25)[0]
    test_instances = instance_generator(1,25)[0]
    cities = test_instances.get_cities()
    print('test instance for miplearn: \n',cities)

    # Load trained solver from disk
    solver = pickle.load(open("solver.pickle", "rb"))
    results = solver.solve(test_instances)
    print("miplearn model: \n")
    #print(results['Log'])

    bestCities = []
    x = test_instances.solution["x"]
    List = list(x.keys())
    for i in List:
        if x[i] == 1.0:
            bestCities.append([*i])
            print(i,'--', x[i])
    
    walltime = results['Wallclock time']
    print(f"wallclocktime: {walltime}")
    
    return cities,bestCities,walltime




##-------------------------RegularSolver--------------------##

def regularSolver():
    test_instance = instance_generator(1,25)[0]
    cities = test_instance.get_cities()
    #print('test instance for regular solve: \n\n',cities)
    
    model = test_instance.to_model()
    
    #Solves
    solver = pyEnv.SolverFactory('gurobi')
    solver.options['timelimit'] = 10
    result = solver.solve(model,tee = True)
    
    walltime = result.Solver.wall_time
    data = result.Problem._list
    LB = data[0].lower_bound
    UB = data[0].upper_bound
    gap = (UB-LB)/UB
    print("\n test \n\n")
    print(gap*100)
    
    
    #Prints the results
    #print("\n regular model wall time: \n")
    #print(result.Solver.wall_time)
    
    ##(city i,city j)
    cities = test_instance.get_cities()
    bestCities = []
    List = list(model.x.keys())
    for i in List:
        if model.x[i]() == 1.0:
            bestCities.append([*i])
            #print(i,'--', model.x[i]())
    
    #print('cities: \n',cities)
    #print('best cities: \n',bestCities)
    return cities,bestCities,walltime

##-------------------------VISUALIZE MAP OF CITIES--------------------##
import matplotlib.pyplot as plt

def plot_cities(cities,bestCities,method,walltime):
    x = cities[:,0]
    y = cities[:,1]
    plt.figure()

    plt.scatter(x,y)
    for i in bestCities:
        xco = []
        yco = []
        cityA = i[0] #from A to B
        cityB = i[1]
        xco.append(cities[cityA-1][0])
        xco.append(cities[cityB-1][0])
        yco.append(cities[cityA-1][1])
        yco.append(cities[cityB-1][1])
        if method == 'miplearn':
            color = 'y'
        elif method == 'basic':
            color = 'k'
        plt.plot(xco,yco,color = color)
        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Map of cities with {method} in {walltime}s")
    plt.legend()
    plt.show()


    

##-------------------------Call functions--------------------##
def runMIPLearn():
    print("\n\n miplearn model")
    c,bestC,time = miplearnSolver()
    plot_cities(c,bestC,'miplearn',round(float(time),4))

def runBasic():
    
    print("\n\n normal solving model")
    c_basic,bestC_basic ,time_basic = regularSolver()
    plot_cities(c_basic,bestC_basic,'basic',round(float(time_basic),4))


runBasic()
runMIPLearn()
