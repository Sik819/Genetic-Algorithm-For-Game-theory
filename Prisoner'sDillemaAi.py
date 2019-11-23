import random
import deap
from deap import base, creator, tools

PD_payOff_Matrix = {"DD": (1,1), "DC": (5,0), "CD": (0,5), "CC": (3,3)}
GC_payOff_Matrix = {"DD": (0,0), "DC": (5,1), "CD": (1,5), "CC": (3,3)}


#This function is used as a helper function for payoff_to_ind1
#it returns the payoff to individual at 1 round 
"""
	 * @param individual1, individual2: 
             Move by individual1 and individual2 ⊆ (0,1)
     * @param game: game played by the individuals ⊆ ("IGC": iterated game of chicken, 
                                                      "IPD": Iterated game of prisoners dilema) 
	 * @return the score achieved by individual1 
	 * return null if moves invalid 
"""

def payoff_at_one_round(individual1, individual2, game):
    individual1_move = individual1
    individual2_move = individual2
    
    if(game == "IPD"):
        if(individual1_move == 1):
            if(individual2_move == 0):
                return PD_payOff_Matrix["DC"][0]
            else:
                return PD_payOff_Matrix["DD"][0]
            
        if(individual1_move == 0):
            if(individual2_move == 0):
                return PD_payOff_Matrix["CC"][0]
            else:
                return PD_payOff_Matrix["CD"][0]
            
    if(game == "IGC"):
        if(individual1_move == 1):
            if(individual2_move == 0):
                return GC_payOff_Matrix["DC"][0]
            else:
                return GC_payOff_Matrix["DD"][0]
            
        if(individual1_move == 0):
            if(individual2_move == 0):
                return GC_payOff_Matrix["CC"][0]
            else:
                return GC_payOff_Matrix["CD"][0]
    return null
            

    
#This function is used to play the actual game individual 1 vs individual 2
"""
	 * @param individual1, individual2: 
             The chromosomes of individual1 and individual2 ⊆ [(0,1)*] 
	 * @param game: game played by the individuals ⊆ ("IGC": iterated game of chicken, 
                                                      "IPD": Iterated game of prisoners dilema) 
	 The game will be played for 16 rounds 
	 * @return the total scores achieved by individual1 for the game
"""
def payoff_to_ind1(individual1, individual2, game):
    sum = 0 
    numRounds = 16 #since there are 16 possibile combinations let there by a totl of 16 rounds
    #At the benigning the individuals have no memory bits since no game has been played
    #Hence update memory bits for first two rounds
    for i in range(2):
        move_ind1= move_by_ind1(individual1, individual2,i+1)
        individual1.append(move_ind1)
        
        move_ind2= move_by_ind1(individual2, individual1,i+1)
        individual2.append(move_ind2)
        
        sum += payoff_at_one_round(move_ind1,move_ind2,game)
    #Get the move for the current round by move_ind1() function
    #As the moves are made update the memory bits for each individual
    for i in range(3,numRounds+1):
        move_ind1= move_by_ind1(individual1, individual2,i) #getMove for the round
        process_move(individual1, move_ind1, 2) #update memory bits
        move_ind2= move_by_ind1(individual2, individual1,i)
        process_move(individual2, move_ind2, 2)
        sum += payoff_at_one_round(move_ind1,move_ind2,game)
    
    #After the games are played the memory bits has to be removed
    for i in range(1,3):
        del individual1[-i]
        del individual2[-i]
    
    return sum

#helper function to convert binary to decimal 
def binaryToDecimal(binary): 
       return int(binary, 2)  

#This function is used to get the move of the individual at a perticular round
"""
	 * @param individual1, individual2: 
             The chromosomes of individual1 and individual2 ⊆ [(0,1)*] 
	 * @param round: The round at which the game is being played ⊆ [(1-16)]
	 The moves are determined using the combination of individual1 and individual2 memory bits
     * @return the move for the individual1 for the current round ⊆ [(0,1)] based on individual1s    stratergy bits
"""
def move_by_ind1(individual1, individual2, round):
    #assuming last two bits of the string is memory bits 
    lastMove = ""
    #if it is round 1 or 2 return the default bits which are in the end of the chromosomes. 
    if(round <=2):
        if(round ==1):
            return individual1[-2] #return default bit for the first round 
        else:
            return individual1[-1] #return default bit for the second round 
    else:    
        for i in range(2):
             lastMove+= str(individual1[-2+i]) + str(individual2[-2+i]) #get individuals move based on their memory bits, turn  it to a bit string
    lastMove = binaryToDecimal(lastMove) #changes bit string to decimal
    return individual1[lastMove]

#This function is used update the bits of the individual
"""
	 * @param individual1: 
             The chromosome of individual1 ⊆ [(0,1)*] 
	 * @param memory depth: The memory depth of the chromosomes at which the game is being played ⊆ (2)
	 Only the last bit of the memory depth is changed. The last bit (previous move) is shifted left while the second last bit is "forgotten" 
     * @return the updated individual
     * @return null if the individual's chromosome 
"""

def process_move(individual, move, memory_depth):
    if(len(individual)<19):
        return null #Individual's memory bits hasn't been initialised yet
    individual[18] = individual[19] 
    individual[19] = move
    return individual

"""
	 * @param individual1: 
             The chromosome of individual ⊆ [(0,1)*] 
	 * @param testSubjects: Contains a list of individuals in the population of the gentic algorithm
	 The function plays a game against the individual vs each individual in the population
     * @return the overall score of the individual after playing the game against each individual in the population
"""
# Evaluation function
def eval_func(individual,testSubjects,game):
    sum=0
    for ind in testSubjects:
        if(ind==individual):
            continue
        sum+=payoff_to_ind1(individual, ind, game)
    return sum,

#The functions below are used to scale each individual fitness value
"""
	 * @params c,f,fMax,fMin,fAvg: c is the amount of times the strongest individual is allowed to reproduce 
     it is always 2 for our case
     f is the invidiuals fitness value 
     fMax is the fitness value of the strongest individual 
     fMin is the fitness value of the weakest individual 
     fAvg is the average fitness values of the individuals
	 * @return the scaled fitness value 
     (refer paper_to_consult section 4.3 for more details)
"""

def scaleFPrime(c,f,fMax,fAvg):
    a = (c-1) * (fAvg/(fMax-fAvg))
    b = fAvg * ((fAvg-c*fAvg)/(fMax-fAvg))
    
    return a*f + b
    
def extremeScaleFPrime(c,f,fAvg,fMin):
    a = fAvg/(fAvg-fMin)
    b= fMin* (fAvg/(fAvg-fMin))
    return a*f + b
    
    
def scaleFitness(c,f,fMax,fAvg,fMin):
    if ((fMax-fAvg)<=1 or (fMin<0) or (fAvg-fMin)<=1):
        fPrime = extremeScaleFPrime(c,f,fAvg,fMin)
    else:
        fPrime = scaleFPrime(c,f,fMax,fAvg)
    
    return fPrime, 
    
        
    

# Create the toolbox with the right parameters
def create_toolbox():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Initialize the toolbox
    toolbox = base.Toolbox()

    # Generate attributes 
    toolbox.register("attr_bool", random.randint, 0, 1)

    # Initialize structures
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
        toolbox.attr_bool, 18)

    # Define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    

    # Register the crossover operator
    toolbox.register("mate", tools.cxOnePoint)

    # Register a mutation operator
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.001)

    # Operator for selecting individuals for breeding
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox
def dereference_toolbox(toolbox):
    # Generate attributes 
    
    toolbox.unregister("attr_bool")

    # Initialize structures
    toolbox.unregister("individual")

    # Define the population to be a list of individuals
    toolbox.unregister("population")
    

    # Register the crossover operator
    toolbox.unregister("mate")

    # Register a mutation operator
    toolbox.unregister("mutate")

    # Operator for selecting individuals for breeding
    toolbox.unregister("select")
    
    
def runGenticAlgorithm(game):
    print("Running Genetic Algorithm for " + game) 
    # Create a toolbox 
   
    # Seed the random number generator
    random.seed(7)

    # Create an initial population of 50 individuals
    population = toolbox.population(n=50)

    # Define probabilities of crossing and mutating
    probab_crossing, probab_mutating  = 0.7, 0.001

    # Define the number of generations
    num_generations = 50
    
    print('\nStarting the evolution process')
    
    # Evaluate the entire population
    
    fitnesses = list(map(lambda x: eval_func(x,population,game) , population)) 

    
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
        
    print('\nEvaluated', len(population), 'individuals')
    if(game=="IPD"):
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        fAvg = sum(fits) / length
        fMin = min(fits)
        fMax = max(fits)
        c=2
    
        for ind, fit in zip(population, fits):
            ind.fitness.values = scaleFitness(c,fit,fMax,fAvg,fMin)
   
    # Iterate through generations
    for g in range(num_generations):
        print("\n===== Generation", g)
        
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            
            # Cross two individuals
            if random.random() < probab_crossing:
                toolbox.mate(child1, child2)

                # "Forget" the fitness values of the children
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            # Mutate an individual
            if random.random() < probab_mutating:
                toolbox.mutate(mutant)
                del mutant.fitness.values
       
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(lambda x: eval_func(x,invalid_ind,game), invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        #Scale fitness values of individuals playing prisoners dillema 
        if(game == "IPD"):
            fits = [ind.fitness.values[0] for ind in invalid_ind]
            length = len(population)
            fAvg = sum(fits) / length
            fMin = min(fits)
            fMax = max(fits)
            c=2
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = scaleFitness(c,fit,fMax,fAvg,fMin)
      
    
        
        print('Evaluated', len(invalid_ind), 'individuals')
        
        # The population is entirely replaced by the offspring
        population[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]
        
        length = len(population)
        mean = sum(fits) / length
        
        print('Min =', min(fits), ', Max =', max(fits))
        print('Average =', round(mean, 2))
        best_ind = tools.selBest(population, 1)[0]
        print('\nBest individual:\n', best_ind)
        
    
    print("\n==== End of evolution")
    
    best_ind = tools.selBest(population, 1)[0]
    print('\nBest individual for ' + game + ':\n', best_ind)
  
toolbox = create_toolbox()
    
if __name__ == "__main__":
    runGenticAlgorithm("IPD")
    runGenticAlgorithm("IGC")
 