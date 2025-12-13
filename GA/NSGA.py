import random
from typing import List
from GA.individuals import Individual
from tqdm import tqdm

class NSGA2: 
    
    def __init__(self, 
                 population: List[Individual]):
        self.population = population


    def dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if ind1 dominates ind2 based on fitness scores"""
        better_in_at_least_one = False

        for f1, f2 in zip(ind1.obj_vector, ind2.obj_vector):
            if f1 < f2:
                # sol1 is worse in this objective → cannot dominate
                return False
            elif f1 > f2:
                # sol1 is strictly better in this objective
                better_in_at_least_one = True

        return better_in_at_least_one

    
    def look_up_table(self):
        look_up = {}
        # Outer loop with tqdm to track progress over population
        for i, ind in enumerate(self.population):
            look_up[i] = {}
            for j, other_ind in enumerate(self.population):
                if i != j:
                    look_up[i][j] = self.dominates(ind, other_ind)
        return look_up
    
    
    def non_dominated_sort(self):
        """Perform non-dominated sorting on the population"""
        look_up = self.look_up_table()
        
        remaining = list(range(len(self.population)))  # Work with indices directly
        front = 0
        
        while remaining:
            current_front = []
            
            for i in remaining:
                # Check if any other REMAINING individual dominates i
                dominated = any(look_up[j][i] for j in remaining if j != i)
                
                if not dominated:
                    self.population[i].NSGA_rank = front
                    current_front.append(i)
            
            if not current_front:
                print("Warning: Possible circular dominance or logic error")
                print(f"Remaining individuals: {len(remaining)}")
                # Debug: check what's happening
                for i in remaining:
                    dominators = [j for j in remaining if j != i and look_up[j][i]]
                    print(f"Individual {i} is dominated by: {dominators}")
                break
            
            # Remove current front from remaining
            for i in current_front:
                remaining.remove(i)
            
            front += 1
        
           
    
    def crowding_distance(self): 
        """Calculate crowding distance for each individual in the population"""
        num_objectives = len(self.population[0].obj_vector)
        for obj_index in range(num_objectives):
            # Sort individuals based on the current objective
            self.population.sort(key=lambda ind: ind.obj_vector[obj_index])
            min_obj = self.population[0].obj_vector[obj_index]
            max_obj = self.population[-1].obj_vector[obj_index]

            # Assign infinite distance to boundary individuals
            self.population[0].NSGA_crowding_distance = float('inf')
            self.population[-1].NSGA_crowding_distance = float('inf')

            # Calculate crowding distance for intermediate individuals
            for i in range(1, len(self.population) - 1):
                prev_obj = self.population[i - 1].obj_vector[obj_index]
                next_obj = self.population[i + 1].obj_vector[obj_index]
                if max_obj - min_obj == 0:
                    distance = 0.0
                else:
                    distance = (next_obj - prev_obj) / (max_obj - min_obj)
                if hasattr(self.population[i], 'NSGA_crowding_distance'):
                    self.population[i].NSGA_crowding_distance += distance
                else:
                    self.population[i].NSGA_crowding_distance = distance
        
        print("Crowding distance calculation completed.")
        


    def binary_tournament_selection(self):
        """
        Select one parent using binary tournament selection.
        Compares 2 randomly selected individuals.
        """
        # Randomly select 2 individuals
        ind1, ind2 = random.sample(self.population, 2)
        
        # Compare using NSGA-II criteria
        # 1. Lower rank wins
        if ind1.NSGA_rank < ind2.NSGA_rank:
            return ind1
        elif ind2.NSGA_rank < ind1.NSGA_rank:
            return ind2
        else:
            # 2. Same rank: higher crowding distance wins
            if ind1.NSGA_crowding_distance > ind2.NSGA_crowding_distance:
                return ind1
            else:
                return ind2
