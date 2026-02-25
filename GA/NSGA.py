import random
from typing import List
from GA.individuals import Individual
from tqdm import tqdm

class NSGA2: 
    
    def __init__(self, 
                 population: List[Individual]):
        self.population = population


    def dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """
        Check if ind1 dominates ind2 based on fitness scores.
        For MAXIMIZATION objectives (as in the paper - accuracy maximization):
        ind1 dominates ind2 if:
        - ind1 is at least as good as ind2 in all objectives, AND
        - ind1 is strictly better than ind2 in at least one objective
        """
        better_in_at_least_one = False
        at_least_as_good_in_all = True

        for f1, f2 in zip(ind1.obj_vector, ind2.obj_vector):
            if f1 < f2:
                # ind1 is worse in this objective → cannot dominate
                at_least_as_good_in_all = False
                return False
            elif f1 > f2:
                # ind1 is strictly better in this objective
                better_in_at_least_one = True

        # ind1 dominates ind2 if it's better in at least one objective
        # and at least as good in all others
        return better_in_at_least_one and at_least_as_good_in_all

    
    def look_up_table(self):
        """
        Create a lookup table for dominance relationships.
        This optimization avoids recalculating dominance repeatedly.
        """
        look_up = {}
        for i in range(len(self.population)):
            look_up[i] = {}
            for j in range(len(self.population)):
                if i != j:
                    look_up[i][j] = self.dominates(self.population[i], self.population[j])
                else:
                    look_up[i][j] = False
        return look_up
    
    
    def non_dominated_sort(self):
        """
        Perform fast non-dominated sorting on the population.
        This is the core NSGA-II sorting algorithm.
        """
        look_up = self.look_up_table()
        n = len(self.population)
        
        # Initialize structures
        domination_count = [0] * n  # How many individuals dominate this one
        dominated_solutions = [[] for _ in range(n)]  # Which individuals this one dominates
        
        # Count dominators and dominated for each individual
        for i in range(n):
            for j in range(n):
                if i != j:
                    if look_up[i][j]:  # i dominates j
                        dominated_solutions[i].append(j)
                    elif look_up[j][i]:  # j dominates i
                        domination_count[i] += 1
        
        # First front: individuals not dominated by anyone
        current_front = []
        for i in range(n):
            if domination_count[i] == 0:
                self.population[i].NSGA_rank = 0
                current_front.append(i)
        
        # Build subsequent fronts
        front_num = 0
        while current_front:
            next_front = []
            for i in current_front:
                # For each individual dominated by i, reduce its domination count
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        self.population[j].NSGA_rank = front_num + 1
                        next_front.append(j)
            
            current_front = next_front
            front_num += 1
    
    
    def crowding_distance(self): 
        """
        Calculate crowding distance for each individual in the population.
        Crowding distance measures how close an individual is to its neighbors.
        Individuals with larger crowding distance are preferred to maintain diversity.
        """
        # Initialize all crowding distances to 0
        for ind in self.population:
            ind.NSGA_crowding_distance = 0.0
        
        num_objectives = len(self.population[0].obj_vector)
        n = len(self.population)
        
        # Calculate crowding distance for each objective
        for obj_index in range(num_objectives):
            # Sort individuals based on the current objective
            self.population.sort(key=lambda ind: ind.obj_vector[obj_index])
            
            # Get min and max values for normalization
            min_obj = self.population[0].obj_vector[obj_index]
            max_obj = self.population[-1].obj_vector[obj_index]
            
            # Assign infinite distance to boundary individuals
            # This ensures they are always selected (preserving extreme points)
            self.population[0].NSGA_crowding_distance = float('inf')
            self.population[-1].NSGA_crowding_distance = float('inf')
            
            # Calculate crowding distance for intermediate individuals
            if max_obj - min_obj == 0:
                # All individuals have the same value for this objective
                # No need to add distance (already initialized to 0)
                continue
            
            for i in range(1, n - 1):
                prev_obj = self.population[i - 1].obj_vector[obj_index]
                next_obj = self.population[i + 1].obj_vector[obj_index]
                
                # Normalized distance to neighbors
                distance = (next_obj - prev_obj) / (max_obj - min_obj)
                
                # Add to cumulative crowding distance
                self.population[i].NSGA_crowding_distance += distance


    def binary_tournament_selection(self) -> Individual:
        """
        Select one parent using binary tournament selection.
        Compares 2 randomly selected individuals using NSGA-II criteria:
        1. Lower rank (better front) wins
        2. If same rank, higher crowding distance wins (more isolated = better diversity)
        """
        # Randomly select 2 individuals
        ind1, ind2 = random.sample(self.population, 2)
        
        # Compare using NSGA-II criteria
        # 1. Lower rank wins (rank 0 is the best front)
        if ind1.NSGA_rank < ind2.NSGA_rank:
            return ind1
        elif ind2.NSGA_rank < ind1.NSGA_rank:
            return ind2
        else:
            # 2. Same rank: higher crowding distance wins (prefer diversity)
            if ind1.NSGA_crowding_distance > ind2.NSGA_crowding_distance:
                return ind1
            else:
                return ind2
    
    
    def select_population(self, target_size: int) -> List[Individual]:
        """
        Select target_size individuals from the population.
        This is used to reduce combined parent+offspring population back to original size.
        
        Process:
        1. Sort by non-dominated fronts
        2. Add complete fronts until we can't fit another complete front
        3. For the last front, sort by crowding distance and take the most isolated individuals
        """
        # Perform non-dominated sorting and crowding distance calculation
        self.non_dominated_sort()
        self.crowding_distance()
        
        # Sort population by rank, then by crowding distance
        self.population.sort(key=lambda ind: (ind.NSGA_rank, -ind.NSGA_crowding_distance))
        
        new_population = []
        current_front = 0
        
        while len(new_population) < target_size:
            # Get all individuals in current front
            front_individuals = [ind for ind in self.population if ind.NSGA_rank == current_front]
            
            if not front_individuals:
                break
            
            # If adding entire front doesn't exceed target, add all
            if len(new_population) + len(front_individuals) <= target_size:
                new_population.extend(front_individuals)
                current_front += 1
            else:
                # Sort current front by crowding distance (descending)
                front_individuals.sort(key=lambda ind: ind.NSGA_crowding_distance, reverse=True)
                
                # Add individuals with largest crowding distance until we reach target
                remaining = target_size - len(new_population)
                new_population.extend(front_individuals[:remaining])
                break
        
        return new_population
    
    
    def run_selection_for_mating(self, num_selections: int = None) -> List[Individual]:
        """
        Prepare population for mating by running NSGA-II selection.
        
        Args:
            num_selections: Number of individuals to select. If None, selects same size as population.
        
        Returns:
            List of selected individuals ready for mating
        """
        if num_selections is None:
            num_selections = len(self.population)
        
        # Run non-dominated sorting and crowding distance
        self.non_dominated_sort()
        self.crowding_distance()
        
        # Perform tournament selection
        selected = []
        for _ in range(num_selections):
            selected.append(self.binary_tournament_selection())
        
        return selected