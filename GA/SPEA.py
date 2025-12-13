from typing import List
from GA.individuals import Individual
import numpy as np
import math

class SPEA2:

    def __init__(self, 
                 population: List[Individual],
                 archive_size: int):
        self.population = population
        self.archive_size = archive_size


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
    

    def strengths(self):
        """Calculate the strength of each individual in the population"""
        self.look_up = self.look_up_table()
        
        for i, ind in enumerate(self.population):
            ind.SPEA_strength = sum(1 for j in self.look_up[i] if self.look_up[i][j])


    def raw_fitness(self):
        """Calculate the raw fitness of each individual"""
        for i, ind in enumerate(self.population):
            dominated_by = [
                j for j in self.look_up
                if self.look_up[j].get(i, False)  # safe: returns False if key i is missing
            ]
            ind.SPEA_raw_fitness = sum(self.population[j].SPEA_strength for j in dominated_by)


    def density_numpy(self):
        """
        Calculate density using numpy for better performance.
        Requires: import numpy as np
        """
        import numpy as np
        
        n = len(self.population)
        k = int(math.sqrt(n))
        
        if k >= n:
            k = n - 1
        if k < 1:
            k = 1
        
        # Create matrix of objective vectors (n x m)
        obj_matrix = np.array([ind.obj_vector for ind in self.population])
        
        for i, ind in enumerate(self.population):
            # Calculate distances to all individuals at once
            # Broadcasting: subtract individual i from all others
            diff = obj_matrix - obj_matrix[i]
            
            # Euclidean distance: sqrt(sum of squared differences)
            distances = np.sqrt(np.sum(diff ** 2, axis=1))
            
            # Remove distance to self (which is 0)
            distances = distances[distances > 0]
            
            # Sort and get k-th nearest neighbor
            distances.sort()
            sigma_k = distances[k - 1]
            
            # Calculate density
            ind.SPEA_density = 1.0 / (sigma_k + 2.0)

    
    def truncate_archive(self, archive):
        """
        Truncate the archive to size N_a using SPEA2's density-based operator.
        Assumes each individual has .obj_vector and .SPEA_density attributes.
        """
        
        while len(archive) > self.archive_size:
            # Find most crowded individual (highest density)
            worst = max(archive, key=lambda ind: ind.SPEA_density)
            archive.remove(worst)
            
            # (Optional) Recompute density here for better accuracy
            # self.population = archive
            # self.density_numpy()

        return archive

