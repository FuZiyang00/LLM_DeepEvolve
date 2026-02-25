from typing import List
from GA.individuals import Individual
import numpy as np
import math

class SPEA2:
    """
    Strength Pareto Evolutionary Algorithm 2 (SPEA-2)
    
    Used in the paper for elite selection to preserve top-performing solutions
    while ensuring diversity through fine-grained fitness assignment and 
    k-th nearest neighbor density estimation.
    """

    def __init__(self, 
                 population: List[Individual],
                 archive_size: int):
        
        self.population = population
        self.archive_size = archive_size
        self.look_up = None

    @staticmethod
    def dominates(ind1: Individual, ind2: Individual) -> bool:
        """
        Check if ind1 dominates ind2 based on fitness scores.
        For MAXIMIZATION objectives:
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
    

    def calculate_strength(self):
        """
        Calculate the strength S(i) of each individual in the population.
        
        Strength S(i) = number of individuals that i dominates
        
        This represents the raw dominance power of an individual.
        """
        self.look_up = self.look_up_table()
        
        for i, ind in enumerate(self.population):
            # Count how many individuals this one dominates
            ind.SPEA_strength = sum(1 for j in self.look_up[i] if self.look_up[i][j])


    def calculate_raw_fitness(self):
        """
        Calculate the raw fitness R(i) of each individual.
        
        Raw fitness R(i) = sum of strengths of all individuals that dominate i
        
        Lower raw fitness is better (0 means non-dominated).
        This is the "fine-grained fitness assignment" mentioned in the paper.
        """
        for i, ind in enumerate(self.population):
            # Find all individuals that dominate this one
            dominated_by = [
                j for j in range(len(self.population))
                if j != i and self.look_up[j][i]
            ]
            # Sum their strengths
            ind.SPEA_raw_fitness = sum(self.population[j].SPEA_strength for j in dominated_by)


    def calculate_density(self):
        """
        Calculate density D(i) using k-th nearest neighbor distance.
        
        This is the "k-th nearest neighbor density estimation" mentioned in the paper.
        Density helps distinguish individuals with the same raw fitness.
        
        Lower density means more isolated (better for diversity).
        """
        n = len(self.population)
        k = int(math.sqrt(n))
        
        # Ensure k is valid
        if k >= n:
            k = n - 1
        if k < 1:
            k = 1
        
        # Create matrix of objective vectors (n x m)
        obj_matrix = np.array([ind.obj_vector for ind in self.population])
        
        for i, ind in enumerate(self.population):
            # Calculate Euclidean distances to all other individuals
            diff = obj_matrix - obj_matrix[i]
            distances = np.sqrt(np.sum(diff ** 2, axis=1))
            
            # Remove distance to self (which is 0)
            distances = distances[distances > 0]
            
            # Sort and get k-th nearest neighbor distance
            distances = np.sort(distances)
            
            if len(distances) >= k:
                sigma_k = distances[k - 1]
            else:
                # If population is very small, use the furthest neighbor
                sigma_k = distances[-1] if len(distances) > 0 else 0.0
            
            # Calculate density (inverse of distance)
            # Adding 2 as per SPEA-2 specification to ensure it's always < 1
            ind.SPEA_density = 1.0 / (sigma_k + 2.0)


    def calculate_fitness(self):
        """
        Calculate final fitness F(i) for each individual.
        
        F(i) = R(i) + D(i)
        
        Where:
        - R(i) is raw fitness (sum of strengths of dominators)
        - D(i) is density (based on k-th nearest neighbor)
        
        Lower fitness is better. Non-dominated individuals have R(i) = 0.
        """
        self.calculate_strength()
        self.calculate_raw_fitness()
        self.calculate_density()
        
        for ind in self.population:
            ind.SPEA_fitness = ind.SPEA_raw_fitness + ind.SPEA_density


    def environmental_selection(self) -> List[Individual]:
        """
        Perform environmental selection to fill the archive.
        
        Steps:
        1. Calculate fitness for all individuals
        2. Copy all non-dominated individuals (F(i) < 1) to archive
        3. If archive is too small: fill with best dominated individuals
        4. If archive is too large: truncate using density
        
        Returns:
            Archive of size archive_size (or less if population is small)
        """
        # Calculate all fitness values
        self.calculate_fitness()
        
        # Step 1: Copy all non-dominated individuals to archive
        archive = [ind for ind in self.population if ind.SPEA_raw_fitness < 1]
        
        # Step 2: If archive is too small, fill with best dominated individuals
        if len(archive) < self.archive_size:
            # Get dominated individuals
            dominated = [ind for ind in self.population if ind.SPEA_raw_fitness >= 1]
            
            # Sort by fitness (lower is better)
            dominated.sort(key=lambda ind: ind.SPEA_fitness)
            
            # Add best dominated individuals until archive is full
            remaining = self.archive_size - len(archive)
            archive.extend(dominated[:remaining])
        
        # Step 3: If archive is too large, truncate using density
        elif len(archive) > self.archive_size:
            archive = self.truncate_archive(archive)
        
        return archive


    def truncate_archive(self, archive: List[Individual]) -> List[Individual]:
        """
        Truncate the archive to archive_size using SPEA-2's density-based operator.
        
        Iteratively removes the individual with the smallest distance to another individual.
        This is done by:
        1. Finding the individual with minimum distance to any other individual
        2. If there's a tie, use second-smallest distance, third-smallest, etc.
        3. Remove that individual and repeat
        
        Args:
            archive: List of individuals (will be modified)
            
        Returns:
            Truncated archive of size archive_size
        """
        archive = archive.copy()  # Don't modify the original list
        
        while len(archive) > self.archive_size:
            n = len(archive)
            
            # Create matrix of objective vectors
            obj_matrix = np.array([ind.obj_vector for ind in archive])
            
            # Calculate all pairwise distances
            distances = np.zeros((n, n))
            for i in range(n):
                diff = obj_matrix - obj_matrix[i]
                distances[i] = np.sqrt(np.sum(diff ** 2, axis=1))
            
            # For each individual, find sorted distances to all others
            min_distances = []
            for i in range(n):
                # Get distances to others (excluding self)
                dists = distances[i][distances[i] > 0]
                dists = np.sort(dists)
                min_distances.append(dists)
            
            # Find individual with smallest distance to another
            # Use lexicographic comparison: compare 1st nearest, then 2nd, etc.
            worst_idx = 0
            worst_dists = min_distances[0]
            
            for i in range(1, n):
                # Compare distance lists lexicographically
                if self._is_more_crowded(min_distances[i], worst_dists):
                    worst_idx = i
                    worst_dists = min_distances[i]
            
            # Remove most crowded individual
            archive.pop(worst_idx)
        
        return archive


    def _is_more_crowded(self, dists1, dists2) -> bool:
        """
        Compare two distance lists lexicographically.
        Returns True if dists1 represents a more crowded individual.
        
        More crowded means smaller distances to neighbors.
        """
        min_len = min(len(dists1), len(dists2))
        
        for i in range(min_len):
            if dists1[i] < dists2[i]:
                return True  # dists1 has smaller distance → more crowded
            elif dists1[i] > dists2[i]:
                return False
        
        # If all compared distances are equal, shorter list is more crowded
        return len(dists1) < len(dists2)


    def select_elite(self) -> List[Individual]:
        """
        Main method to select elite individuals using SPEA-2.
        This is what's called in the paper's evolutionary loop.
        
        Returns:
            List of elite individuals (archive)
        """
        return self.environmental_selection()


# Backwards compatibility methods (matching your original interface)
    def strengths(self):
        """Alias for calculate_strength (for backwards compatibility)"""
        return self.calculate_strength()
    
    def raw_fitness(self):
        """Alias for calculate_raw_fitness (for backwards compatibility)"""
        return self.calculate_raw_fitness()
    
    def density_numpy(self):
        """Alias for calculate_density (for backwards compatibility)"""
        return self.calculate_density()