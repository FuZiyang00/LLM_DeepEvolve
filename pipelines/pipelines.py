import json
import os
import random
import pandas as pd
from GA.individuals import Individual
from GA.genetic_algorithm import GeneticAlgorithm
from GA.SPEA import SPEA2
from GA.NSGA import NSGA2
from pipelines.utils import validate_code, create_contrastive_policy_seed, data_pipeline
import asyncio
from typing import List


async def create_single_child(GA: GeneticAlgorithm,
                              parent_1: Individual,
                              parent_2: Individual,
                              seed_block_name: str,
                              seed_block_code: str):
    
    crossover_code, p1_block_name = await GA.crossover(parent_1, parent_2)
    validate_code(p1_block_name, crossover_code)
    child = parent_1.clone()
    child.gene_blocks[p1_block_name].mutate(new_code=crossover_code)
    network = child.get_network()  # Validate the child's network

    if network is None:
        print("Child network creation failed.")
        child.is_valid = False
        return child

    improved_code = parent_1.gene_blocks[seed_block_name].code
    to_be_improved_code = child.gene_blocks[seed_block_name].code
    mutated_block_code = await GA.mutation(block_code = to_be_improved_code, 
                                           og_code = seed_block_code, 
                                           improved_code = improved_code)

    validate_code(seed_block_name, mutated_block_code)
    child.gene_blocks[seed_block_name].mutate(new_code=mutated_block_code)
    network = child.get_network()   
    if network is None:
        print("Child network creation after mutation failed.")
        child.is_valid = False
    else:
        print(f"New child created from parents {parent_1.index} and {parent_2.index}.")
        child.is_valid = True
        del network

    return child


async def create_batch_children(GA: GeneticAlgorithm,
                                NSGA2_selector,
                                seed_block_name: str,
                                seed_block_code: str,
                                num_children: int) -> List[Individual]:
    
    async def create_child():
        parent_1 = NSGA2_selector.binary_tournament_selection()
        parent_2 = NSGA2_selector.binary_tournament_selection()
        child = await create_single_child(GA, parent_1, parent_2, seed_block_name, seed_block_code)
        return child
    
    tasks = [create_child() for _ in range(num_children)]
    results = await asyncio.gather(*tasks)
    return results
    


def GA_pipeline(data: pd.DataFrame,
                population_size=10,
                generations=10,
                elite_size=5,
                yaml_file='prompts.yaml', 
                population=None): 
    
    training_dataloader, validation_dataloader, input_dim = data_pipeline(data)
    
    initial_seed = create_contrastive_policy_seed(input_dim)
    og_block_name, og_block_code = initial_seed.get_random_block()

    if population is None:
        GA = GeneticAlgorithm(input_dim=input_dim, 
                              population_size=population_size,
                              generations=generations,
                              yaml_file=yaml_file)
        
        GA.seed_individual = initial_seed  
        
        GA.generate_initial_population(initial_seed)
    else:
        GA = GeneticAlgorithm(input_dim=input_dim, 
                              population_size=population_size,
                              generations=generations,
                              yaml_file=yaml_file)
        GA.population = population
        for i, ind in enumerate(GA.population):
            ind.index = i  # assign a stable index to each individual

    print(f"Initial population of {len(GA.population)} individuals created.")
    logs = {}
    elites = []
    log_dir = "logs"

    for gen in range(generations):
        gen_logs = GA.evaluate_population(training_dataloader, validation_dataloader)
        logs[f'Generation_{gen+1}'] = gen_logs

        log_path = os.path.join(log_dir, f"generation_{gen+1}.json")
        with open(log_path, "w") as f:
            json.dump(gen_logs, f, indent=4)
        
        # SPEA2 selection to select elites
        spea_selector = SPEA2(population=GA.population, archive_size=elite_size)
        spea_selector.strengths()
        spea_selector.raw_fitness()
        spea_selector.density_numpy()
        elites = [ind for ind in GA.population if ind.SPEA_raw_fitness == 0]

        if len(elites) < elite_size:
            remaining_slots = elite_size - len(elites)
            dominated = [ind for ind in GA.population if ind.SPEA_raw_fitness > 0]
            dominated.sort(key=lambda x: x.SPEA_raw_fitness)
            elites.extend(dominated[:remaining_slots])
        
        elif len(elites) > elite_size:
            elites = spea_selector.truncate_archive(elites)

        selected_indices = [ind.index for ind in elites]
        print(f"Generation {gen+1} evaluated. Elites selected: {len(elites)} \n {selected_indices}")

        # Create offspring to refill population
        NSGA2_selector = NSGA2(population=GA.population)
        NSGA2_selector.non_dominated_sort()
        NSGA2_selector.crowding_distance()

        remaining_slots = population_size - len(elites)
        max_attempts = 5
        while remaining_slots > 0 and max_attempts > 0:

            batch_children = asyncio.run(create_batch_children(GA,
                                                               NSGA2_selector,
                                                               og_block_name,
                                                               og_block_code,
                                                               num_children=remaining_slots))
            max_attempts -= 1
            for ind in batch_children:
                if ind.is_valid:
                    elites.append(ind)
                    remaining_slots -= 1
                    if len(elites) >= population_size:
                        break
            if max_attempts == 0 and remaining_slots > 0:
                print(f"Could not create enough valid children after multiple attempts. Filling remaining slots with copies of elites.")
                while remaining_slots > 0:
                    random_elite = random.choice(elites)
                    elite_copy = random_elite.clone()
                    elite_copy.index = len(elites)
                    elites.append(elite_copy)
                    remaining_slots -= 1

                    
        print(f"Generation {gen+1} completed. Population refilled to {len(elites)} individuals.")

        GA.clear_population()
        GA.population = elites
        for i, ind in enumerate(GA.population):
            ind.index = i  # assign a stable index to each individual

    return logs, GA.population
        