import pandas as pd
from GA.individuals import Individual, Genome_block
from GA.genetic_algorithm import GeneticAlgorithm
from GA.SPEA import SPEA2
from GA.NSGA import NSGA2
from pipelines.utils import (validate_and_add,
                             create_contrastive_policy_seed, 
                             data_pipeline, select_random_dominator, 
                             setup_logger)
import asyncio
from typing import Optional, List

logger = setup_logger(__name__)


# ============================================================================
# 2. Function for running the whole evolution process
# ============================================================================
    
async def run_evolution(data: pd.DataFrame,
                  population_size: int = 10, 
                  generations: int = 10, 
                  elite_archive_size = 5, 
                  initial_population: Optional[List[Individual]] = None):

    training_dataloader, validation_dataloader, input_dim = data_pipeline(data)

    # init seed
    initial_seed: Individual = create_contrastive_policy_seed(input_dim)
    initial_seed.index = 0
    genetic_algorithm: GeneticAlgorithm = GeneticAlgorithm(input_dim=input_dim, 
                                        population_size=population_size,)
    
    if initial_population:
        # Use provided list as first generation
        genetic_algorithm.population = list(initial_population)
        genetic_algorithm.assign_index()
        logger.info(f"Using provided initial population of {len(genetic_algorithm.population)} individuals.")

    else: 
        # create initial population 
        await genetic_algorithm.generate_initial_population(initial_seed)

        # validating the children
        to_be_del = []
        for i in range(1, len(genetic_algorithm.population)):
            validate_and_add(child = genetic_algorithm.population[i], 
                            parent_1=initial_seed, 
                            offsprings=to_be_del) 
        
        del to_be_del

        logger.info(f"Initial population of {len(genetic_algorithm.population)} individuals created.")
        # evaluate initial population 
        await genetic_algorithm.evaluate_population(training_dataloader, validation_dataloader)
        genetic_algorithm.save_population(generation_index=0)

    for gen in range(12, generations):

        spea_2 = SPEA2(population=genetic_algorithm.population, 
                       archive_size=elite_archive_size)
        elite_archive = spea_2.environmental_selection()

        offsprings = []
        # crossover 
        for _ in range(0, 2, 2):
            nsga_2 = NSGA2(population=genetic_algorithm.population)

            parent_1: Individual = nsga_2.binary_tournament_selection()
            parent_2: Individual = nsga_2.binary_tournament_selection()

            parent_3: Individual = nsga_2.binary_tournament_selection()
            parent_4: Individual = nsga_2.binary_tournament_selection()

            async def run_crossovers():
                return await asyncio.gather(
                    genetic_algorithm.crossover(parent1=parent_1, parent2=parent_2),
                    genetic_algorithm.crossover(parent1=parent_3, parent2=parent_4),
                )

            child_1, child_2 = await run_crossovers()

            for c in (child_1, child_2):
                validate_and_add(child=c, parent_1= parent_1, offsprings=offsprings)

        # mutation
        mutated_offsprings = []
        for i in range(0, 2, 2):
            dominating_individual = select_random_dominator(initial_seed, elite_archive)

            if dominating_individual:
                logger.info("Using EoT Mutation")
                dom_genome_name = GeneticAlgorithm.difference_with_seed(individual=dominating_individual, 
                                                                        seed=initial_seed)
                if dom_genome_name:
                    dom_genome_code = dominating_individual.gene_blocks[dom_genome_name].code
                
                else: 
                    dom_genome_name, dom_genome_code = dominating_individual.get_random_block()
                
                seed_genome_code: Genome_block = initial_seed.gene_blocks[dom_genome_name].code 
                
                child_1: Individual = offsprings[i]
                child_1_genome_code: Genome_block = child_1.gene_blocks[dom_genome_name].code

                child_2: Individual= offsprings[i+1]
                child_2_genome_code: Genome_block = child_2.gene_blocks[dom_genome_name].code

                async def run_eot_mutation():
                    return await asyncio.gather(
                        genetic_algorithm.mutation(block_code=child_1_genome_code, 
                                                    og_code=seed_genome_code, 
                                                    improved_code=dom_genome_code, 
                                                    original_ind=child_1, 
                                                    genome_name=dom_genome_name),

                        genetic_algorithm.mutation(block_code=child_2_genome_code, 
                                                    og_code=seed_genome_code, 
                                                    improved_code=dom_genome_code, 
                                                    original_ind=child_2, 
                                                    genome_name=dom_genome_name),
                    )
                
                mutated_child_1, mutated_child_2 = await run_eot_mutation()
                
                for c in (mutated_child_1, mutated_child_2):
                    validate_and_add(child=c, parent_1= child_1, offsprings=mutated_offsprings)
            
            else: 
                child_1: Individual = offsprings[i]
                genome_name_1, genome_code_1 = child_1.get_random_block()
        
                child_2: Individual= offsprings[i+1]
                genome_name_2, genome_code_2= child_2.get_random_block()

                async def run_mutation():
                    return await asyncio.gather(
                        genetic_algorithm.mutation(original_ind=child_1, 
                                                    genome_name=genome_name_1, 
                                                    block_code=genome_code_1, 
                                                    eot_prob=0.1),

                        genetic_algorithm.mutation(original_ind=child_2, 
                                                    genome_name=genome_name_2, 
                                                    block_code=genome_code_2, 
                                                    eot_prob=0.1),)

                mutated_child_1, mutated_child_2 = await run_mutation()

                for c in (mutated_child_1, mutated_child_2):
                    validate_and_add(child=c, parent_1= child_1, offsprings=mutated_offsprings)

        genetic_algorithm.population.extend(mutated_offsprings)      
        genetic_algorithm.assign_index()
        await genetic_algorithm.evaluate_population(training_dataloader=training_dataloader, 
                                              validation_dataloader=validation_dataloader)
        
        # selection with SPEA2
        if len(genetic_algorithm.population) > population_size:
            new_population = genetic_algorithm.selection(elite_archive=elite_archive, 
                                                         population_size=population_size, 
                                                         total_population=genetic_algorithm.population)
            
            genetic_algorithm.set_new_population(new_population=new_population)
            genetic_algorithm.save_population(generation_index=gen)
