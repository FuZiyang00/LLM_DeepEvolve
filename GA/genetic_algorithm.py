from GA.individuals import Individual
from typing import List
import yaml
import random
from torch.utils.data import DataLoader
import json
from LLM.agent import BaseLLMAgent
from config import settings
import asyncio
from GA.SPEA import SPEA2
from pipelines.utils import validate_code, setup_logger, validate_and_add, parse_full_genome_response
# ============================================================================
# 2. GENETIC ALGORITHM CLASS
# ============================================================================

logger = setup_logger(__name__)

class GeneticAlgorithm:
    def __init__(self,
                 input_dim: int, 
                 population_size: int, 
                 ):
        
        self.input_dim = input_dim
        self.population_size = population_size
        self.population = []

        prompts_archive = {
            "constraints_prompt": settings.prompts_path / "constraints.yml",
            "crossover_prompt": settings.prompts_path / "crossover.yml",
            "expert_roles_prompt": settings.prompts_path / "expert_roles.yml",
            "mutation_prompt": settings.prompts_path / "mutation.yml",
            "premise_prompt": settings.prompts_path / "premise.yml",
            "error_prompt": settings.prompts_path / "errors.yml",
            "repair_prompt": settings.prompts_path / "repair_full_genome.yml"
        }

        for attr_name, path in prompts_archive.items():
            with open(path, "r", encoding="utf-8") as f:
                setattr(self, attr_name, yaml.safe_load(f) or " ")
    
        self.llm_agent = BaseLLMAgent()

    
    def assign_index(self): 
        for idx, ind in enumerate(self.population):
            ind.index = idx

    
    async def _create_one_child(self, 
                                initial_seed: Individual) -> Individual:
        """Create a single child by cloning the seed and mutating one random block."""

        block_name, block_code = initial_seed.get_random_block()
        
        mutated_individual: Individual = await self.mutation(original_ind=initial_seed,
                                                                genome_name=block_name,
                                                                block_code=block_code,
                                                                eot_prob=0.1)
        return mutated_individual
    
    
    async def generate_initial_population(self, 
                                          initial_seed: Individual) -> List[Individual]:
        """Function to generate the initial population"""

        self.population.append(initial_seed)
        self.seed_individual: Individual = initial_seed

        # Generate population_size - 1 children concurrently
        n_children = self.population_size - 1
        
        children = await asyncio.gather(
            *[self._create_one_child(initial_seed) for _ in range(n_children)])
    
        for c in children:
            validate_and_add(child=c, 
                             parent_1=initial_seed, 
                             offsprings=self.population)

        self.assign_index()
    

    def save_population(self, generation_index):
        generation_logs = {}
        for i, individual in enumerate(self.population):
            generation_logs[f'Individual_{i+1}'] = {
                                                    "fitness_scores": individual.fitness_scores,
                                                    "obj_vector": individual.obj_vector,   
                                                    "individual": individual.to_dict()    
                                                    }
        
        log_path = settings.logs_path / f"generation_{generation_index}.json"
        with open(log_path, "w") as f:
            json.dump(generation_logs, f, indent=4)
            

    # Evaluate the entire population
    async def evaluate_population(self, 
                                    training_dataloader: DataLoader,
                                    validation_dataloader: DataLoader):
        
        for i, individual in enumerate(self.population):
            
            if individual.obj_vector: # don't retrain already trained networks
                continue 
            
            desc = f"Evaluating Individual {i+1} | Training epochs"
            error_msg = individual.fitness_evaluation(training_dataloader, validation_dataloader, desc=desc)
            
            if not error_msg: continue
            
            elif error_msg:
                new_individual = await self.repair_after_training_failure(
                    individual, error_msg, max_attempts=3
                )
                if new_individual:
                    retry_error = new_individual.fitness_evaluation(training_dataloader, 
                                                                    validation_dataloader, desc=desc)

                    if not retry_error:
                        self.population[i] = new_individual
                        new_individual.index = i  # keep index in sync
                        logger.info("Repairement successful!")

                    elif retry_error:
                        logger.info("Repairement failed")

                elif new_individual is None:
                    logger.info("Repairement failed")
        
        for individual in self.population:
            if not individual.obj_vector:
                raise ValueError("All individuals must have fitness scores before selection.")
            

    async def repair_after_training_failure(self, 
                                            invalid_individual: Individual, 
                                            error_message: str, 
                                            max_attempts: int):
        
        new_individual = None
        template = self.repair_prompt["Repair_Full_Genome"]
        input_dim = invalid_individual.input_dim

        # randomly selecting the personality 
        role = random.choice(self.expert_roles_prompt["expert_roles"])
        role_description = role["description"]

        # formatting premise 
        premise = self.premise_prompt["Premise"].format(expert_roles=role_description)
        constraints = self.constraints_prompt["Constraint"]
        system_prompt = premise + constraints

        user_content = template.format(
            error_message = error_message,
            encoder_code = invalid_individual.gene_blocks["Encoder"].code,
            normalizer_code = invalid_individual.gene_blocks["Normalizer"].code,
            contrastive_policy_code = invalid_individual.gene_blocks["ContrastivePolicyNetwork"].code
        )

        for attempt in range(max_attempts): 
            messages = [{"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": user_content}]
            response_code = await self.llm_agent._run(messages=messages, temperature=0)
            new_individual = parse_full_genome_response(response=response_code, 
                                                        input_dim=input_dim)
            
            if new_individual:
                return new_individual

            else:
                logger.info(f"Repairement failed at attempt {attempt}")
                continue
        
        return new_individual
        
        
    
    # Mutation function using LLM
    async def mutation(self,
                       original_ind: Individual, 
                       genome_name: str,
                       block_code: str, og_code: str = None, improved_code: str = None,
                       eot_prob: float = random.random()):

        # randomly selecting the personality 
        role = random.choice(self.expert_roles_prompt["expert_roles"])
        role_name = role["name"]
        role_description = role["description"]
        role_temperatature = role["temperature"]

        # formatting premise 
        premise = self.premise_prompt["Premise"].format(expert_roles=role_description)
        constraints = self.constraints_prompt["Constraint"]
        system_prompt = premise + constraints

        # Max LLM attempts 
        max_attempts = 5
        mutated_child = None

        if eot_prob <= 0.5:

            plain_mutations = self.mutation_prompt["Plain_Mutation"]
            chosen = next(m for m in plain_mutations if m["name"] == role_name)
            mutation = chosen["description"]
            mutation_prompt = mutation.format(code_block = block_code)
            
            for i in range(max_attempts): 
                messages = [
                            {"role": "system", "content": system_prompt}, 
                            {"role": "user", "content": mutation_prompt}
                        ]
                
                response_code = await self.llm_agent._run(messages=messages, 
                                                          temperature=role_temperatature)
                
                valid, error = validate_code(genome_name, response_code)

                if valid and not error:
                    mutated_child: Individual = original_ind.clone()
                    mutated_child.gene_blocks[genome_name].mutate(new_code=response_code)
                    network = mutated_child.get_network()

                    if isinstance(network, str):
                        error_prompt = self.error_prompt["Errors"]
                        formatted_error_prompt = error_prompt.format(error_source=response_code, errors=network)
                        logger.info(f"[MUTATION]: Invalid mutation attempt {i} due to: \n {network}")
                        mutation_prompt+=formatted_error_prompt
                        continue                    
                
                    else:
                        mutated_child.is_valid = True
                        del network
                        logger.info(f"[MUTATION]: child created from individual {original_ind.index}")
                        return mutated_child
                
                elif not valid and error:
                    error_prompt = self.error_prompt["Errors"]
                    formatted_error_prompt = error_prompt.format(error_source=response_code, errors=error)
                    logger.info(f"[MUTATION]: Invalid mutation attempt {i} due to: \n {error}")
                    mutation_prompt+=formatted_error_prompt
                    continue

        else: 
            eot_template = self.mutation_prompt["EoT_mutation"]
            formatted_eot_prompt = eot_template.format(block_X_seed=og_code,        
                                                       block_X_elite=improved_code, 
                                                       block_Y=block_code)
            
            for i in range(max_attempts):
                
                messages = [
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": formatted_eot_prompt}
                ]

                response_code = await self.llm_agent._run(messages=messages, temperature=role_temperatature)
                valid, error = validate_code(genome_name, response_code)

                if valid and not error:
                    mutated_child: Individual = original_ind.clone()
                    mutated_child.gene_blocks[genome_name].mutate(new_code=response_code)
                    network = mutated_child.get_network()

                    if isinstance(network, str):
                        error_prompt = self.error_prompt["Errors"]
                        formatted_error_prompt = error_prompt.format(error_source=response_code, errors=network)
                        logger.info(f"[MUTATION]: Invalid mutation attempt {i} due to: \n {network}")
                        formatted_eot_prompt+=formatted_error_prompt
                        continue 

                    else: 
                        del network
                        logger.info(f"[MUTATION]: child created from individual {original_ind.index}")
                        return mutated_child
                
                elif not valid and error:
                    error_prompt = self.error_prompt["Errors"]
                    formatted_error_prompt = error_prompt.format(error_source=response_code, errors=error)
                    logger.info(f"[MUTATION]: Invalid mutation attempt {i} due to: \n {error}")
                    formatted_eot_prompt+=formatted_error_prompt
                    continue
            
        return mutated_child
        

    # Crossover function using LLM
    async def crossover(self, 
                        parent1: Individual, 
                        parent2: Individual) -> Individual:
        
        # randomly selecting the personality 
        role = random.choice(self.expert_roles_prompt["expert_roles"])
        role_description = role["description"]
        role_temperatature = role["temperature"]

        # formatting premise 
        premise = self.premise_prompt["Premise"].format(expert_roles=role_description)
        constraints = self.constraints_prompt["Constraint"]
        system_prompt = premise + constraints
        
        # Single point crossover
        crossover_prompt = self.crossover_prompt["Crossover"]
        p1_block_name, p1_block_code = parent1.get_random_block()
        p2_block_code = parent2.gene_blocks[p1_block_name].code

        formatted_crossover_prompt = crossover_prompt.format(code_block_1=p1_block_code, 
                                                             code_block_2=p2_block_code)
        
        max_attempts = 5
        child = None
        for i in range (max_attempts): 
            messages = [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": formatted_crossover_prompt}
            ]
            
            response_code = await self.llm_agent._run(messages=messages, 
                                                        temperature=role_temperatature)
            
            valid, error = validate_code(p1_block_code, response_code)

            if valid and not error:
                child = parent1.clone()
                child.gene_blocks[p1_block_name].mutate(new_code=response_code)
                network = child.get_network()

                if isinstance(network, str):
                    error_prompt = self.error_prompt["Errors"]
                    formatted_error_prompt = error_prompt.format(error_source=response_code, errors=network)
                    logger.info(f"[CROSSOVER] Invalid crossover attempt {i} due to: \n {network}")
                    formatted_crossover_prompt+=formatted_error_prompt
                    continue 

                else:
                    logger.info(f"[CROSSOVER]: New child created from parents {parent1.index} and {parent2.index}.")
                    del network
                    return child
            
            elif not valid and error:
                error_prompt = self.error_prompt["Errors"]
                formatted_error_prompt = error_prompt.format(error_source=response_code, errors=error)
                logger.info(f"[CROSSOVER] Invalid crossover attempt {i} due to: \n {error}")
                formatted_crossover_prompt+=formatted_error_prompt
                continue

        return child  
    
    
    def clear_population(self):
        self.population = []

    
    def set_new_population(self, new_population):
        self.population = new_population
        self.assign_index()


    @staticmethod
    def difference_with_seed(individual: Individual, 
                             seed: Individual) -> int:
        """Calculate the difference between an individual and the seed individual"""
        random_element = None
        differences = []
        for block_name in seed.gene_blocks:
            if block_name in individual.gene_blocks:
                if seed.gene_blocks[block_name].code != individual.gene_blocks[block_name].code:
                    differences.append(block_name)

        if differences: 
            random_element = random.choice(differences)
        return random_element
    

    @staticmethod
    def selection(elite_archive:List[Individual],
                  population_size: int, 
                  total_population: List[Individual]): 
        
        remaining_slots = population_size - len(elite_archive)
        elite_idx = [ind.index for ind in elite_archive]

        # 1. filter to only the non elite individuals
        non_elites = [ind for ind in total_population if ind.index not in elite_idx]

        # 2. select the best ones in the remaining.
        selection_spea = SPEA2(population=non_elites, 
                                archive_size=remaining_slots)
        top_non_elites = selection_spea.environmental_selection()

        # 3. set new population
        new_population = list(elite_archive)
        new_population.extend(top_non_elites)
        return new_population
            
