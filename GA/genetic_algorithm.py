import json
from urllib import response
from GA.individuals import Individual
from typing import List
import yaml
from openai import AsyncOpenAI 
import asyncio
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from dotenv import load_dotenv
import re

# ============================================================================
# 2. GENETIC ALGORITHM CLASS
# ============================================================================

class GeneticAlgorithm:
    def __init__(self,
                 input_dim: int, 
                 population_size: int, 
                 generations: int,
                 yaml_file: str):
        
        self.input_dim = input_dim
        self.population_size = population_size
        self.generations = generations
        with open(yaml_file, 'r') as file:
            self.prompts = yaml.safe_load(file)

        load_dotenv()   
        api_key = os.getenv("OPENAI_API_KEY") 
        self.openai = AsyncOpenAI(api_key=api_key)
        self.seed_individual = None
        
    
    def sanitize_code(self, code: str) -> str:
        replacements = {
            "“": '"', "”": '"',
            "‘": "'", "’": "'",
            "–": "-", "—": "-",
        }
        for old, new in replacements.items():
            sanitized_code = code.replace(old, new)
        matches = re.findall(r'####\s*(.*?)\s*####', sanitized_code, re.DOTALL)
        if not matches:
            # No delimiters, return cleaned response
            return sanitized_code.strip()
        # Take the longest block (works for single or multiple matches)
        longest_match = max(matches, key=lambda x: len(x.strip()))
        return longest_match.strip()

    
    async def LLM_response(self,
                           system_prompt: str, 
                           prompt: str) -> str:
        """
        Native async version using AsyncOpenAI.
        """
        response = await self.openai.chat.completions.create(
            model="gpt-5",  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
        )

        new_code = response.choices[0].message.content
        safe_code = self.sanitize_code(new_code)
        return safe_code


    # Mutation function using LLM
    async def mutation(self,
                 block_code: str,
                 og_code: str = "",
                 improved_code: str = "",
                 eot_prob: float = random.random()):

        # system prompt
        expert = random.choice(self.prompts['expert_roles'])
        expert_role = expert['description']
        premise = self.prompts['Premise']
        system_prompt = premise.format(expert_roles=expert_role)


        if eot_prob < 0.3:
            # plain mutation
            plain_mutation_prompt = self.prompts['Initial mutation']
            constraint = self.prompts['Constraints'][1]["template"]
            prompt = plain_mutation_prompt.format(constraints=constraint, 
                                                  original_class=block_code)
            response_code = await self.LLM_response(system_prompt, prompt)

        else:
            # EoT mutation
            eot_mutation_prompt = self.prompts['EoT_mutation']
            constraint = self.prompts['Constraints'][2]["template"]
            prompt = eot_mutation_prompt.format(initial_code=og_code,
                                                modified_code=improved_code,
                                                code_to_augment=block_code,
                                                constraints=constraint)

            response_code = await self.LLM_response(system_prompt, prompt)
        
        return response_code
    

     # Function to generate the initial population
    def generate_initial_population(self,
                                    initial_seed_code: Individual,
                                    eot_prob: float = 0.2) -> List[Individual]:

        self.population = []
        self.population.append(initial_seed_code)
        self.seed_individual = initial_seed_code

        # Fill population with progress bar
        for _ in tqdm(range(len(self.population), self.population_size), desc="Generating population"):
            new_individual = initial_seed_code.clone()
            block_name, block_code = new_individual.get_random_block()
            mutated_code = self.mutation(block_code=block_code, eot_prob=eot_prob)
            new_individual.gene_blocks[block_name].mutate(new_code=mutated_code)
            self.population.append(new_individual)
        
        for i, ind in enumerate(self.population):
            ind.index = i  # assign a stable index to each individual
    

    # Evaluate the entire population
    def evaluate_population(self, 
                         training_dataloader: DataLoader,
                         validation_dataloader: DataLoader):
        
        generation_logs = {}
        for i, individual in enumerate(self.population):
            desc = f"Evaluating Individual {i+1} | Training epochs"
            individual.fitness_evaluation(training_dataloader, validation_dataloader, desc=desc)
            generation_logs[f'Individual_{i+1}'] = individual.fitness_scores

        for individual in self.population:
            if not individual.fitness_scores:
                raise ValueError("All individuals must have fitness scores before selection.")
        
        return generation_logs
    
    

    # Crossover function using LLM
    async def crossover(self, parent1: Individual, 
                        parent2: Individual) -> Individual:
        
        # system prompt
        expert = random.choice(self.prompts['expert_roles'])
        expert_role = expert['description']
        premise = self.prompts['Premise']
        system_prompt = premise.format(expert_roles=expert_role)
        
        # Single point crossover
        crossover_prompt = self.prompts['Crossover']
        constraint = self.prompts['Constraints'][2]["template"]
        p1_block_name, p1_block_code = parent1.get_random_block()
        p2_block_code = parent2.gene_blocks[p1_block_name].code

        prompt = crossover_prompt.format(constraints=constraint,
                                         p1_block = p1_block_code,
                                         p2_block = p2_block_code)

        new_code = await self.LLM_response(system_prompt,prompt)
        return new_code, p1_block_name
    
    
    
    def clear_population(self):
        self.population = []


    def difference_with_seed(self, individual: Individual) -> int:
        """Calculate the difference between an individual and the seed individual"""
        differences = []
        for block_name in self.seed_individual.gene_blocks:
            if block_name in individual.gene_blocks:
                if self.seed_individual.gene_blocks[block_name].code != individual.gene_blocks[block_name].code:
                    differences.append(block_name)
        return differences