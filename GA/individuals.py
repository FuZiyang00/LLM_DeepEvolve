from typing import Dict, List, Optional
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from neural_net.network import Network_Trainer
import math

# ============================================================================
# 1. INDIVIDUAL AND GENOME REPRESENTATION
# ============================================================================

class Genome_block:
    """Represents a single code block (gene) in the individual's genome"""
    def __init__(self, 
                 block_name: str, 
                 code: str):
        self.block_name = block_name  # Name of the code block
        self.code = code              # Actual code of the block

    def mutate(self, new_code: str):
        """Mutate the code block with new code"""
        self.code = new_code

    def to_dict(self):
        """Convert genome block to dictionary (for saving)"""
        return {"block_name": self.block_name,
                "code": self.code}

    @classmethod
    def from_dict(cls, data):
        """Rebuild genome block from dictionary"""
        return cls(block_name=data["block_name"], code=data["code"])
    

class Individual:
    """Represents a complete neural network individual in the evolution"""
    def __init__(self, 
                 input_dim: int,
                 gene_blocks:Dict[str, Genome_block]):
        self.gene_blocks = gene_blocks  # Dictionary of genome blocks
        self.fitness_scores = None
        self.input_dim = input_dim  # To be set based on dataset
        self.index = None
        self.obj_vector = []
        self.fitness_scores = {}
        self.is_valid = True

        # for SPEA2 and NSGA-II
        self.SPEA_strength = 0
        self.SPEA_raw_fitness = 0
        self.SPEA_density = 0.0
        self.SPEA_final_fitness = 0.0
        self.NSGA_rank = 0
        self.NSGA_crowding_distance = 0.0


    def get_network(self):
        """Compile the individual's genome into a neural network"""
        if not self.is_valid:
            return None
            
        namespace = {'torch': torch, 'nn': nn, 'F': torch.nn.functional, 
                     'List': List, 'Optional': Optional, 'Dict': Dict, 'math': math}
        
        # Execute each code block to make classes available
        execution_order = ['Encoder', 'Normalizer', 'ContrastivePolicyNetwork']
        for block_name in execution_order:
            if block_name in self.gene_blocks:
                try:
                    exec(self.gene_blocks[block_name].code, namespace)
                except Exception as e:
                    print(f"[Warning] Failed to execute {block_name}: {e}")
                    self.is_valid = False
                    return None
        
        main_class = namespace.get('ContrastivePolicyNetwork')
        if main_class is None:
            print("[Warning] ContrastivePolicyNetwork class not defined in genome.")
            self.is_valid = False
            return None
        
        return main_class(self.input_dim)
    
    def get_random_block(self) -> str:
        """Select a random block name from the genome"""
        block_name = random.choice(list(self.gene_blocks.keys()))
        block_code = self.gene_blocks[block_name].code
        return block_name, block_code

    def fitness_evaluation(self, 
                           training_data: DataLoader,
                           validation_data: DataLoader, 
                           desc: str): 
        
        if len(self.obj_vector) > 0:
            return # Already evaluated 
        
        network = self.get_network()
        
        if network is None:
            self.fitness_scores['training_accuracy'] = 0.0
            self.fitness_scores['validation_accuracy'] = 0.0
            return
        
        trainer = Network_Trainer(model=network)
        try: 
            training_acc, validation_acc = trainer.training(
                train_loader=training_data,
                val_loader=validation_data, 
                desc=desc)
        except Exception as e:
            print(f"[Warning] Training failed for individual {self.index + 1}: {e}")
            for block in self.gene_blocks.values():
                print(f"--- {block.block_name} ---")
                print(block.code)
            self.fitness_scores['training_accuracy'] = 0.0
            self.fitness_scores['validation_accuracy'] = 0.0
            self.is_valid = False
            return

        self.obj_vector.append(training_acc)
        self.obj_vector.append(validation_acc)
        self.fitness_scores['training_accuracy'] = training_acc
        self.fitness_scores['validation_accuracy'] = validation_acc
        print(f"Training Acc: {training_acc:.4f} | Validation Acc: {validation_acc:.4f}")


    def clone(self) -> 'Individual':
        """Create a deep copy of the individual"""
        new_blocks = {name: Genome_block(block.block_name, block.code) 
                      for name, block in self.gene_blocks.items()}
        new_individual = Individual(self.input_dim, new_blocks)
        new_individual.fitness_scores = self.fitness_scores
        new_individual.SPEA_strength = self.SPEA_strength
        new_individual.SPEA_density = self.SPEA_density
        new_individual.NSGA_rank = self.NSGA_rank
        new_individual.NSGA_crowding_distance = self.NSGA_crowding_distance
        return new_individual


    def to_dict(self):
        """Convert individual to dictionary (for saving)"""
        return {
            "input_dim": self.input_dim,
            "gene_blocks": {k: v.to_dict() for k, v in self.gene_blocks.items()}
        }


    @classmethod
    def from_dict(cls, data):
        """Rebuild individual from dictionary"""
        gene_blocks = {k: Genome_block.from_dict(v) for k, v in data["gene_blocks"].items()}
        return cls(input_dim=data["input_dim"], gene_blocks=gene_blocks)