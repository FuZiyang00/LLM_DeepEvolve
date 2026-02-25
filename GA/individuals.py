from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from neural_net.network import Network_Trainer
import math
import copy
from pathlib import Path
import json
import traceback

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
        self.input_dim = input_dim  # To be set based on dataset
        self.generation = None
        self.index = None
        self.obj_vector = []
        self.fitness_scores = {}
        self.is_valid = True
        self.modified_gene = None

        # NSGA 
        self.NSGA_rank = 0
        self.NSGA_crowding_distance = 0

        # SPEA
        self.SPEA_strength = 0
        self.SPEA_raw_fitness = 0
        self.SPEA_density = 0
        self.SPEA_fitness = 0


    def get_network(self):
        """Compile the individual's genome into a neural network"""

        namespace = {'torch': torch, 'nn': nn, 'F': torch.nn.functional, 
                     'List': List, 'Optional': Optional, 'Dict': Dict, 'math': math}
        
        error_msg = None
        
        # Execute each code block to make classes available
        execution_order = ['Encoder', 'Normalizer', 'ContrastivePolicyNetwork']
        for block_name in execution_order:
            if block_name in self.gene_blocks:
                try:
                    exec(self.gene_blocks[block_name].code, namespace)
                except Exception as e:
                    print(f"[Warning] Failed to execute {block_name}: {e}")
                    self.is_valid = False
                    tb_str = traceback.format_exc()
                    error_msg = tb_str
                    return error_msg
        
        main_class = namespace.get('ContrastivePolicyNetwork')
        if main_class is None:
            print("[Warning] ContrastivePolicyNetwork class not defined.")
            self.is_valid = False
            error_msg = "ContrastivePolicyNetwork class not defined."
            return error_msg

        try:
            return main_class(self.input_dim)
        
        except Exception as e:
            print(f"[Warning] Failed to build network: {e}")
            self.is_valid = False
            tb_str = traceback.format_exc()
            error_msg = tb_str
            return error_msg
        

    def fitness_evaluation(self, 
                           training_data: DataLoader,
                           validation_data: DataLoader, 
                           desc: str): 
        
        error_ms = None

        network = self.get_network()
        if isinstance(network, str):
            error_ms = network
            self.fitness_scores['validation_accuracy'] = 0.0
            self.fitness_scores["parameters_count"] = 0.0

            self.set_obj_vector_worst()
            return error_ms
        
        trainer = Network_Trainer(model=network)
        try: 
            validation_acc, parameters_count = trainer.training(
                train_loader=training_data,
                val_loader=validation_data, 
                desc=desc)
            
        except Exception as e:
            print(f"[Warning] Training failed for individual")
            tb_str = traceback.format_exc()
            error_msg = tb_str
            self.fitness_scores['validation_accuracy'] = 0.0
            self.fitness_scores["parameters_count"] = 0.0
            self.is_valid = False
            self.set_obj_vector_worst()

            return error_msg

        self.obj_vector.append(validation_acc)
        self.obj_vector.append(-parameters_count)

        self.fitness_scores['validation_accuracy'] = validation_acc
        self.fitness_scores["parameters_count"] = parameters_count

        return error_ms

    
    def get_random_block(self) -> str:
        """Select a random block name from the genome"""
        block_name = random.choice(list(self.gene_blocks.keys()))
        block_code = self.gene_blocks[block_name].code
        return block_name, block_code
    

    def clone(self) -> 'Individual':
        """Create a deep copy of the individual (evaluation state reset so the copy can be evaluated)."""
        other = copy.deepcopy(self)
        other.obj_vector = []
        other.fitness_scores = {}
        # Optional: reset NSGA/SPEA so they are recomputed in the next generation
        other.NSGA_rank = 0
        other.NSGA_crowding_distance = 0
        other.SPEA_strength = 0
        other.SPEA_raw_fitness = 0
        other.SPEA_density = 0
        other.SPEA_fitness = 0
        return other

    
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
    

    
    def set_obj_vector_worst(self, n_objectives: int = 2) -> None:
        """Set this individual's objective vector to worst (finite) values.
        For maximization objectives (e.g. accuracy), worst is a very large negative number.
        Uses finite values to avoid NaN in SPEA/NSGA distance calculations.
        """
        worst = -1.0e15  # finite
        self.obj_vector = [worst] * n_objectives


    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Individual":
        """Load a saved individual from a JSON file.

        Supports two formats:
        - Raw format: JSON with top-level "input_dim" and "gene_blocks" (from to_dict()).
        - Wrapped format: JSON with an "individual" key containing the genome dict,
          and optional "fitness_scores" and "obj_vector" to restore on the instance.

        Args:
            path: Path to the JSON file (str or pathlib.Path).

        Returns:
            Individual instance. If the file contained fitness_scores/obj_vector, they are restored.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "individual" in data:
            ind = cls.from_dict(data["individual"])
            if "fitness_scores" in data:
                ind.fitness_scores = data["fitness_scores"]
            if "obj_vector" in data:
                ind.obj_vector = list(data["obj_vector"])
            return ind
        return cls.from_dict(data)