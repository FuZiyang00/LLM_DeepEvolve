from GA.individuals import Individual, Genome_block
import ast
from torch.utils.data import DataLoader
import logging
import re
import random
import sys
from typing import List, Optional
from GA.SPEA import SPEA2
import traceback
import re
from torch.utils.data import DataLoader
from torch import Generator
from neural_net.dataset import ContrastivePolicyDataset, FixedSamplesDataset, precompute_contrastive_samples
from config import settings

# ============================================================================
# 1. Function to generate a seed Individual
# ============================================================================

def create_contrastive_policy_seed(input_dim: int) -> Individual:
    gene_blocks = {
        'Encoder': Genome_block('Encoder', 
'''
class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128], 
                 embedding_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)
'''),
        
        'Normalizer': Genome_block('Normalizer', 
'''
class Normalizer(nn.Module):
    def __init__(self, norm_type='l2', dim=1):
        super().__init__()
        self.norm_type = norm_type
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
'''),
        
        'ContrastivePolicyNetwork': Genome_block('ContrastivePolicyNetwork', 
'''
class ContrastivePolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128],
                 embedding_dim: int = 64, dropout: float = 0.1, 
                 temperature: float = 0.07):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, embedding_dim, dropout)
        self.normalizer = Normalizer()
        self.temperature = temperature
    
    def forward(self, x):
        embeddings = self.encoder(x)
        normalized_embeddings = self.normalizer(embeddings)
        return normalized_embeddings
''')
    }
    return Individual(input_dim, gene_blocks)


# =============================================================================================
# 2.set of functions to sanitize the llm generated string code and to validate its functionning
# =============================================================================================

def sanitize_code(code: str) -> str:
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


def validate_code(code_name, code: str) -> tuple[bool, str | None]:
    try:
        ast.parse(code)
        return True, None
    except SyntaxError:
        full_traceback = traceback.format_exc()
        return False, full_traceback
    

def validate_and_add(child: Individual, 
                     parent_1: Individual, 
                     offsprings: List):

    if child: 
        child.is_valid = True
        offsprings.append(child)
    else: 
        invalid_child: Individual = parent_1.clone()
        invalid_child.set_obj_vector_worst()
        invalid_child.is_valid = False
        offsprings.append(invalid_child)
    
# def validate_and_run():
#     validate_code(dom_genome_name, mutated_genome_code)
#     mutated_offspring: Individual = child.clone()
#     mutated_offspring.gene_blocks[dom_genome_name].mutate(new_code=mutated_genome_code)

#     network = mutated_offspring.get_network()  # Validate the child's network

#     if network is None:
#         print("Child network creation after mutation failed.")
#         mutated_offspring.is_valid = False
#     else:
#         print(f"New child created from parents {parent_1.index} and {parent_2.index}.")
#         mutated_offspring.is_valid = True
#         mutated_offsprings.append(mutated_offspring)
#         counter += 1
#         del network


# =============================================================================================
# 3. Function for generating training set and validation set 
# =============================================================================================


def data_pipeline(df, seed: Optional[int] = None):
    if seed is None:
        seed = getattr(settings, "data_seed", 42)
    training_df = df[df["year"] < 2022].reset_index(drop=True)

    validation_df = training_df[training_df["year"] > 2020].reset_index(drop=True)
    training_df = training_df[training_df["year"] <= 2020].reset_index(drop=True)

    training_df.drop(columns=['year', 'is_company_italian'], inplace=True)
    validation_df.drop(columns=['year', 'is_company_italian'], inplace=True)
    dummy_cols = ['Sector', "REGION_GROUP"]
    target = 'target'

    training_dataset_raw = ContrastivePolicyDataset(df=training_df,
                                                    target_column=target,
                                                    dummy_cols=dummy_cols)

    validation_dataset_raw = ContrastivePolicyDataset(df=validation_df,
                                                      target_column=target,
                                                      dummy_cols=dummy_cols)

    # Precompute fixed samples so every individual and generation sees the same data
    training_samples = precompute_contrastive_samples(training_dataset_raw, seed)
    validation_samples = precompute_contrastive_samples(validation_dataset_raw, seed)

    training_dataset = FixedSamplesDataset(training_samples, training_dataset_raw.input_dim)
    validation_dataset = FixedSamplesDataset(validation_samples, validation_dataset_raw.input_dim)

    g = Generator().manual_seed(seed)
    training_dataloader = DataLoader(training_dataset,
                                     batch_size=16,
                                     shuffle=True,
                                     drop_last=True,
                                     generator=g,
                                     collate_fn=ContrastivePolicyDataset.collate_fn)

    g_val = Generator().manual_seed(seed + 1)  # different seed so val order is fixed but not same as train
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=16,
                                       shuffle=True,
                                       drop_last=True,
                                       generator=g_val,
                                       collate_fn=ContrastivePolicyDataset.collate_fn)

    input_dim = training_dataset.input_dim
    return training_dataloader, validation_dataloader, input_dim


# =============================================================================================
# 3. Logger function 
# =============================================================================================


# Constants for the format you requested
LOG_FORMAT = "[%(asctime)s] [%(filename)s] : %(message)s"
DATE_FORMAT = "%d/%m/%Y"  # DD/MM/YYYY

def setup_logger(name: str, 
                 level=logging.INFO):
    """
    Creates a configured logger instance.
    
    Args:
        name: The name of the logger (usually __name__).
        log_file: Optional Path object. If provided, logs will also be written to this file.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    # 1. Initialize Logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 2. Idempotency Check (Prevent duplicate logs if function is called twice)
    if logger.hasHandlers():
        return logger

    # 3. Create Formatter
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # 4. Console Handler (Standard Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger




# ============================================================================
# 4. DOMINANCE AND SELECTION HELPERS
# ============================================================================


def select_random_dominator(seed: Individual, population: List[Individual]) -> Optional[Individual]:
    """
    From a population (seed excluded), randomly select one individual that dominates the seed.
    Returns None if population is empty or no individual dominates the seed.
    """
    if not population:
        return None
    if not getattr(seed, "obj_vector", None) or len(seed.obj_vector) == 0:
        return None
    dominators = [
        ind for ind in population
        if getattr(ind, "obj_vector", None)
        and len(ind.obj_vector) == len(seed.obj_vector)
        and SPEA2.dominates(ind, seed)
    ]
    if not dominators:
        return None
    return random.choice(dominators)




BLOCK_NAMES = ["Encoder", "Normalizer", "ContrastivePolicyNetwork"]

def parse_full_genome_response(response: str, input_dim: int):
    """
    Parse LLM full-genome repair response into an Individual ready for retraining.
    Returns None if the response does not contain exactly the three blocks.
    """
    text = response.strip()
    if not text:
        return None

    pattern = (
        r'####\s*('
        + '|'.join(re.escape(b) for b in BLOCK_NAMES)
        + r')\s*####\s*\n?(.*?)(?=\s*####\s*(?:'
        + '|'.join(re.escape(b) for b in BLOCK_NAMES)
        + r')\s*####|\Z)'
    )
    matches = re.findall(pattern, text, re.DOTALL)

    try: 
        parsed = {}
        for block_name, code in matches:
            code = code.strip()
            if block_name in parsed:
                return None
            parsed[block_name] = code

    except Exception:
        return None

    if set(parsed.keys()) != set(BLOCK_NAMES):
        return None

    
    gene_blocks = {
        name: Genome_block(block_name=name, code=parsed[name])
        for name in BLOCK_NAMES
    }
    return Individual(input_dim=input_dim, gene_blocks=gene_blocks)