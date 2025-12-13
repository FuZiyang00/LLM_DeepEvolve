from GA.individuals import Individual, Genome_block
import ast
from neural_net.dataset import ContrastivePolicyDataset
from torch.utils.data import DataLoader


def validate_code(code_name, code: str) -> bool:
    try:
        ast.parse(code)
        return True   
    except SyntaxError as e:
        print(f"[Validator] Syntax error: {e}, in block {code_name}.")
        print(f"[Validator] Code: {code}")
        return False


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



def data_pipeline(df):
    training_df = df[df["year"] < 2022].reset_index(drop=True)

    validation_df = training_df[training_df["year"] > 2020].reset_index(drop=True)
    training_df = training_df[training_df["year"] <= 2020].reset_index(drop=True)
    
    training_df.drop(columns=['year', 'is_company_italian'], inplace=True) 
    validation_df.drop(columns=['year', 'is_company_italian'], inplace=True)
    dummy_cols = ['Sector', "REGION_GROUP"]
    target = 'target'

    training_dataset = ContrastivePolicyDataset(df=training_df, 
                                                target_column=target,
                                                dummy_cols=dummy_cols)
    
    validation_dataset = ContrastivePolicyDataset(df=validation_df, 
                                                  target_column=target,
                                                  dummy_cols=dummy_cols)
    
    training_dataloader = DataLoader(training_dataset, 
                                     batch_size=16,
                                     shuffle=True, 
                                     drop_last=True,
                                     collate_fn=ContrastivePolicyDataset.collate_fn)
    
    validation_dataloader = DataLoader(validation_dataset, 
                                       batch_size=16,
                                       shuffle=True,
                                       drop_last=True,
                                       collate_fn=ContrastivePolicyDataset.collate_fn)
    
    input_dim = training_dataset.input_dim
    return training_dataloader, validation_dataloader, input_dim
