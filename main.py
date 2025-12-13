import pandas as pd
from pipelines.pipelines import GA_pipeline
import json
from GA.individuals import Individual 
import os

if __name__ == "__main__":

    with open("logs/openai_init_pop.json", "r") as f:
        data = json.load(f)
        restored_population = [Individual.from_dict(ind) for ind in data]
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    df = pd.read_csv('data/final_data.csv', sep=';', low_memory=False)
    logs, final_population = GA_pipeline(df, 
                       population_size=10,
                       generations=10,
                       yaml_file='prompts.yaml',
                       elite_size=5, 
                       population=restored_population
                    )
    
    with open('logs/run2_logs.json', 'w') as f:
        json.dump(logs, f, indent=4)

    with open("logs/run_2_pop.json", "w") as f:
        json.dump([ind.to_dict() for ind in final_population], f, indent=4)






    