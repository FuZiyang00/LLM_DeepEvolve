import pandas as pd
from pipelines.pipelines import run_evolution
import json
from GA.individuals import Individual 
from config import settings
from pathlib import Path
import asyncio

def load_population_from_generation_log(log_path: str | Path) -> list[Individual]:
    """Load a list of Individual from a generation_*.json log file."""
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    population = []
    for key in sorted(data.keys(), key=lambda x: int(x.split("_")[1])):
        entry = data[key]
        ind = Individual.from_dict(entry["individual"])
        if "obj_vector" in entry and entry["obj_vector"]:
            ind.obj_vector = list(entry["obj_vector"])
        if "fitness_scores" in entry:
            ind.fitness_scores = dict(entry["fitness_scores"])
        population.append(ind)
    return population

if __name__ == "__main__":

    current_gen = 11
    file_path = settings.logs_path / f"generation_{current_gen}.json"
    initial_population = load_population_from_generation_log(file_path)

    df = pd.read_csv('final_data.csv', sep=';', low_memory=False)

    asyncio.run(run_evolution(
        data=df, 
        population_size=10, 
        generations=20,
        elite_archive_size=4, 
        initial_population= initial_population
        ))
    

    
    # with open('logs/run2_logs.json', 'w') as f:
    #     json.dump(logs, f, indent=4)

    # with open("logs/run_2_pop.json", "w") as f:
    #     json.dump([ind.to_dict() for ind in final_population], f, indent=4)






    