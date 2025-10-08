import pickle
import os
from typing import List
import numpy as np

def population_snapshot(population:List[str],num_generation:int):
    os.makedirs(".snapshots", exist_ok=True)
    with open(f".snapshots/gen{num_generation} .pkl", "wb") as f:
        pickle.dump(population, f)
    try:
        with open(f".snapshots/gen{num_generation}.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return population

def population_snapshot_num(num_generation:int):
    try:
        with open(f".snapshots/gen{num_generation}.pkl", "rb") as f:
            return pickle.load(f)
    except:
        pass

def fitness_snapshot(fitness:List[float],num_generation:int):
    os.makedirs(".snapshots", exist_ok=True)
    with open(f".snapshots/gen_fitness{num_generation} .pkl", "wb") as f:
        pickle.dump(fitness, f)
    try:
        with open(f".snapshots/gen_fitness{num_generation}.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return fitness

def load_snapshot(generation: int, folder: str = "oldData"):
    filename = f"{folder}/gen_{generation}_data.pkl"
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data