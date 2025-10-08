from typing import List

Individual = str


def split_genes(individual:Individual) -> List[str]:
    genes = []
    current = 0

    for _ in range(2):
        genes.append(individual[current:current+4])
        current += 4

    for _ in range(3):
        genes.append(individual[current:current+5])
        current += 5
    return genes


def join_genes(genes: List[str]) -> Individual:

    return ''.join(genes)