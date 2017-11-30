import os

def reader(path):
    results = []
    with open(path,'rt') as file:
        for line in file:
            results.append(line)
        return results

path = "/Users/apple/Documents/tu_darmstadt/Masterarbeit/Documents/Fuernkranz Johannes" \
       "/Refinement and selection heuristics in subgroup discovery and classification rule learning.txt"
text = reader(path)
print(text[2])
