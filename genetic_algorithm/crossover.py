# Uniform Crossover
def crossover(parent1, parent2):
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length.")

    offspring1 = []
    offspring2 = []

    for i in range(len(parent1)):
        if random.random() < 0.5:
            offspring1.append(parent1[i])  # Take from parent1
            offspring2.append(parent2[i])  # Take from parent2
        else:
            offspring1.append(parent2[i])  # Take from parent2
            offspring2.append(parent1[i])  # Take from parent1

    return offspring1, offspring2


import random

def multi_point_crossover(parent1, parent2, group_size=5):
    # Ensure the parents have the same length and divisible by group_size
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length.")
    if len(parent1) % group_size != 0:
        raise ValueError("Parent length must be divisible by the group size.")

    # Number of groups
    num_groups = len(parent1) // group_size

    # Split parents into groups
    parent1_groups = [parent1[i * group_size: (i + 1) * group_size] for i in range(num_groups)]
    parent2_groups = [parent2[i * group_size: (i + 1) * group_size] for i in range(num_groups)]

    # Initialize offspring
    offspring1 = []
    offspring2 = []

    # Alternate swapping groups between the parents with random decision for each group
    for i in range(num_groups):
        rand_value = random.random()  # Random value between 0 and 1

        if rand_value < 0.5:  # Take from Parent 1 for Offspring 1, Parent 2 for Offspring 2
            offspring1.extend(parent1_groups[i])
            offspring2.extend(parent2_groups[i])
        else:  # Take from Parent 2 for Offspring 1, Parent 1 for Offspring 2
            offspring1.extend(parent2_groups[i])
            offspring2.extend(parent1_groups[i])

    return offspring1, offspring2

def perform_crossover(models):
    random.shuffle(models)  # Shuffle the models to ensure random pairing

    crossovered_models = []
    for i in range(0, len(models) - 1, 2):
        parent1 = models[i].encoding
        parent2 = models[i + 1].encoding

        child1_model = None
        child2_model = None

        while child1_model is None or child2_model is None:
            # Perform uniform crossover on the first 4 digits
            child1, child2 = crossover(parent1[:4], parent2[:4])
            # Perform multi-point crossover on the rest
            param1, param2 = multi_point_crossover(parent1[4:], parent2[4:])

            # Combine the blocks and parameters to form complete children
            child1 = child1 + param1
            child2 = child2 + param2

            # Create new models and validate them
            child1_model = UNet(child1, 3, 3)
            if not child1_model.isOk:
                child1_model = None  # Reset if invalid

            child2_model = UNet(child2, 3, 3)
            if not child2_model.isOk:
                child2_model = None  # Reset if invalid

            # If both children are valid, break the loop
            if child1_model and child2_model:
                crossovered_models.append(child1_model)
                crossovered_models.append(child2_model)

        # Add valid offspring to the list
    # If odd number of models, retain the last model unchanged
    if len(models) % 2 == 1:
        crossovered_models.append(models[-1])

    return crossovered_models
