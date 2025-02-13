import random

def mutate_chromosome(chromosome, num_models=4, param_length=5, small_value_range=(-2, 2)):
    # Randomly mutate one model (block)
    random_model_index = random.randint(0, num_models - 1)
    chromosome[random_model_index] = random.randint(1, 5)  # Assign random block type between 1 and 5

    # Randomly mutate two parameters from the parameter section (after the first `num_models`)
    param_indices = random.sample(range(len(chromosome[num_models:])), 2)
    for idx in param_indices:
        # Get the actual index in the full chromosome
        actual_idx = num_models + idx
        current_value = chromosome[actual_idx]

        # Add or subtract a random small value
        small_value = random.randint(*small_value_range)
        new_value = max(0, min(9, current_value + small_value))  # Ensure valid range for parameters

        chromosome[actual_idx] = new_value

    return chromosome


def mutate_all_chromosomes(chromosomes, num_models=4, param_length=5, small_value_range=(-3, 3)):
    mutated_models = []
    for model in chromosomes:
        # Extract the encoded list from each model (assuming 'encoding' is the correct attribute)
        chromosome = model.encoding

        # Attempt mutation until a valid model is found
        mutated_model = None
        retry_limit = 1000  # Max number of mutation attempts
        retry_count = 0

        while mutated_model is None or not mutated_model.isOk:
            if retry_count >= retry_limit:
                #print("Max retry limit reached for mutation")
                break
            # Mutate the chromosome
            mutated_chromosome = mutate_chromosome(chromosome, num_models, param_length, small_value_range)

            # Create a new model (assuming 'UNet' constructor works as expected)
            mutated_model = UNet(mutated_chromosome, 3, 3)
            retry_count += 1

        if mutated_model and mutated_model.isOk:
            # Add the valid mutated model to the list
            mutated_models.append(mutated_model)

    return mutated_models
