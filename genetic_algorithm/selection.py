import random

def russian_roulette_selection(num_to_select, model_list):
    # Find the minimum fitness score
    minimum_fitness = model_list[0].fitness_score
    for model in model_list:
        minimum_fitness = min(minimum_fitness, model.fitness_score)

    # Shift fitness scores if the minimum is negative
    if minimum_fitness < 0:
        for model in model_list:
            model.fitness_score += (-minimum_fitness)

    # Calculate the total fitness of the population
    total_fitness = sum(model.fitness_score for model in model_list)

    # Check if total fitness is zero
    if total_fitness == 0:
        raise ValueError("Total fitness is zero; cannot perform roulette selection.")

    # Normalize fitness to get selection probabilities
    selection_probabilities = [model.fitness_score / total_fitness for model in model_list]

    # Ensure num_to_select does not exceed the size of the population
    if num_to_select > len(model_list):
        print(f"Warning: num_to_select ({num_to_select}) exceeds the population size ({len(model_list)}). Adjusting to {len(model_list)}.")
        num_to_select = len(model_list)

    # Select models based on their fitness scores using roulette selection
    selected_models = []
    while num_to_select > 0:
        rand_value = random.random()
        cumulative_probability = 0.0
        for i, prob in enumerate(selection_probabilities):
            cumulative_probability += prob
            if rand_value <= cumulative_probability:
                selected_models.append(model_list[i])
                break
        num_to_select -= 1

    return selected_models
