def geneticAlgorithm(generations=3):
  initialPopulation=assignFitness(random_models(100))

  for gen in range(generations):
    print(f"Generation {gen+1}")

    selec=russian_roulette_selection(50,initialPopulation)

    crossovered_models=perform_crossover(selec)

    mutated_models=mutate_all_chromosomes(crossovered_models)

    mutated_assigned=assignFitness(mutated_models)

    initialPopulation.extend(mutated_assigned)

    print()
    for model in initialPopulation:
        print(model.encoding, " " , model.fitness_score)

  russian_roulette_selection(50,initialPopulation)

  return initialPopulation

result=geneticAlgorithm(2)
print("After all Generations")
for model in result:
    print(model.encoding, " ", model.fitness_score)