import random

CROSSOVER_PROBABILITY = 0.8


class Chromosome:

    '''represents a candid solution'''
    def __init__(self, string_length):
        self._genes = []
        self._fitness = 0
        i = 0
        while i < string_length:
            if random.random() >= 0.5:
                self._genes.append(1)
            else:
                self._genes.append(0)
            i = i + 1

    def get_genes(self):
        return self._genes

    def set_genes(self, chromosome):
        self._genes = chromosome

    def get_fitness(self):
        summation = 0
        i = 0
        for i in range(self._genes.__len__()):
            #summation = summation + self._genes[i]
            summation = summation + (2**i)*self._genes[i]
        return summation

    def __str__(self):
        return self._genes.__str__()


class Population:
    '''respresents a population of candidate solutions'''
    def __init__(self, size, string_size):
        self._chromosomes = []
        i = 0
        while i < size:
            self._chromosomes.append(Chromosome(string_size))
            i += 1

    def get_chromosomes(self):
        return self._chromosomes

    def set_chromosome(self, chromosome):
        self._chromosomes.append(chromosome)


class GeneticAlgorithm:
    # where mutation and crossover happens

    def __init__(self, population_size, string_length, mutation_prob):
        self._pop_size = population_size
        self._string_length = string_length
        self._mutation_prob = mutation_prob

    def evolve(self, pop):
        children = self._mutate_population(self._crossover_population(pop))
        # print_population(children, 200)
        evolved_population = self.combine_children_and_parents(pop, children)
        # print_population(evolved_population, 300)
        return evolved_population

    def combine_children_and_parents(self, population1, population2):
        children_parents = Population(0, 0)
        for k in range(self._pop_size):
            children_parents.get_chromosomes().append(population1.get_chromosomes()[k])
        for j in range(self._pop_size/2):
            children_parents.get_chromosomes().append(population2.get_chromosomes()[j])
            # print_population(children_parents, 90)
        return children_parents

    def _crossover_chromosomes(self, chromosome1, chromosome2):
        child1 = Chromosome(0)
        child2 = Chromosome(0)
        father = chromosome1
        mother = chromosome2
        index = random.randint(1, self._string_length - 2)
        child1.set_genes(father.get_genes()[:index] + mother.get_genes()[index:])
        child2.set_genes(mother.get_genes()[:index] + father.get_genes()[index:])
        return child1, child2

    def _mutate_chromosome(self, chromosome):
        for i in range(self._string_length):
            if random.random() < self._mutation_prob:
                if random.random() < 0.5:
                    chromosome.get_genes()[i] = 1
                else:
                    chromosome.get_genes()[i] = 0

    def _crossover_population(self, pop):
        crossover_pop = Population(0, 0)
        mating_pool = Population(0, 0)
        chooseparents = pop.get_chromosomes()
        parents_size = self._pop_size / 2
        max_fitness = sum(chromosome.get_fitness() for chromosome in chooseparents)
        for j in range(parents_size):
            pick = random.uniform(0, max_fitness)       # using a roulett wheel
            current = 0
            for chromosome in chooseparents:
                current += chromosome.get_fitness()
                if current > pick:
                    mating_pool.get_chromosomes().append(chromosome)
        j = 0
        while j < parents_size:
            parent1 = mating_pool.get_chromosomes()[j]
            j += 1
            parent2 = mating_pool.get_chromosomes()[j]
            j += 1
            child1, child2 = self._crossover_chromosomes(parent1, parent2)
            crossover_pop.get_chromosomes().append(child1)
            crossover_pop.get_chromosomes().append(child2)
        # print_population(crossover_pop, 100)
        return crossover_pop

    def _mutate_population(self, pop):
        for i in range(self._pop_size/2):
            self._mutate_chromosome(pop.get_chromosomes()[i])
        return pop


def print_population(pop, gen_number):
    print"\n------------------------------------"
    print "generation #", gen_number, "|fittest chromosome fitness:", pop.get_chromosomes()[0].get_fitness()
    print "------------------------------------"
    i = 0
    for x in pop.get_chromosomes():
        print "chromosome #", i, " :", x, "fitness", x.get_fitness()
        i += 1


def GA(population_size, strings_length, target, number_of_iterations, mutation_probability):
    genetic = GeneticAlgorithm(population_size, strings_length, mutation_probability)
    population = Population(population_size, strings_length)    # the random first generation
    population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)  # sorting the generation by fitness
    # print_population(population, 0)
    iter_counter = 0
    while (iter_counter < number_of_iterations) and (population.get_chromosomes()[0].get_fitness() != target):
        new_generation = genetic.evolve(population)
        new_generation.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
        population = Population(0, 0)
        for i in range(population_size):
            population.get_chromosomes().append(new_generation.get_chromosomes()[i])
        iter_counter += 1
        # print_population(population, iter_counter)
    return population.get_chromosomes()[0], iter_counter,


def scenario():
    print "Fitness : Summation of (2**i)xi"
    print "*** scenario #1: Population size->10, string size->10, iteration number->50"
    population_size = 10
    strings_length = 10
    target = 1023
    number_of_iterations = 50
    mutation_probability = 1/population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations, mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest/5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation/5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #2: Population size->20, string size->10, iteration number->50"
    population_size = 20
    strings_length = 10
    target = 1023
    number_of_iterations = 50
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #3: Population size->50, string size->10, iteration number->50"
    population_size = 50
    strings_length = 10
    target = 1023
    number_of_iterations = 50
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #4: Population size->10, string size->50, iteration number->50"
    population_size = 10
    strings_length = 50
    target = 1125899906842623
    number_of_iterations = 50
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #5: Population size->20, string size->50, iteration number->50"
    population_size = 20
    strings_length = 50
    target = 1125899906842623
    number_of_iterations = 50
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #6: Population size->50, string size->50, iteration number->50"
    population_size = 50
    strings_length = 50
    target = 1125899906842623
    number_of_iterations = 50
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #7: Population size->10, string size->100, iteration number->50"
    population_size = 10
    strings_length = 100
    target = 1267650600228229401496703205375
    number_of_iterations = 50
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #8: Population size->20, string size->100, iteration number->50"
    population_size = 20
    strings_length = 100
    target = 1267650600228229401496703205375
    number_of_iterations = 50
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #9: Population size->50, string size->100, iteration number->50"
    population_size = 50
    strings_length = 100
    target = 1267650600228229401496703205375
    number_of_iterations = 50
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "Fitness : Summation of xi"
    print "*** scenario #10: Population size->10, string size->10, iteration number->100"
    population_size = 10
    strings_length = 10
    target = 1023
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #11: Population size->20, string size->10, iteration number->100"
    population_size = 20
    strings_length = 10
    target = 1023
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #12: Population size->50, string size->10, iteration number->100"
    population_size = 50
    strings_length = 10
    target = 1023
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #13: Population size->10, string size->50, iteration number->100"
    population_size = 10
    strings_length = 50
    target = 1125899906842623
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #14: Population size->20, string size->50, iteration number->100"
    population_size = 20
    strings_length = 50
    target = 1125899906842623
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #15: Population size->50, string size->50, iteration number->100"
    population_size = 50
    strings_length = 50
    target = 1125899906842623
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #16: Population size->10, string size->100, iteration number->100"
    population_size = 10
    strings_length = 100
    target = 1267650600228229401496703205375
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #17: Population size->20, string size->100, iteration number->100"
    population_size = 20
    strings_length = 100
    target = 1267650600228229401496703205375
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #18: Population size->50, string size->100, iteration number->100"
    population_size = 50
    strings_length = 100
    target = 1267650600228229401496703205375
    number_of_iterations = 100
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "Fitness : Summation of xi"
    print "*** scenario #19: Population size->10, string size->10, iteration number->200"
    population_size = 10
    strings_length = 10
    target = 1023
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #20: Population size->20, string size->10, iteration number->200"
    population_size = 20
    strings_length = 10
    target = 1023
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #21: Population size->50, string size->10, iteration number->200"
    population_size = 50
    strings_length = 10
    target = 1023
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #22: Population size->10, string size->50, iteration number->200"
    population_size = 10
    strings_length = 50
    target = 1125899906842623
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #23: Population size->20, string size->50, iteration number->200"
    population_size = 20
    strings_length = 50
    target = 1125899906842623
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #24: Population size->50, string size->50, iteration number->200"
    population_size = 50
    strings_length = 50
    target = 1125899906842623
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #25: Population size->10, string size->100, iteration number->200"
    population_size = 10
    strings_length = 100
    target = 1267650600228229401496703205375
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #26: Population size->20, string size->100, iteration number->200"
    population_size = 20
    strings_length = 100
    target = 1267650600228229401496703205375
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest

    print "*** scenario #27: Population size->50, string size->100, iteration number->200"
    population_size = 50
    strings_length = 100
    target = 1267650600228229401496703205375
    number_of_iterations = 200
    mutation_probability = 1 / population_size
    avg_fittest = 0
    avg_number_of_generation = 0
    frequency_of_ones_in_fittest = []
    for k in range(strings_length):
        frequency_of_ones_in_fittest.append(0)
    for i in range(5):
        best, generation_number = GA(population_size, strings_length, target, number_of_iterations,
                                     mutation_probability)
        avg_fittest = best.get_fitness() + avg_fittest
        avg_number_of_generation = generation_number + avg_number_of_generation
        for j in range(best.get_genes().__len__()):
            frequency_of_ones_in_fittest[j] = frequency_of_ones_in_fittest[j] + best.get_genes()[j]
    print "avg of best fitness in 5 runs: ", avg_fittest / 5
    print "avg of number of generations in 5 runs: ", avg_number_of_generation / 5
    print "frequency of ones in fittest in 5 runs: ", frequency_of_ones_in_fittest




scenario()






