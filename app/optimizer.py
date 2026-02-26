import numpy as np
import joblib
import os

MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise Exception("Model not found. Please run train.py first.")

model = joblib.load(MODEL_PATH)


def genetic_optimize(bounds, population_size=50, generations=30, mutation_rate=0.1):

    dim = len(bounds)

    # 初始化种群
    population = np.zeros((population_size, dim))

    for i in range(dim):
        low, high = bounds[i]
        population[:, i] = np.random.uniform(low, high, population_size)

    for generation in range(generations):

        fitness = model.predict(population)

        # 选择前50%最优个体
        sorted_indices = np.argsort(fitness)[::-1]
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]

        survivors = population[: population_size // 2]

        # 交叉
        children = []
        while len(children) < population_size // 2:

            parents = survivors[np.random.choice(
                len(survivors), 2, replace=False)]
            crossover_point = np.random.randint(1, dim)

            child = np.concatenate(
                [parents[0][:crossover_point], parents[1][crossover_point:]]
            )

            # 变异
            if np.random.rand() < mutation_rate:
                mutate_dim = np.random.randint(0, dim)
                low, high = bounds[mutate_dim]
                child[mutate_dim] = np.random.uniform(low, high)

            children.append(child)

        population = np.vstack((survivors, np.array(children)))

    # 最终最优
    final_fitness = model.predict(population)
    best_index = np.argmax(final_fitness)

    return population[best_index], final_fitness[best_index]
