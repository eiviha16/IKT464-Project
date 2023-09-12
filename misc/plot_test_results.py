from matplotlib import pyplot as plt
import csv
import numpy as np

def plot(data, text):

    plt.plot(data['timesteps'], data['mean'])

    plt.fill_between(data['timesteps'], np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']), alpha=0.25)
    plt.ylabel(f'Points')
    plt.xlabel(f'Timesteps')
    plt.legend()
    plt.title(f'{text["title"]}')
    plt.show()

def get_csv(run_id):
    data = {'mean': [], 'std': [], 'timesteps': []}
    with open(f'../results/{run_id}/test_results.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] != 'mean':
                data['mean'].append(float(row[0]))
                data['std'].append(float(row[1]))
                data['timesteps'].append(float(row[2]))
    return data

def plot_test_results(run_id, text):
    data = get_csv(run_id)
    plot(data, text)

if __name__ == "__main__":
    text = {'title': 'DQN'}
    plot_test_results('run_10', text)