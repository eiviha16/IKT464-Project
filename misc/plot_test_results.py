from matplotlib import pyplot as plt
import csv
import numpy as np

def plot(data, text, file_path):
    plt.plot(data['timesteps'], data['mean'])
    plt.fill_between(data['timesteps'], np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']), alpha=0.25)
    plt.ylabel(f'Points')
    plt.xlabel(f'Timesteps')
    plt.title(f'{text["title"]}')
    plt.savefig(f'{file_path}/sample_plot.png')
    #plt.show()

def get_csv(file_path):
    data = {'mean': [], 'std': [], 'timesteps': []}
    with open(f'{file_path}/test_results.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] != 'mean':
                data['mean'].append(float(row[0]))
                data['std'].append(float(row[1]))
                data['timesteps'].append(float(row[2]))
    return data

def plot_test_results(file_path, text):
    data = get_csv(file_path)
    plot(data, text, file_path)

if __name__ == "__main__":
    text = {'title': 'DQN'}
    plot_test_results('run_15', text)