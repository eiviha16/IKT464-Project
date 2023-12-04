from matplotlib import pyplot as plt
import csv
import numpy as np


def plot(data, text, file_path):
    plt.show()
    for key in data:
        x = np.arange(1 * 5, 5 * (len(data[key]['timesteps']) + 1), step=5)
        plt.plot(x, data[key]['mean'], label=key)
        # plt.plot(data['timesteps'], data['mean'])
        plt.fill_between(x, np.array(data[key]['mean']) - np.array(data[key]['std']),
                         np.array(data[key]['mean']) + np.array(data[key]['std']), alpha=0.25)
    # plt.fill_between(data['timesteps'], np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']), alpha=0.25)
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.ylabel(f'Rewards')
    plt.xlabel(f'Episodes')
    plt.title(f'{text["title"]}')
    plt.legend()
    # plt.savefig(f'{file_path}/sample_plot.png')
    plt.savefig(f'../results/comparison_plots/reward_plot.png')
    plt.show()


def plot_feedback(data, text, file_path):
    for key in data:
        norm = (data[key]['ratio'] - np.min(data[key]['ratio'])) / (
                    np.max(data[key]['ratio']) - np.min(data[key]['ratio']))
        x = np.arange(0, len(norm))
        x *= 100
        plt.plot(x, norm, label=key)
    # plt.fill_between(data['timesteps'], np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']), alpha=0.25)
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.yticks([0, 0.25, 0.5, 0.75, 1], ["Type I 0", '0.25', '0.5', '0.75', 'Type II 1'])
    plt.ylabel(f'Ratio')
    plt.xlabel(f'Episodes')
    plt.title(f'{text["title"]}')
    plt.legend()
    plt.tight_layout(pad=1.0)
    plt.savefig(f'../results/comparison_plots/feedback_plot.png')
    plt.show()


def plot_q_values(data, text, file_path):
    for key in data:
        norm = (np.array(data[key]['q1']) + np.array(data[key]['q2'])) / 2
        x = np.arange(0, len(norm))
        x *= 100
        plt.plot(x, norm, label=key)
    # plt.fill_between(data['timesteps'], np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']), alpha=0.25)
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.ylabel(f'Q-values')
    plt.xlabel(f'Episodes')
    plt.title(f'{text["title"]}')
    plt.legend()
    plt.tight_layout(pad=1.0)
    plt.savefig(f'../results/comparison_plots/q_values_plot.png')
    plt.show()

    # plt.show()


def plot_q_valuess(data, text, file_path):
    _data = {'TMQN': {'q1': [], 'q2': []}}
    x = []
    for i in range(len(data['TMQN']['q1'])):
        if i == 100:
            break
        _data['TMQN']['q1'].append(data['TMQN']['q1'][i])
        _data['TMQN']['q2'].append(data['TMQN']['q2'][i])
        x.append(i)
    plt.plot(_data['TMQN']['q1'], label='TM 1')
    plt.plot(_data['TMQN']['q2'], label='TM 2')
    # plt.fill_between(data['timesteps'], np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']), alpha=0.25)
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.ylabel(f'Q-values')
    plt.xlabel(f'Step')
    plt.title(f'{text["title"]}')
    plt.legend()
    plt.tight_layout(pad=1.0)
    plt.savefig(f'../results/comparison_plots/q_values_plot_run_218.png')
    plt.show()


def plot_actions(data, text, file_path):
    # for key in data:
    #    if key != 'timesteps':
    # norm = (data[key] - np.min(data[key])) / (np.max(data[key]) - np.min(data[key]))
    norm = (data['ratio'] - np.min(data['ratio'])) / (np.max(data['ratio']) - np.min(data['ratio']))

    x = np.arange(0, len(data['ratio']))
    x = x * 100
    plt.plot(x, norm)
    # plt.fill_between(data['timesteps'], np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']), alpha=0.25)
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.yticks([0, 0.25, 0.5, 0.75, 1], ["TM1 0", '0.25', '0.5', '0.75', 'TM2 1'])
    plt.ylabel(f'Ratio')
    plt.xlabel(f'Episodes')
    plt.title(f'{text["title"]}')
    plt.legend()
    plt.tight_layout(pad=1.0)
    plt.savefig(f'{file_path}/actions_plot.png')
    plt.show()


def get_ratio(data):
    new_data = {}
    for _key in data:
        # _feedback = {'TM 1': [], 'TM 2': [], 'timesteps': []}
        _feedback = {'ratio': [], 'timesteps': []}
        for i in range(len(data[_key]['1_typeI'])):
            if i % 100 == 0:
                """
                if data['1_typeI'][i] != 0:
                    _feedback['TM 1'].append(data['1_typeII'][i] / data['1_typeI'][i])
                else:
                    _feedback['TM 1'].append(data['1_typeII'][i] / 1)
    
                if data['2_typeI'][i] != 0:
                    _feedback['TM 2'].append(data['2_typeII'][i] / data['2_typeI'][i])
                else:
                    _feedback['TM 2'].append(data['2_typeII'][i] / 1)
                """

                if data[_key]['1_typeI'][i] + data[_key]['2_typeI'][i] != 0:
                    _feedback['ratio'].append((data[_key]['1_typeII'][i] + data[_key]['2_typeII'][i]) / (
                                data[_key]['1_typeI'][i] + data[_key]['2_typeI'][i]))
                else:
                    _feedback['ratio'].append((data[_key]['1_typeII'][i] + data[_key]['1_typeII'][i]) / 1)
                _feedback['timesteps'].append(data[_key]['timesteps'][i])

        for key in _feedback:
            _feedback[key] = np.array(_feedback[key])
        new_data[_key] = _feedback
    return new_data


def get_action_ratio(data):
    new_data = {'ratio': [], 'timesteps': []}
    for i in range(len(data['tm1'])):
        if i % 100 == 0:
            if data['tm1'][i] != 0:
                new_data['ratio'].append(data['tm2'][i] / data['tm1'][i])
            else:
                new_data['ratio'].append(data['tm2'][i] / 1)
            new_data['timesteps'].append(data['timesteps'][i])
    return new_data


def get_csv_performance(file_paths):
    data = {}
    for key in file_paths:
        sample = {'mean': [], 'std': [], 'timesteps': []}
        file_path = file_paths[key]

        with open(f'{file_path}/test_results.csv', 'r') as file:
            csv_reader = csv.reader(file)
            i = 0
            for row in csv_reader:
                if i % 1 == 0:
                    if row[0] != 'mean':
                        sample['mean'].append(float(row[0]))
                        sample['std'].append(float(row[1]))
                        sample['timesteps'].append(float(row[2]))
                i += 1
        data[key] = sample
    return data


def get_csv_feedback(file_paths):
    data = {}
    for key in file_paths:
        sample = {'1_typeI': [], '1_typeII': [], '2_typeI': [], '2_typeII': [], 'timesteps': []}
        with open(f'{file_paths[key]}/feedback.csv', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] != '1_typeI':
                    sample['1_typeI'].append(float(row[0]))
                    sample['1_typeII'].append(float(row[1]))
                    sample['2_typeI'].append(float(row[2]))
                    sample['2_typeII'].append(float(row[3]))
                    sample['timesteps'].append(float(row[4]))
        data[key] = sample
    return data


def get_q_vals(file_paths):
    data = {}
    for key in file_paths:
        sample = {'q1': [], 'q2': [], 'timesteps': []}
        with open(f'{file_paths[key]}/q_values.csv', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] != 'q1':
                    sample['q1'].append(float(row[0]))
                    sample['q2'].append(float(row[1]))
                    sample['timesteps'].append(float(row[2]))
        data[key] = sample
    return data


def get_actions(file_path):
    # 1_typeI, 1_typeII, 2_typeI, 2_typeII, steps

    data = {'tm1': [], 'tm2': [], 'timesteps': []}
    with open(f'{file_path}/actions.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] != 'tm1':
                data['tm1'].append(float(row[0]))
                data['tm2'].append(float(row[1]))
                data['timesteps'].append(float(row[2]))
    return data


def plot_test_results(file_paths, text):
    data = get_csv_performance(file_paths)
    plot(data, text, file_paths)


def feedback(file_path, text):
    data = get_csv_feedback(file_path)
    data = get_ratio(data)
    plot_feedback(data, text, file_path)


def actions(file_path, text):
    data = get_actions(file_path)
    data = get_action_ratio(data)
    plot_actions(data, text, file_path)


def q_vals(file_path, text):
    data = get_q_vals(file_path)
    plot_q_values(data, text, file_path)


def q_valss(file_path, text):
    data = get_q_vals(file_path)
    plot_q_valuess(data, text, file_path)


if __name__ == "__main__":
    text = {'title': 'Feedback'}  # {'DQN multi-step': '../results/DQN-n-step-TD/run_4', 'DQN': '../results/DQN/run_74',
    files = {'TMQN': '../results/TMQN/run_218', 'TMQN multi-step': '../results/TMQN-n-step-TD/run_66',
             'TMQN with balanced feedback': '../results/TMQN_w_feedback_balance/run_18',
             'TMQN with balanced update': '../results/TMQN_w_update_balance/run_9'}#, 'DQN multi-step': '../results/DQN-n-step-TD/run_4', 'DQN': '../results/DQN/run_74'}
    plot_test_results(files, text)
    #feedback(files, text)
    # plot_test_results(files, text)
    #q_vals(files, text)
    # ac
    # text = {'title': 'Q-values'}
    # file = {'TMQN': "../results/TMQN/run_218"}
    # text = {'title': 'Feedback'}
    # files = {'0.25': '../results/TMQN_w_feedback_balance/run_18', '0.50': '../results/TMQN_w_feedback_balance/run_17', '0.75': '../results/TMQN_w_feedback_balance/run_19', 'dynamic': '../results/TMQN_w_feedback_balance/run_5'}
    # feedback(files, text)
    # q_valss(file, text)
