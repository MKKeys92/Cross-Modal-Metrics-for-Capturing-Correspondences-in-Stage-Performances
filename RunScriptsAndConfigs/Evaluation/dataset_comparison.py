import os
import Sources.utils.util as util
import Sources.utils.logger as logger
from Sources.utils.functional import load_data
from Sources.utils.util import dotdict
import argparse
import numpy as np
from Sources.Evaluation.evaluator import LightEvaluator
from Sources.Evaluation.evaluator import set_bp_edge
from Sources.Evaluation.evaluator import set_bp_fill
import matplotlib.pyplot as plt
import Sources.Preprocessing.Light.AbstractionLayers as AL



def load_dataset_and_create_evaluator(p_data, base_args):
    base_args.dataset_path = os.path.join(base_args.dataset_dir,p_data)

    train_data, test_data, val_data, dataset_config = load_data(base_args)

    if not base_args.test_only:
        data = np.concatenate((train_data, test_data, val_data))
    else:
        data = test_data

    util.add_all_audio_features(data, dataset_config)

    d = {}
    config = dotdict(d)
    config.prepro_config = dataset_config

    evaluator = LightEvaluator(None, None, config, base_args, is_dataset_eval=True)
    evaluator.set_test_data(data)

    return data, evaluator, config

def main():
    """ Main function """

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='which GPU to use, CPU has value -1')

    parser.add_argument('--dataset_dir', type=str, default='../../Data/Input/CombinedDataSets',
                        help='directory of datasets')

    parser.add_argument('--test_only', type=bool, default=False,
                        help='load only test data of dataset')

    base_args = parser.parse_args()

    logger.init(None)

    x_tick_distance = 2

    s1 = {"name": "real", "filename": "base_Aug23_11-18-50.pkl", "bp_color": "green",
          "bp_offset": -0.3}
    s2 = {"name": "random", "filename": "base_random_Aug23_11-25-44.pkl", "bp_color": "red", "bp_offset": 0.3}

    sets = [s1,s2]

    results = []
    data = []
    evals = []
    configs = []

    lighting_abstraction_layer = None

    for idx in range(len(sets)):
        s = sets[idx]
        d, e, c = load_dataset_and_create_evaluator(s["filename"],base_args)
        data.append(d)
        evals.append(e)
        configs.append(c)
        r={}
        results.append(r)
        if lighting_abstraction_layer is None:
            lighting_abstraction_layer = c.prepro_config.lighting_abstraction_layer
        else:
            assert(lighting_abstraction_layer == c.prepro_config.lighting_abstraction_layer)

    labels = ['number of songs','durations','bpms', 'colors']

    al = AL.get_abstraction_layer(lighting_abstraction_layer)


    for i in range(len(data)):
        d = data[i]
        r = results[i]

        #number of songs
        n = len(d)
        r[labels[0]]= n
        logger.log('number of recordings ' + sets[i]["name"] + ': ' + str(n))

        #durations
        durations = []
        for da in d:
            durations.append(len(da['lighting_array'])/1800)
        r[labels[1]] = durations
        mean_duration = np.mean(durations)
        std_durations = np.std(durations)
        logger.log('overall duration ' + sets[i]["name"] + ': ' + str(sum(durations)))
        logger.log('mean duration ' + sets[i]["name"] + ': ' + str(mean_duration))
        logger.log('std duration ' + sets[i]["name"] + ': ' + str(std_durations))

        #bpms
        bpms = []
        for da in d:
            tempo = da['all_audio_features']['tempo']
            bpms.append(tempo)
        r[labels[2]] = bpms
        mean_bpm = np.mean(bpms)
        std_bpm = np.std(bpms)
        logger.log('mean bpm ' + sets[i]["name"] + ': ' + str(mean_bpm))
        logger.log('std bpm ' + sets[i]["name"] + ': ' + str(std_bpm))


        # colors
        saturations = []
        for da in d:
            l = da['lighting_array'].copy()
            l,_ = al.get_sat(l)
            size = l.size
            l = l>0
            n_c = np.sum(l)
            saturations.append(n_c/size*100)
        r[labels[3]] = saturations
        mean_saturations = np.mean(saturations)
        logger.log('mean colors ' + sets[i]["name"] + ': ' + str(mean_saturations))


    names = []
    for i in range(len(sets)):
        names.append(sets[i]['name'])


    data_durations = []
    data_bpms = []
    data_colors = []
    for r in results:
        data_durations.append(r[labels[1]])
        data_bpms.append(r[labels[2]])
        data_colors.append(r[labels[3]])

    data_plot = [data_durations,data_bpms,data_colors]
    graph_labels=['duration', 'bpm', 'color']
    y_labels = ['min', 'bpm', 'percent']
    fig, ax = plt.subplots(1, 3)
    for i in range(3):
        bp = ax[i].boxplot(data_plot[i], labels=names)
        ax[i].set_title(graph_labels[i])
        ax[i].set_ylabel(y_labels[i])
    plt.show()

    #eval for every dataset

    var_names = ['$\Gamma_\mathrm{beat \leftrightarrow peak}$', '$\Gamma_\mathrm{beat \leftrightarrow valley}$',
                 'N_BA_Max', 'N_BA_Min', '$\Gamma_\mathrm{loud \leftrightarrow bright}$',
                 '$\Gamma_\mathrm{change}$', 'P Nov', 'R Nov', '$\Gamma_\mathrm{boundary}$',
                 '$\Gamma_\mathrm{structure}$', '$\Psi_\mathrm{intensity}$', '$\Psi_\mathrm{color}$',
                 '$\Psi_\mathrm{pan}$', '$\Psi_\mathrm{tilt}$', '$\Gamma_\mathrm{novelty}$']


    vl_dic = {}
    for dsi in range(len(sets)):
        s_label = sets[dsi]['name']
        vl_dic[s_label] = []
        for k in range(len(var_names)):
            vl_dic[s_label].append([])

        dataset = data[dsi]
        N = len(dataset)

        for j in range(N):
            v = evals[dsi].eval_single_datapoint(
                j, None)

            for k in range(len(var_names)):
                a = v[k]
                if var_names[k] == '$\Gamma_\mathrm{beat \leftrightarrow peak}$' or var_names[
                    k] == '$\Gamma_\mathrm{beat \leftrightarrow valley}$':
                    if v[k + 2] > 0:
                        a /= v[k + 2]
                    else:
                        continue
                vl_dic[s_label][k].append(a)

    idx = [0, 1, 4, 9, 14]  # which eval scores are interesiting
    x = []

    final_data = []
    means = []
    stds = []

    for i in idx:
        x.append(var_names[i])

    for j in range(len(data)):
        f_d = []
        m = []
        std = []
        name = sets[j]['name']
        for i in idx:
            val = vl_dic[name][i]
            f_d.append(val)
            m = np.mean(val)
            std = np.std(val)

            logger.log(
                f"Var {var_names[i]} for Subset {name}: mean {m}, std {std}")

        final_data.append(f_d)
        means.append(m)
        stds.append(std)
    pos_mid = np.array(range(len(x))) * x_tick_distance
    fig, ax = plt.subplots()
    boxes = []

    plt.axhline(y=0, color='grey', linestyle='-')

    for j in range(len(data)):
        pos = list(pos_mid + sets[j]['bp_offset'])
        bp = ax.boxplot(final_data[j], positions=pos, patch_artist=True)
        set_bp_edge(bp, sets[j]['bp_color'])
        set_bp_fill(bp, 'white')
        boxes.append(bp['boxes'][0])


    ax.legend(boxes, names)
    plt.xticks(pos_mid, x)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=12)
    ax.set_ylim([-1,1])

    plt.show()

if __name__ == '__main__':
    main()