import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as scio
import math
import seaborn as sns
import hdbscan
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn import manifold
import heapq
from numpy.random import shuffle

if __name__ == '__main__':
    for ii in range(10):
        t1 = time.time()
        file_name = 'Mosaic_150_150_360_60.mat'

        mat_data = scio.loadmat(file_name)
        X = mat_data['X'].astype(np.float32)
        S = mat_data['S'].astype(np.float32)
        B = mat_data['B'].astype(np.float32)
        data = np.reshape(X, (-1, X.shape[2]))

        bg_num = 450
        del_rate = 0.05

        dim = 30
        # Data dimension reduction，PCA
        pca = PCA(n_components=dim)
        pca.fit(data)
        data_new = pca.transform(data)
        # HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=8)

        clusterer.fit(data_new)
        label = clusterer.labels_
        label[clusterer.probabilities_ < 0.5] = -1
        count = np.count_nonzero(label == -1)

        unique_values, counts = np.unique(label, return_counts=True)
        dict={}
        # print results
        for value, count in zip(unique_values, counts):
            # print(f"{value}：{count} times")
            dict[count] =value


        for i in range(len(clusterer.exemplars_)):
            clusterer.exemplars_[i] = np.unique(clusterer.exemplars_[i], axis=0)

        core_counts = []
        core_pts_cnt = 0
        for i in clusterer.exemplars_:
            core_counts.append(len(i))
        core_pts_cnt = sum(core_counts)
        print(core_pts_cnt)

        sort_idx_core_counts = np.argsort(core_counts)

        BG = np.array([])
        BG = BG.reshape((0, np.shape(data)[1]))
        choose_idx = []
        # Avoid noise and clusters with the most points
        for i in range(len(sort_idx_core_counts) - 1):
            choose_num = math.ceil(bg_num * len(clusterer.exemplars_[sort_idx_core_counts[i]]) / core_pts_cnt)
            choose_num = min(len(clusterer.exemplars_[sort_idx_core_counts[i]]), choose_num)
            # np.random.shuffle(clusterer.exemplars_[sort_idx_core_counts[i]])
            clusterer.exemplars_[sort_idx_core_counts[i]] = np.unique(clusterer.exemplars_[sort_idx_core_counts[i]], axis=0)

            for j in range(choose_num):
                BG = np.vstack(
                    [BG, data[
                        np.where((data_new == clusterer.exemplars_[sort_idx_core_counts[i]][j]).all(axis=1))[0][0]
                    ]
                     ][:])
                choose_idx.append(
                    np.where((data_new == clusterer.exemplars_[sort_idx_core_counts[i]][j]).all(axis=1))[0][0])

        # Largest cluster
        clusterer.exemplars_[sort_idx_core_counts[len(clusterer.exemplars_) - 1]] = np.unique(
            clusterer.exemplars_[sort_idx_core_counts[len(clusterer.exemplars_) - 1]], axis=0)
        for j in range(
                min(bg_num - len(BG), len(clusterer.exemplars_[sort_idx_core_counts[len(clusterer.exemplars_) - 1]]))):
            BG = np.vstack([BG, data[np.where(
                (data_new == clusterer.exemplars_[sort_idx_core_counts[len(clusterer.exemplars_) - 1]][j]).all(axis=1))[0][
                0]]][:])
            choose_idx.append(np.where(
                (data_new == clusterer.exemplars_[sort_idx_core_counts[len(clusterer.exemplars_) - 1]][j]).all(axis=1))[0][
                                  0])


        if bg_num > len(BG):
            rest_num = bg_num - len(BG)
            counts = np.delete(counts, 0)
            unique_values = np.delete(unique_values, 0)
            sort_idx_counts = np.argsort(counts)
            for i in range(len(sort_idx_counts) - 1):
                if bg_num > len(BG):
                    choose_num = math.ceil(rest_num * counts[sort_idx_counts[i]] / sum(counts))
                    choose_num = min(counts[i], choose_num)

                    # Select the unselected ones
                    j_choose = 0
                    for each_label_bg in data[label == dict[counts[sort_idx_counts[i]]]]:
                        # if j_choose < choose_num and each_label_bg not in BG:
                        if j_choose < choose_num and bg_num > len(BG):
                            if np.any(np.all(data == each_label_bg, axis=1)):
                                BG = np.vstack([BG, each_label_bg])
                                choose_idx.append(np.where((data == each_label_bg)))
                                j_choose += 1
                        else:
                            break

            choose_num = bg_num-len(BG)
            # Largest cluster
            if choose_num > 0:
                j_choose = 0
                for each_label_bg in data[label == dict[counts[sort_idx_counts[-1]]]]:
                    if j_choose < choose_num and bg_num > len(BG):
                        if np.any(np.all(data == each_label_bg, axis=1)):
                            BG = np.vstack([BG, each_label_bg])
                            choose_idx.append(np.where((data == each_label_bg)))
                            j_choose += 1
                    else:
                        break

        print(len(BG))


        # scio.savemat('BG'+file_name+str(bg_num[k])+'.mat', {'BG': BG.T})
        for j in S.T:
            dist = []
            for i in BG:
                # The distance between the core point of each cluster and the target point was calculated
                dist.append(np.linalg.norm(j - i))

            num_to_remove = int(len(dist) * del_rate / len(S.T))
            to_remove = heapq.nsmallest(num_to_remove, dist)
            indexes_to_remove = [i for i, x in enumerate(dist) if x in to_remove]
            BG = np.delete(BG, indexes_to_remove, axis=0)

            # Ensure that each deletion operation does not affect the position of previously undeleted elements
            for i in sorted(indexes_to_remove, reverse=True):
                del choose_idx[i]

        # plt.plot(BG.T)
        # scio.savemat('BG'+file_name+str(bg_num[k])+'_purify.mat', {'BG_purify': BG.T})


        t2 = time.time()
        print('time:', round(t2 - t1, 3), 's')
