import numpy as np
from elastic_clustering import elmap_class
import matplotlib.pyplot as plt
from utils import *
from itertools import chain, combinations
from sklearn.cluster import AgglomerativeClustering

#returns powerset of iterable object
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

# defined features    
reaching_demos, pressing_demos, pushing_demos, writing_demos = read_all_RAIL()
reaching_demo_comp = reaching_demos[28]
pushing_demo_comp = pushing_demos[29] 
writing_demo_comp = writing_demos[0]
    
def reaching_similarity(traj):
    return angular_similarity(reaching_demo_comp, traj)

def pushing_similarity(traj):
    return angular_similarity(pushing_demo_comp, traj)

def writing_similarity(traj):
    return angular_similarity(writing_demo_comp, traj)
    
#function to normalize feature returns
# list of features: list of results from each feature
def to_array_normalized(list_of_features):
    feature_array = np.array(list_of_features)
    n_demos, n_features = np.shape(feature_array)
    #range map each column from 0 to 1
    for i in range(n_features):
        out_max = 1.
        out_min = 0.
        feature_array[:, i] = (feature_array[:, i] - np.min(feature_array[:, i])) * (out_max - out_min) / (np.max(feature_array[:, i]) - np.min(feature_array[:, i])) + out_min
    return feature_array
    
#Featurizer object to turn demonstrations into features and cluster
class Featurizer(object):

    #object initialization
    def __init__(self, demos=[], names=[], stretch=0.005):
        self.feature_list = []
        self.feature_names = []
        
        self.demo_list = demos
        self.name_list = names
        self.lmbda = stretch
        self.demo_feature_list = [[x(demo) for x in self.feature_list] for demo in self.demo_list]
        self.classifier = elmap_class(to_array_normalized(self.demo_feature_list), lmbda=self.lmbda)
        
    #add demos to the object and recluster
    def add_demos(self, demos, names):
        self.demo_list = self.demo_list + demos
        self.name_list = self.name_list + names
        new_feature_list = [[x(demo) for x in self.feature_list] for demo in demos]
        self.demo_feature_list = self.demo_feature_list + new_feature_list
        self.classifier = elmap_class(to_array_normalized(self.demo_feature_list), lmbda=self.lmbda)
    
    #change the features on which to cluster    
    def change_features(self, features, names):
        self.feature_list = features
        self.feature_names = names
        self.demo_feature_list = [[x(demo) for x in self.feature_list] for demo in self.demo_list]
        self.classifier = elmap_class(to_array_normalized(self.demo_feature_list), lmbda=self.lmbda)
     
    #list the current demonstrations and their clusters
    def list_clusters(self):
        clusters = self.classifier.get_clusters()
        num_labels = self.classifier.get_num_labels()
        for i in range(num_labels):
            print('')
            print('-----------------')
            print('Cluster ' + str(i))
            print('-----------------')
            for j in range(len(clusters)):
                if clusters[j] == i:
                    print(self.name_list[j])
     
    #helper function to get clusters with named demonstrations
    def get_cluster_name_list(self):
        clusters = self.classifier.get_clusters()
        num_labels = self.classifier.get_num_labels()
        cluster_names = []
        for i in range(num_labels):
            in_cluster_names = []
            for j in range(len(clusters)):
                if clusters[j] == i:
                    in_cluster_names.append(self.name_list[j])
            if len(in_cluster_names) > 0:
                cluster_names.append(in_cluster_names)
        return cluster_names
        
    #helper function to get a list of list of clusters
    def clusters_to_list_of_list(self):
        clusters = self.classifier.get_clusters()
        num_labels = self.classifier.get_num_labels()
        cluster_list_list = [[] for i in range(num_labels)]
        for i in range(len(clusters)):
            for j in range(num_labels):
                if clusters[i] == j:
                    cluster_list_list[j].append(i)
        return cluster_list_list
    
    #plot the results of clustering using trajectories
    def plot_clustered_demos(self):
        clusters = self.classifier.get_clusters()
        n_pts, n_dims = np.shape(self.demo_list[0])
        if n_dims == 2:
            plt.figure()
            for i in range(len(self.demo_list)):
                plt.plot(self.demo_list[i][:, 0], self.demo_list[i][:, 1], color=self.classifier.colors[clusters[i]])
        else:
            print('Dimension not yet implemented!')
        plt.show()
        
    #plot results of clustering, but instead of plotting on top of each other plot each cluster in its own row
    def plot_clustered_demos_grid(self):
        n_demos = len(self.demo_list)
        n_clusters = self.classifier.get_num_labels()
        clusters = self.classifier.get_clusters()
        n_pts, n_dims = np.shape(self.demo_list[0])
        max_cluster_size = 0
        cll = self.clusters_to_list_of_list()
        print(cll)
        for i in range(n_clusters):
            if len(cll[i]) > max_cluster_size:
                max_cluster_size = len(cll[i])
        print(n_demos, n_clusters, n_pts, n_dims, max_cluster_size)
        if n_dims == 2:
            fig, axs = plt.subplots(ncols=n_clusters, nrows=max_cluster_size)
            for i in range(n_clusters):
                for j in range(len(cll[i])):
                    print(i, j, cll[i], cll[i][j], clusters[cll[i][j]])
                    print(self.classifier.colors[clusters[cll[i][j]]])
                    axs[j, i].plot(self.demo_list[cll[i][j]][:, 0], self.demo_list[cll[i][j]][:, 1], color=self.classifier.colors[clusters[cll[i][j]]])
        else:
            print('Dimension not yet implemented!')
        plt.show()
        
    #plot the results of featurizing demonstrations
    def plot_features(self):
        clusters = self.classifier.get_clusters()
        for i in range(len(self.feature_list)):
            plt.figure()
            plt.title(self.feature_names[i])
            for j in range(len(self.demo_list)):
                plt.plot(j, self.demo_feature_list[j][i], '.', color=self.classifier.colors[clusters[j]])
        plt.show()

#test against agglomerative clustering    
def RAIL_test_agglomerative():
    np.random.seed(1)
    reaching_demos, pressing_demos, pushing_demos, writing_demos = read_all_RAIL()
    reaching_names = ['reaching'] * len(reaching_demos)
    pushing_names = ['pushing'] * len(pushing_demos)
    writing_names = ['writing'] * len(writing_demos)
    
    all_demos = reaching_demos + pushing_demos + writing_demos
    all_names = reaching_names + pushing_names + writing_names
    
    features = [reaching_similarity, pushing_similarity, writing_similarity]
    feature_names = ['reaching_similarity', 'pushing_similarity', 'writing_similarity']
    
    ft = Featurizer(all_demos, all_names, stretch=1.2)
    ft.change_features(features, feature_names)
    cl_name_list = ft.get_cluster_name_list()
    print('Elastic Clustering Reaching/Pushing/Writing totals')
    for cluster in cl_name_list:
        num_reaching = 0
        num_pushing = 0
        num_writing = 0
        for name in cluster:
            if name == 'reaching':
                num_reaching = num_reaching + 1
            if name == 'pushing':
                num_pushing = num_pushing + 1
            if name == 'writing':
                num_writing = num_writing + 1
        print([num_reaching, num_pushing, num_writing])
    elmap_labels = ft.classifier.get_clusters()
    
    clustering = AgglomerativeClustering(n_clusters=3).fit(np.array(ft.demo_feature_list))
    agglom_clusters = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(clustering.labels_)):
        if i < 60:
            agglom_clusters[clustering.labels_[i]][0] += 1
        if i >= 60 and i < 120:
            agglom_clusters[clustering.labels_[i]][1] += 1
        if i >= 120:
            agglom_clusters[clustering.labels_[i]][2] += 1
    print('Agglomerative Clustering Reaching/Pushing/Writing totals')
    print(agglom_clusters)
        
    agglomerative_labels = clustering.labels_ + 1
    
    plt.rcParams['figure.figsize'] = (9, 7)
    fig = plt.figure()
    
    ax = fig.add_subplot(3, 3, 1, projection='3d')
    ax.set_title('Cluster 1')
    ax.set_zlabel('Ideal')
    for i in range(len(reaching_demos)):
        reaching_demos[i] = reaching_demos[i] - reaching_demos[i][-1, :]
        ax.plot(reaching_demos[i][:, 0], reaching_demos[i][:, 1], reaching_demos[i][:, 2], 'r', lw=1)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=35, elev=28)
    
    ax = fig.add_subplot(3, 3, 2, projection='3d')
    ax.set_title('Cluster 2')
    for i in range(len(pushing_demos)):
        pushing_demos[i] = pushing_demos[i] - pushing_demos[i][-1, :]
        ax.plot(pushing_demos[i][:, 0], pushing_demos[i][:, 1], pushing_demos[i][:, 2], 'g', lw=1)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=35, elev=28)
    
    ax = fig.add_subplot(3, 3, 3, projection='3d')
    ax.set_title('Cluster 3')
    for i in range(len(writing_demos)):
        writing_demos[i] = writing_demos[i] - writing_demos[i][-1, :]
        ax.plot(writing_demos[i][:, 0], writing_demos[i][:, 1], writing_demos[i][:, 2], 'b', lw=1)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=35, elev=28)
    
    ax = fig.add_subplot(3, 3, 4, projection='3d')
    ax.set_zlabel('Elastic \n Clustering')
    for i in range(len(reaching_demos)):
        if elmap_labels[i] == 0:
            reaching_demos[i] = reaching_demos[i] - reaching_demos[i][-1, :]
            ax.plot(reaching_demos[i][:, 0], reaching_demos[i][:, 1], reaching_demos[i][:, 2], 'r', lw=1)
    for i in range(len(pushing_demos)):
        if elmap_labels[i + len(reaching_demos)] == 0:
            pushing_demos[i] = pushing_demos[i] - pushing_demos[i][-1, :]
            ax.plot(pushing_demos[i][:, 0], pushing_demos[i][:, 1], pushing_demos[i][:, 2], 'g', lw=1)
    for i in range(len(writing_demos)):
        if elmap_labels[i + len(reaching_demos) + len(pushing_demos)] == 0:
            writing_demos[i] = writing_demos[i] - writing_demos[i][-1, :]
            ax.plot(writing_demos[i][:, 0], writing_demos[i][:, 1], writing_demos[i][:, 2], 'b', lw=1)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=35, elev=28)
    
    ax = fig.add_subplot(3, 3, 5, projection='3d')
    for i in range(len(reaching_demos)):
        if elmap_labels[i] == 1:
            reaching_demos[i] = reaching_demos[i] - reaching_demos[i][-1, :]
            ax.plot(reaching_demos[i][:, 0], reaching_demos[i][:, 1], reaching_demos[i][:, 2], 'r', lw=1)
    for i in range(len(pushing_demos)):
        if elmap_labels[i + len(reaching_demos)] == 1:
            pushing_demos[i] = pushing_demos[i] - pushing_demos[i][-1, :]
            ax.plot(pushing_demos[i][:, 0], pushing_demos[i][:, 1], pushing_demos[i][:, 2], 'g', lw=1)
    for i in range(len(writing_demos)):
        if elmap_labels[i + len(reaching_demos) + len(pushing_demos)] == 1:
            writing_demos[i] = writing_demos[i] - writing_demos[i][-1, :]
            ax.plot(writing_demos[i][:, 0], writing_demos[i][:, 1], writing_demos[i][:, 2], 'b', lw=1)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=35, elev=28)
    
    ax = fig.add_subplot(3, 3, 6, projection='3d')
    for i in range(len(reaching_demos)):
        if elmap_labels[i] == 2:
            reaching_demos[i] = reaching_demos[i] - reaching_demos[i][-1, :]
            ax.plot(reaching_demos[i][:, 0], reaching_demos[i][:, 1], reaching_demos[i][:, 2], 'r', lw=1)
    for i in range(len(pushing_demos)):
        if elmap_labels[i + len(reaching_demos)] == 2:
            pushing_demos[i] = pushing_demos[i] - pushing_demos[i][-1, :]
            ax.plot(pushing_demos[i][:, 0], pushing_demos[i][:, 1], pushing_demos[i][:, 2], 'g', lw=1)
    for i in range(len(writing_demos)):
        if elmap_labels[i + len(reaching_demos) + len(pushing_demos)] == 2:
            writing_demos[i] = writing_demos[i] - writing_demos[i][-1, :]
            ax.plot(writing_demos[i][:, 0], writing_demos[i][:, 1], writing_demos[i][:, 2], 'b', lw=1)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=35, elev=28)
    
    ax = fig.add_subplot(3, 3, 7, projection='3d')
    ax.set_zlabel('Agglomerative \n Clustering')
    for i in range(len(reaching_demos)):
        if agglomerative_labels[i] == 3:
            reaching_demos[i] = reaching_demos[i] - reaching_demos[i][-1, :]
            reach, = ax.plot(reaching_demos[i][:, 0], reaching_demos[i][:, 1], reaching_demos[i][:, 2], 'r', lw=1)
    for i in range(len(pushing_demos)):
        if agglomerative_labels[i + len(reaching_demos)] == 3:
            pushing_demos[i] = pushing_demos[i] - pushing_demos[i][-1, :]
            push, = ax.plot(pushing_demos[i][:, 0], pushing_demos[i][:, 1], pushing_demos[i][:, 2], 'g', lw=1)
    for i in range(len(writing_demos)):
        if agglomerative_labels[i + len(reaching_demos) + len(pushing_demos)] == 3:
            writing_demos[i] = writing_demos[i] - writing_demos[i][-1, :]
            write, = ax.plot(writing_demos[i][:, 0], writing_demos[i][:, 1], writing_demos[i][:, 2], 'b', lw=1)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=35, elev=28)
    
    ax = fig.add_subplot(3, 3, 8, projection='3d')
    for i in range(len(reaching_demos)):
        if agglomerative_labels[i] == 1:
            reaching_demos[i] = reaching_demos[i] - reaching_demos[i][-1, :]
            reach, = ax.plot(reaching_demos[i][:, 0], reaching_demos[i][:, 1], reaching_demos[i][:, 2], 'r', lw=1)
    for i in range(len(pushing_demos)):
        if agglomerative_labels[i + len(reaching_demos)] == 1:
            pushing_demos[i] = pushing_demos[i] - pushing_demos[i][-1, :]
            push, = ax.plot(pushing_demos[i][:, 0], pushing_demos[i][:, 1], pushing_demos[i][:, 2], 'g', lw=1)
    for i in range(len(writing_demos)):
        if agglomerative_labels[i + len(reaching_demos) + len(pushing_demos)] == 1:
            writing_demos[i] = writing_demos[i] - writing_demos[i][-1, :]
            write, = ax.plot(writing_demos[i][:, 0], writing_demos[i][:, 1], writing_demos[i][:, 2], 'b', lw=1)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=35, elev=28)
    
    ax = fig.add_subplot(3, 3, 9, projection='3d')
    for i in range(len(reaching_demos)):
        if agglomerative_labels[i] == 2:
            reaching_demos[i] = reaching_demos[i] - reaching_demos[i][-1, :]
            reach, = ax.plot(reaching_demos[i][:, 0], reaching_demos[i][:, 1], reaching_demos[i][:, 2], 'r', lw=1)
    for i in range(len(pushing_demos)):
        if agglomerative_labels[i + len(reaching_demos)] == 2:
            pushing_demos[i] = pushing_demos[i] - pushing_demos[i][-1, :]
            push, = ax.plot(pushing_demos[i][:, 0], pushing_demos[i][:, 1], pushing_demos[i][:, 2], 'g', lw=1)
    for i in range(len(writing_demos)):
        if agglomerative_labels[i + len(reaching_demos) + len(pushing_demos)] == 2:
            writing_demos[i] = writing_demos[i] - writing_demos[i][-1, :]
            write, = ax.plot(writing_demos[i][:, 0], writing_demos[i][:, 1], writing_demos[i][:, 2], 'b', lw=1)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=35, elev=28)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.1)  
    fig.legend([reach, push, write], ['Reaching', 'Pushing', 'Writing'], loc='lower center', ncol=3)
    plt.show()
    
if __name__ == '__main__':
    RAIL_test_agglomerative()