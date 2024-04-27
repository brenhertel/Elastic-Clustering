import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as mcolors

import matplotlib as mpl
mpl.rc('font', family='Times New Roman', size=18)

LAMBDA = 0.0052

#function to cluster data to nodes in the elastic map
#Given nodes and data, it returns the clusters for each node (as a list of indices) as well as C, the sum of each cluster, which is needed for optimization of the map.
# data: data to be clustered
# nodes: nodes of the elastic map
def associate(data, nodes):
    n_pts, n_dims = np.shape(data)
    n_nodes, n_dims = np.shape(nodes)
    clusters = [-1 for _ in range(n_pts)]
    for i in range(n_pts):
        clusters[i] = np.argmin([np.linalg.norm(data[i] - nodes[j]) for j in range(n_nodes)])
    C = np.zeros((n_nodes, n_dims))
    for i in range(n_nodes):
        sum = C[i]
        for j in range(n_pts):
            if (clusters[j] == i):
                sum = sum + data[j]
        #C[i] = sum / n_pts
        C[i] = sum
    return clusters, C
    
#function to calculate the A matrix used in optimization
# clusters: clusters returned from clustering/expectation step
# E: E matrix initially calculated from node-to-node connections
def calc_A(clusters, E):
    A = np.copy(E)
    n_nodes, _ = np.shape(A)
    n_pts = len(clusters)
    C = np.array(clusters)
    for i in range(n_nodes):
        A[i, i] = A[i, i] + np.sum(C == i)
    return A
    
#function to optimize an elastic map (Expectation-Maximization)
# data: data to be clustered
# nodes: initial guess for map
# stretch: stretching constant, higher values prompt nodes to be farther from each other
def optimize_map(data, nodes, stretch=0.005):
    #initialization
    lmda = -stretch
    n_data, n_dims = np.shape(data)
    n_nodes, n_dims = np.shape(nodes)
    E = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            E[i, i] = E[i, i] + lmda
            E[j, j] = E[j, j] + lmda
            E[i, j] = E[i, j] - lmda
            E[j, i] = E[j, i] - lmda
    
    #expectation
    clusters, C = associate(data, nodes)
    #maximization
    A = calc_A(clusters, E)
    new_nodes = np.linalg.lstsq(A, C, rcond=None)[0]
    #repeat until convergence
    new_clusters, C = associate(data, new_nodes)
    iter = 1
    #print([iter, calc_nrg(data, new_nodes, new_clusters)])
    while (new_clusters != clusters) and (iter < 20):
        clusters = new_clusters
        nodes = new_nodes
        A = calc_A(clusters, E)
        new_nodes = np.linalg.lstsq(A, C, rcond=None)[0]
        new_clusters, C = associate(data, new_nodes)
        iter = iter + 1
        #print([iter, calc_nrg(data, new_nodes, new_clusters)])
    return new_nodes, new_clusters

#function to calculate the energy of a given map
# data: data to be clustered
# nodes: final optimized nodes
# clusters: final optimized clusters
# lmda: stretching constant
def calc_nrg(data, nodes, clusters, lmda):
    n_pts, n_dims = np.shape(data)
    n_nodes, n_dims = np.shape(nodes)
    Uy = 0.
    for i in range(n_pts):
        Uy = Uy + np.linalg.norm(data[i] - nodes[clusters[i]])**2
    Ue = 0.
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            if (i != j):
                Ue = Ue + np.linalg.norm(nodes[i] - nodes[j])**2
    #return (Uy / n_pts) + (lmda * Ue)
    return Uy + (lmda * Ue)
    

#class object for handling elastic clustering
class elmap_class(object):
    
    #object initialization
    def __init__(self, initial_points=[], nlabels=1, lmbda=0.005):
        self.colors = list(mcolors.TABLEAU_COLORS.keys())
        self.point_list = initial_points
        self.cluster_list = []
        self.cluster_centers = []
        self.num_points = len(self.point_list)
        self.num_labels = nlabels
        self.nrg = -1
        self.stretch = lmbda
        if self.num_points > 0:
            self.cluster_points()

    #getter functions
    def get_nrg(self):
        return self.nrg

    def get_clusters(self):
        return self.cluster_list

    def get_centers(self):
        return self.cluster_centers
        
    def get_num_labels(self):
        return self.num_labels

    #function to cluster given points, iteratively clusters with increasing number of centers until energy minimum is found
    def cluster_points(self):
        if len(self.cluster_centers) == 0:
            #first time clustering, initialize map guess
            self.cluster_centers = self.point_list[np.random.choice(self.num_points, size=self.num_labels, replace=False)]
        
        self.cluster_centers, self.cluster_list = optimize_map(self.point_list, self.cluster_centers, stretch=self.stretch)
        self.nrg = calc_nrg(self.point_list, self.cluster_centers, self.cluster_list, lmda=self.stretch)
        
        check_next_cluster = True
        while (check_next_cluster):
            #check if adding another cluster reduces energy
            temp_cluster_centers = copy.deepcopy(self.cluster_centers)
            temp_cluster_centers = np.vstack((temp_cluster_centers, self.point_list[-1]))
            temp_cluster_centers, temp_cluster_list = optimize_map(self.point_list, temp_cluster_centers, stretch=self.stretch)
            temp_nrg = calc_nrg(self.point_list, temp_cluster_centers, temp_cluster_list, lmda=self.stretch)
            if (temp_nrg < self.nrg):
                #better to add another cluster
                self.cluster_centers = temp_cluster_centers
                self.cluster_list = temp_cluster_list
                self.nrg = temp_nrg
            else:
                #stop checking to see if another cluster should be added
                check_next_cluster = False
        self.num_labels = len(self.cluster_centers)
    
    #add points to points already stored in object and recluster
    def add_points(self, points):
        if self.num_points == 0:
            self.point_list = points
        else:
            self.point_list = np.vstack((self.point_list, points))  
        self.num_points = len(self.point_list)          
        self.cluster_points()
        
    #display clustering results
    def display_info(self):
        print('Points:')
        print(self.point_list)
        print('Clusters:')
        print(self.cluster_list)
        print('Centers:')
        print(self.cluster_centers)
      
    #plot clustering results, each cluster is its own color    
    def plot(self, mode='show', title='', fpath=''):
        n_pts, n_dims = np.shape(self.point_list)
        if n_dims == 2:
            fig = plt.figure()
            plt.axis('equal')
            plt.title(title)
            for i in range(self.num_labels):
                plt.plot(self.cluster_centers[i][0], self.cluster_centers[i][1], color=self.colors[i % len(self.colors)], marker='*', ms=10)
                plt.scatter(self.point_list[np.array(self.cluster_list) == i][:, 0], self.point_list[np.array(self.cluster_list) == i][:, 1], color=self.colors[i], marker='.', s=8)
        elif n_dims == 3:
            print('Not yet implemented!')
        else:
            print('Plotting not supported in this dimension!')
        if mode == 'show':
            plt.show()
        else:
            plt.savefig(fpath + '/' + title + '.png', dpi=300, bbox_inches='tight')
            plt.close('all')

#cluster on 3 centers with iteratively different numbers of points            
def points_main():
    PIC_FPATH = '../pictures/sim_data'
    np.random.seed(1)
    num_points_list = [10, 20, 30]
    n_dims = 2
    centers = [[0, 0], [1, 2], [2, -1]]
    std_dev = 0.5
    for n_pts in num_points_list:
        pts1 = np.random.normal(loc=centers[0], scale=std_dev, size=(n_pts, n_dims))
        pts2 = np.random.normal(loc=centers[1], scale=std_dev, size=(n_pts, n_dims))
        pts3 = np.random.normal(loc=centers[2], scale=std_dev, size=(n_pts, n_dims))
        pts = np.vstack((pts1, pts2, pts3))
        EMclass = elmap_class(pts, lmbda=0.4)
        #EMclass.display_info()
        EMclass.plot(mode='show', title=str(n_pts) + ' Samples per Center', fpath=PIC_FPATH)
    
#cluster on 3 centers, iteratively adding points
def iterative_main():
    PIC_FPATH = '../pictures/elmap_classifier/basics'
    np.random.seed(1)
    n_pts = 60
    n_dims = 2
    n_nodes = 1
    pts1 = np.random.normal(loc=[5, 5], size=(n_pts//3, n_dims))
    pts2 = np.random.normal(loc=[0, 0], size=(n_pts//3, n_dims))
    pts3 = np.random.normal(loc=[7, 0], size=(n_pts//3, n_dims))
    #pts = np.vstack((pts1, pts2, pts3))
    EMclass = elmap_class(pts1, lmbda=0.4)
    EMclass.plot(mode='show', title='20 points - 1 True Center', fpath=PIC_FPATH)
    EMclass.add_points(pts2)
    EMclass.plot(mode='show', title='40 points - 2 True Centers', fpath=PIC_FPATH)
    EMclass.add_points(pts3)
    EMclass.plot(mode='show', title='60 points - 3 True Centers', fpath=PIC_FPATH)
    
#cluster with 3 centers on random uniform points (no iterative process)
def random_uniform_main():
    np.random.seed(1)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    n_pts = 20
    n_dims = 2
    n_nodes = 3
    pts = np.random.uniform(size=(n_pts, n_dims))
    nodes = pts[np.random.choice(n_pts, n_nodes, replace=False)]
    
    new_nodes, clusters = optimize_map(pts, nodes)
    
    plt.figure()
    for i in range(n_pts):
        plt.plot(pts[i, 0], pts[i, 1], colors[clusters[i]] + '.')
    plt.plot(new_nodes[:, 0], new_nodes[:, 1], 'ko')
    plt.show()
    

if __name__ == '__main__':
    print('Clustering on random points')
    random_uniform_main()
    print('Iteratively adding new points to cluster')
    iterative_main()
    print('Iteratively increasing the number of points')
    points_main()