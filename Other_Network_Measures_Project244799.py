'''
Stephanie L. Mathias, August 2022. Masters Project: Network Parameters of Synesthete Connectomes.

This script attempts to curate 4 network parameters for the full matrices: average node degree, clustering coefficient, small worldness and connection density.
The values from this didn't appear valid so no subsequent data was used in the project.

Some functions in this script are taken and adapted from:
https://notebooks.githubusercontent.com/view/ipynb?color_mode=auto&commit=cc7cbcdd0b615ecbccdb8f71625a3f0dc5bd1e3f&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f676973742f44694779742f33633036313236653637386534623335616664656334336134393433393137642f7261772f636337636263646430623631356563626363646238663731363235613366306463356264316533662f67726170685f6d656173757265735f696d706c656d656e746174696f6e2e6970796e62&logged_in=false&nwo=DiGyt%2F3c06126e678e4b35afdec43a4943917d&path=graph_measures_implementation.ipynb&repository_id=106798930&repository_type=Gist [Accessed 02 August 2022]
'''

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.preprocessing import normalize, MinMaxScaler

# file paths
other_set_file = r"_insert_analytical_set_filepath_"

# import files
other_set = pd.read_csv(other_set_file)

# normalise
other_norm = other_set.copy()
scaler=MinMaxScaler(feature_range=(0,1))
cols = [str(x) for x in range(1,129601)]
for c in cols:
    other_norm[c] = scaler.fit_transform(other_norm[[c]])

#Function: triangle number
def triangle_num(matrix):
    n_nodes = len(matrix)

    mean_vector = np.empty([n_nodes])
    for i in range(n_nodes):
        triangles = np.array([[matrix[i, j] * matrix[i, k] * matrix[j, k] for j in range(n_nodes)] for k in range(n_nodes)])**(1/3)
        mean_vector[i] = (1/2) * np.sum(triangles, axis=(0,1))
    return mean_vector

#Function: calculate unweighted node degree centrality
def unw_node_degree(matrix):
    return np.sum(np.ceil(matrix), axis=-1)

#Function: calculate average degree centrality
def av_node_degree(matrix):
    nodes_degs = np.sum(matrix, axis=-1)
    return np.mean(nodes_degs)

#Function: calculate clustering coefficient
def clust_coeff(matrix):
    n = len(matrix)
    t = triangle_num(matrix)
    k = unw_node_degree(matrix)
    return (1/n) * np.sum((2 * t)/(k * (k - 1)))

#Function: calculate small worldness
def random_reference_nx(matrix, niter=5):
    G = nx.convert_matrix.from_numpy_array(matrix)
    G_ref = nx.algorithms.smallworld.random_reference(G, niter=niter)
    return nx.convert_matrix.to_numpy_array(G_ref)

#Function: get inverse of a matrix
def inv_matrix(matrix):
    new_matrix = 1 - matrix.copy()
    return remove_self_connections(new_matrix)

#Function: get weighted shortest path
def weighted_shortest_path(matrix):
    matrix = inv_matrix(matrix)
    n_nodes = len(matrix)
    distances = np.empty([n_nodes, n_nodes])
    for i in range(n_nodes):
        for j in range(n_nodes):
            distances[i,j] = matrix[i, j]

    for i in range(n_nodes):
        distances[i,i] = 0

    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if distances[i, j] > distances[i, k] + distances[k, j]:
                    distances[i, j] = distances[i, k] + distances[k, j]
    return distances

#Function: get weighted clustering coefficient
def weighted_clustering_coeff_z(matrix):
    n_nodes = len(matrix)
    ccs = []
    for i in range(n_nodes):
        upper = np.sum([[matrix[i,j] * matrix[j,k] * matrix[i,k] for k in range(n_nodes)] for j in range(n_nodes)])
        lower = np.sum([[matrix[i,j] * matrix[i,k] for k in range(n_nodes) if j!=k] for j in range(n_nodes)])
        ccs.append(upper/lower)
    return np.mean(ccs)

#Function: remove self connections
def remove_self_connections(matrix):
    for i in range(len(matrix)):
        matrix[i, i] = 0
    return matrix

#Function: get weighted characteristic path length
def weighted_characteristic_path_length(matrix):
    n_nodes = len(matrix)
    min_distances = weighted_shortest_path(matrix)
    sum_vector = np.empty(n_nodes)
    for i in range(n_nodes):
        sum_vector[i] = (1/(n_nodes-1)) * np.sum([min_distances[i, j] for j in range(n_nodes) if j != i])

    return (1/n_nodes) * np.sum(sum_vector)

#Function: create a random matrix
def random_reference(G, niter=1, D=None, seed=np.random.seed(np.random.randint(0, 2**30))):

    from networkx.utils import cumulative_distribution, discrete_sequence


    G = G.copy()
    keys = [i for i in range(len(G))]
    degrees = weighted_node_degree(G)
    cdf = cumulative_distribution(degrees)  # cdf of degree

    nnodes = len(G)
    nedges = nnodes *(nnodes - 1) // 2 # NOTE: assuming full connectivity
    if D is None:
        D = np.zeros((nnodes, nnodes))
        un = np.arange(1, nnodes)
        um = np.arange(nnodes - 1, 0, -1)
        u = np.append((0,), np.where(un < um, un, um))

        for v in range(int(np.ceil(nnodes / 2))):
            D[nnodes - v - 1, :] = np.append(u[v + 1 :], u[: v + 1])
            D[v, :] = D[nnodes - v - 1, :][::-1]

    niter = niter * nedges
    ntries = int(nnodes * nedges / (nnodes * (nnodes - 1) / 2))
    swapcount = 0

    for i in range(niter):
        n = 0
        while n < ntries:
            # pick two random edges without creating edge list
            # choose source node indices from discrete distribution
            (ai, bi, ci, di) = discrete_sequence(4, cdistribution=cdf, seed=seed)
            if len(set((ai, bi, ci, di))) < 4:
                continue  # picked same node twice
            a = keys[ai]  # convert index to label
            b = keys[bi]
            c = keys[ci]
            d = keys[di]


            # only swap if we get closer to the diagonal

            ab = G[a, b]
            cd = G[c, d]
            G[a, b] = cd
            G[b, a] = cd
            G[c, d] = ab
            G[d, c] = ab

            swapcount += 1
            break

    return G

#Function: clustering coefficient sigma
def weighted_sw_sigma(matrix, n_avg=1):
    sigmas = []
    for i in range(n_avg):
        random_graph = random_reference(matrix)
        C = weighted_clustering_coeff_z(matrix)
        C_rand = weighted_clustering_coeff_z(random_graph)
        L = weighted_characteristic_path_length(matrix)
        L_rand = weighted_characteristic_path_length(random_graph)
        sigma = (C/C_rand) / (L/L_rand)
        sigmas.append(sigma)

    return np.mean(sigmas)

#Function: get weighted characteristic path length
def weighted_characteristic_path_length(matrix):
    n_nodes = len(matrix)
    min_distances = weighted_shortest_path(matrix)
    sum_vector = np.empty(n_nodes)
    for i in range(n_nodes):
        sum_vector[i] = (1/(n_nodes-1)) * np.sum([min_distances[i, j] for j in range(n_nodes) if j != i])
    return (1/n_nodes) * np.sum(sum_vector)

#Function: get weighted shortest path
def weighted_shortest_path(matrix):
    matrix = inv_matrix(matrix)
    n_nodes = len(matrix)
    distances = np.empty([n_nodes, n_nodes])
    for i in range(n_nodes):
        for j in range(n_nodes):
            distances[i,j] = matrix[i, j]
    for i in range(n_nodes):
        distances[i,i] = 0

    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if distances[i, j] > distances[i, k] + distances[k, j]:
                    distances[i, j] = distances[i, k] + distances[k, j]

    return distances

#Function: get connection density
def conn_den(matrix):
    net = nx.from_numpy_matrix(matrix)
    den = nx.density(net)
    return den

#Function: get four selected network measures
def curate_net_params(matrix):

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(matrix)
    norm_matrix = normalize(scaled_X, norm='l1', axis=1, copy=True)

    v_node_degree = av_node_degree(matrix)
    clust_coef = clust_coeff(norm_matrix)
    small_world = weighted_sw_sigma(matrix)
    connx_den = conn_den(norm_matrix)

    return [v_node_degree,clust_coef,small_world,connx_den]

# convert datasets to matrices
other_norm_cols = other_norm.columns.values.tolist()
main_cols_nums = other_norm_cols[:-2]

IDs_list = []
group_list = []
corr_matrices = []
matrices_metrics = []

def corr_matrix(data):

    ID = data['ID'][0]
    group = data['group'][0]
    mat = np.zeros((360, 360), dtype=int)

    corrs = data[main_cols_nums]
    corrs_nums = corrs.head(1)

    corrs_np = np.array(corrs_nums)
    shape = (360,360)
    mat_corr = corrs_np.reshape(shape)

    IDs_list.append(ID)
    group_list.append(group)
    corr_matrices.append(mat_corr)

for index, row in other_norm.iterrows():
    nums = row[:-2].to_list()
    group = row[-1:].to_list()[0]
    ID = row[-2:-1].to_list()[0]

    mat = np.zeros((360, 360), dtype=int)
    corrs_np = np.array(nums)
    shape = (360,360)
    mat_corr = corrs_np.reshape(shape)

    IDs_list.append(ID)
    group_list.append(group)
    corr_matrices.append(mat_corr)

# get four network parameters per subject
    index = [x for x in range(0,243)]
for i in index:
    print(IDs_list[i])
    print(group_list[i])
    params = curate_net_params(corr_matrices[i])
    print(params)
    matrices_metrics.append(params)

# store outputs
average_node_deg = []
clust_coeffs = []
small_worlds = []
conn_dens = []

for j in matrices_metrics:
    average_node_deg.append(j[0])
    clust_coeffs.append(j[1])
    small_worlds.append(j[2])
    conn_dens.append(j[3])

# create dataframe for network measures values with IDs and group
metrics_df = pd.DataFrame()
metrics_df['subject_ID'] = IDs_list
metrics_df['group'] = group_list
metrics_df['av_node_degree'] = average_node_deg
metrics_df['connection_density'] = conn_dens
metrics_df['clustering_coeff'] = clust_coeffs
metrics_df['small_worldness'] = small_worlds
