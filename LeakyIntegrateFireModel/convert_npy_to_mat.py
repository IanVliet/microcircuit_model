import numpy as np
import scipy.io

with open('manc_v1.0_indegrees.npy', 'rb') as indegree_file:
    in_degree_elements = np.load(indegree_file)

with open('manc_v1.0_outdegrees.npy', 'rb') as outdegree_file:
    out_degree_elements = np.load(outdegree_file)

scipy.io.savemat('data_manc.mat', dict(din=in_degree_elements.astype(float), dout=out_degree_elements.astype(float)))
