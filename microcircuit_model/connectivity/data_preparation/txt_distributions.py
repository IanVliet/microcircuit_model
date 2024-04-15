import scipy.io
import numpy as np
from data_prep_utils import *

mat = scipy.io.loadmat("../../../Convolutive model/261881/Giacopelli_et_al_2020_ModelDB/data.mat")
in_degree_elements, out_degree_elements = mat['dinEE'][0], mat['doutEE'][0]
to_txt_distributions(in_degree_elements, out_degree_elements)
