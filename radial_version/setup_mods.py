import numpy as np
import ast
from galario.double import get_image_size

def read_param_file(file):

    params={}
    with open(file, 'r') as f:
        for line in f:
            line = line.split("#", 1)[0].strip() #This will skip in-line comments
            if "=" in line:
                pkey, pvalue = line.split("=", maxsplit=1)
                pkey = pkey.strip()
                pvalue = ast.literal_eval(pvalue.strip())
                if isinstance(pvalue, list):
                    pvalue = np.array(pvalue, dtype=float)
                params[pkey] = pvalue
    return params

def initialize_data(datatable, frequency):

    u, v, re, im, w = np.require(np.loadtxt(datatable, unpack=True), requirements='C')
    wavelength = 299792458/frequency
    u /= wavelength
    v /= wavelength

    nx, dx = get_image_size(u, v)
    return  nx, dx, u, v, re, im, w