import numpy as numpy
import pandas as pd 

from matplotlib import pyplot as plt 

import nrrd

import os

base_dir = "/cluster/projects/radiomics/Temp/sejin/LIDC_IDRI_save"


for j in range(10):
    i = j + 100
    img, _ = nrrd.read(os.path.join(base_dir, "images/LIDC-IDRI-0{}.nrrd").format(i))
    mask, _ = nrrd.read(os.path.join(base_dir, "masks/LIDC-IDRI-0{}.nrrd").format(i))

    print(img.shape)
    print(mask.shape)
