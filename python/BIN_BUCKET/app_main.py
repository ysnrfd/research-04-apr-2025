from engine import *
from engine_extra import *
import numpy as np

nnr = nn()
nnc = extra_engine()
mmc1 = nnr.nnp(2,2,2,2)
mmc2 = nnc.ext_engine(3,3,3)
mult_mmc = mmc1*mmc2
#---------------------------------
sigma_1 = nnr.nnp(2,2,2,2)
sigma_2 = nnc.ext_engine(2,2,2)
sigma_nc = sigma_1**sigma_2

#---------------------------------
#sigma_opt = sigma_1/sigma_2*sigma_nc**sigma_1+sigma_1+sigma_2*np.pi
#sigma_opt_2 = np.array([sigma_opt])*np.pi/sigma_opt
#show = nnr.nnp(2,2,2,2)
#print('dim:', show)
#print('mult mmc:', mult_mmc, 'mmc1:', mmc1, 'mmc2:', mmc2)
#---------------------------------

print('sigma nc:', sigma_nc)


