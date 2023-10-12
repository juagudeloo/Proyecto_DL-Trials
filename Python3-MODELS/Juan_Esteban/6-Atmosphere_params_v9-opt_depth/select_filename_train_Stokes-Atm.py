#!/usr/bin/env python
import subprocess
path = "/girg/juagudeloo/Proyecto_DL-Trials/Python3-MODELS/Juan_Esteban/6-Atmosphere_params_v9-opt_depth"
subprocess.call(path+'Train_Stokes-Atm_model.py 080000', shell = True)
