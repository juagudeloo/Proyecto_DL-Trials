from train_generate.nn_features import AtmTrainVisualMixin, LightTrainVisualMixin, NN_ModelCompileMixin
from train_generate.data_class import DataClass
from boundaries import low_boundary, top_boundary

################################################################################################################
# DEFINITIVE CLASSES
################################################################################################################

class AtmObtainModel(DataClass, AtmTrainVisualMixin, NN_ModelCompileMixin):
    def __init__(self, ptm, opt_len, nx = 480, ny = 256, nz = 480, low_boundary = low_boundary(), top_boundary = top_boundary(), create_scaler = False, 
                light_type = "Intensity"):
        self.opt_len = opt_len
        DataClass.__init__(self, ptm, nx, ny, nz, low_boundary, top_boundary, create_scaler, light_type)
        AtmTrainVisualMixin.__init__(self)
        
class LightObtainModel(DataClass, LightTrainVisualMixin, NN_ModelCompileMixin):
    def __init__(self, ptm, opt_len, nx = 480, ny = 256, nz = 480,  low_boundary = low_boundary(), top_boundary = top_boundary(), create_scaler = False, 
                light_type = "Intensity"):
        self.opt_len = opt_len
        DataClass.__init__(self, ptm, nx, ny, nz, low_boundary, top_boundary, create_scaler, light_type)
        LightTrainVisualMixin.__init__(self)