from train_generate.nn_features import AtmTrainVisualMixin, LightTrainVisualMixin, NN_ModelCompileMixin
from train_generate.data_class import DataClass

################################################################################################################
# DEFINITIVE CLASSES
################################################################################################################

class AtmObtainModel(DataClass, AtmTrainVisualMixin, NN_ModelCompileMixin):
    def __init__(self, ptm, nx = 480, ny = 256, nz = 480, lower_boundary = 180, create_scaler = False, 
                light_type = "Intensity"):
        DataClass.__init__(self, ptm, nx, ny, nz, lower_boundary, create_scaler, light_type)
        AtmTrainVisualMixin.__init__(self)
        
class LightObtainModel(DataClass, LightTrainVisualMixin, NN_ModelCompileMixin):
    def __init__(self, ptm, nx = 480, ny = 256, nz = 480,  lower_boundary = 180, NN_Mer_boundary = 180, create_scaler = False, 
                light_type = "Intensity"):
        DataClass.__init__(self, ptm, nx, ny, nz, lower_boundary, create_scaler, light_type)
        LightTrainVisualMixin.__init__(self)