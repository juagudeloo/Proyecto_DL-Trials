from nn_features import AtmTrainVisualMixin, LightTrainVisualMixin
from data_class import DataClass

################################################################################################################
# DEFINITIVE CLASSES
################################################################################################################

class AtmObtainModel(DataClass, AtmTrainVisualMixin):
    def __init__(self, nx = 480, ny = 256, nz = 480, lower_boundary = 180, create_scaler = False, 
                light_type = "Intensity"):
        DataClass.__init__(nx, ny, nz, lower_boundary, create_scaler)
        AtmTrainVisualMixin.__init__(light_type)
        
class LightObtainModel(DataClass, LightTrainVisualMixin):
    def __init__(self, nx = 480, ny = 256, nz = 480, lower_boundary = 180, create_scaler = False, 
                light_type = "Intensity"):
        DataClass.__init__(nx, ny, nz, lower_boundary, create_scaler)
        LightTrainVisualMixin.__init__(light_type)