import unittest
import os
import numpy as np
from train_generate.extract_compile.data_class import scaling

class TestScalingMethod(unittest.TestCase):
    def test_creating_scaler(self):
        scaler_names = ["test_a", "test_b"]
        create_scaler = True
        arrays = []
        count = 0
        for i in range(len(scaler_names)):
            np.random.seed(100+i)
            arrays.append(np.random.rand(10*(i+1),10))
            scaling(arrays[i], scaler_names[i], create_scaler)
            count += os.listdir().count(scaler_names[i]+".npy")
        self.assertEqual(count, 2)
        for i in range(len(scaler_names)):
            os.remove(scaler_names[i]+".npy")

        
if __name__ == "__main__":
    unittest.main()