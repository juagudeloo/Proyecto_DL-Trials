import numpy as np

self_filename = []
file_interval = np.arange(54000, 223000, 10000)
for i in range(len(file_interval)):
        self_filename.append(str(file_interval[i]))
print(self_filename)
