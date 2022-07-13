# VERSION 5.4 

Compared to the version 5.3, this version is not reshaping the input data to be a cube, but 
in this version it will remain ravel for the x and z dimensions, to use less memory and have
the data already dispossed to be splitted in training set and testing set.

#### NOTES:
-   self.mvyy must be divided by self.mrho to obtain the actual velocity, because originally
    self.mvyy is the momentum (not with mass but with density). Due to this, division by zero 
    is possible to be encountered. Therefore, whenever there exists a zero value in any column
    column of mrho, that column is going to be setted to a column full of ones. This is done so 
    so that the division self.mvyy/self.mrho can be effectuated and then also this columns are 
    going to be ignored for all the input and output data correspondingly.

-   The charging of the stokes parameters must be done with the indices in the inverse order they
    are specified in the original function, i.e., iy,ix contray to the original disposition which
    is ix,iy, so that the corresponding column values to the inputs are obtain.