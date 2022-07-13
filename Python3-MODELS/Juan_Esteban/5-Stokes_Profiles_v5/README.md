# VERSION 5.4 

Compared to the version 5.3, this version is not reshaping the input data to be a cube, but 
in this version it will remain ravel for the x and z dimensions, to use less memory and have
the data already dispossed to be splitted in training set and testing set.

# VERSION 5.5

Compared to the version 5.4, this version is not using the Sequential API to create the nn model,
but here we use the functional API along with the tf.keras.models.Model object to create the nn model.
The reason for this is that when we create the model with the Sequential API in a function, the class
we have created does not recognize the precious creation of the model.

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

-   Using the classes allows us to charge data in different times so that we can free memory before 
    using a new set of data. For example, we can just upload the training data and train the model. 
    once that's done, we call the function of prediction in which we use the same trained model but 
    at that moment we will have freed the memory from the training data, giving space for the predic-
    tion data.