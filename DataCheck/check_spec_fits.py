from astropy.io import fits

def main():
    
    # Open the FITS file
    hdulist = fits.open("../../Real_observation_data/nb_6302_2020-04-24T11:26:22_scans=0-151,155-184,186-227,229-239_stokes_corrected_export2021-06-08T08:24:35_im.fits")
    hdulist.info()

    # Print the header of the primary HDU
    print(repr(hdulist[0].header))

    # Print the data in the primary HDU
    print(hdulist[0].data)

    # Close the file
    hdulist.close()



if __name__ == "__main__":
    main()