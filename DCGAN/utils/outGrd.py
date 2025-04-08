import numpy as np
import os


def outGrd(filepath, grdata, gdmin, gdmax):
    """
    Save data as Surfer 6 Text format GRD file
    
    Parameters:
    -----------
    filepath : str
        Path to save the GRD file
    grdata : numpy.ndarray
        Grid data to be saved
    gdmin : float
        Minimum value in the grid data
    gdmax : float
        Maximum value in the grid data
    """
    # print(grdata.shape, type(grdata), Nx,Ny)
    # print(filepath)
    # os.system('pause')
    
    # Alternative method to save as Surfer 6 Text format GRD file
    # with open(filepath, 'w') as f:
    #     f.write('DSAA\n')
    #     f.write('{0} {1}\n'.format(Nx, Ny))
    #     f.write('{0} {1}\n'.format(Xmin, Xmax))
    #     f.write('{0} {1}\n'.format(Ymin, Ymax))
    #     f.write('{0} {1}\n'.format(gdmin, gdmax))
    #
    # f.close()
    
    # Create header with fixed grid size (100x100) and range (0-400 for both X and Y)
    header = "DSAA\n 100 100\n 0.0 400.0\n 0.0 400.0\n " + str(gdmin) + ' ' + str(gdmax)
    # print(header)
    # os.system('pause')

    # Save the grid data with the header
    np.savetxt(filepath, grdata, header=header, comments='', fmt='%8.3f')

    return
