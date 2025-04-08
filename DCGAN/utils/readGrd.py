## read grd file
# @Time: 2023/2/17 10:59
# @Author: WangLin
# @File: ReadGrd.py: Reading grd files
# @Software: PyCharm

import os
import numpy as np


# Function to read grd files, returns grid file array grdata(Ny,Nx), Ny is the number of columns - corresponding to dim=0
def readGrdbynp(filepath):
    """
    Read Surfer 6 Text format GRD file and return the grid data as numpy array
    
    Parameters:
    -----------
    filepath : str
        Path to the GRD file
        
    Returns:
    --------
    gd : numpy.ndarray
        Grid data with shape (Ny, Nx)
    """
    with open(filepath, "r", encoding="UTF-8") as infile:
        infile.readline()  # Skip the first line of the standard surfer 6 text format file with the label DSAA
        str = infile.readline().split()  # Read the second line with point and line numbers
        Nx = int(str[0])
        Ny = int(str[1])
        # print(Nx, Ny)
        str = infile.readline().split()  # Read the third line with X direction minimum and maximum values
        Xmin = float(str[0])
        Xmax = float(str[1])
        # print(Xmin, Xmax)
        str = infile.readline().split()  # Read the fourth line with Y direction minimum and maximum values
        Ymin = float(str[0])
        Ymax = float(str[1])
        str = infile.readline().split()  # Read the fifth line with grid data minimum and maximum values
        gdmin = float(str[0])
        gdmax = float(str[1])
        # print(gdmin, gdmax)
    infile.close()
    gd = np.loadtxt(filepath, skiprows=5)
    # print(gd.shape, gd.min(), gd.max())
    # os.system('pause')
    return gd


# Local subroutine test
if __name__ == '__main__':
    Fgafile = r"D:\Project\DL_interface_inversion\origin_data\dg\FwGravofBasin_0001_Dens_01.grd"
    grddata = readGrdbynp(Fgafile)
    print(grddata.shape)
    print(grddata.min())
