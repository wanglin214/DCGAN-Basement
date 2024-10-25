import numpy as np
import os


def outGrd(filepath, grdata, gdmin, gdmax):
    # print(grdata.shape, type(grdata), Nx,Ny)
    # print(filepath)
    # os.system('pause')
    # 将数据保存为Surfer 6 Text格式的GRD文件
    # with open(filepath, 'w') as f:
    #     f.write('DSAA\n')
    #     f.write('{0} {1}\n'.format(Nx, Ny))
    #     f.write('{0} {1}\n'.format(Xmin, Xmax))
    #     f.write('{0} {1}\n'.format(Ymin, Ymax))
    #     f.write('{0} {1}\n'.format(gdmin, gdmax))
    #
    # f.close()
    header = "DSAA\n 100 100\n 0.0 400.0\n 0.0 400.0\n " + str(gdmin) + ' ' + str(gdmax)
    # print(header)
    # os.system('pause')

    np.savetxt(filepath, grdata, header=header, comments='', fmt='%8.3f')

    return
