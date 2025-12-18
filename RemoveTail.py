import numpy as np
#
from OCTpy import Oct3Dimage
import matplotlib.pyplot  as plt
from concurrent.futures import ThreadPoolExecutor

def tail_reduction(img_flow, BitDepth=16, amp=0.07, dec=0.012):
    """
    :param img_stru: structural volume
    :param img_flow: flow volume
    :param BitDepth: 16 for lab built system dicom file
    :return:
    """
    # normalize the volume to 0-1
    bitrange = 2 ** BitDepth
    IMSf = img_flow/bitrange

    # for 3d image
    if img_flow.ndim==3:
        nZ, nX, nY= IMSf.shape

        vz = np.arange(0, nZ)

        ffilter = amp * np.exp(-dec * vz).reshape(-1, 1)

        # IMSf = IMS
        print('Calculating the tail free volume')

        # filter the image
        for iY in range(0, nY):
            print('frame = ', iY)
            for iZ in range(0, nZ-1):
                tmp = IMSf[iZ + 1::, :, iY] - IMSf[iZ, :, iY] * ffilter[:nZ - iZ - 1, :]
                IMSf[iZ + 1::, :, iY] = tmp

    # for 2d slice
    elif img_flow.ndim==2:
        nZ, nX = IMSf.shape

        vz = np.arange(0, nZ)
        ffilter = amp * np.exp(-dec * vz).reshape(-1, 1)

        print('Calculating the tail free image')

        # filter the image
        for iZ in range(0, nZ - 1):
            tmp = IMSf[iZ + 1::, :] - IMSf[iZ, :] * ffilter[:nZ - iZ - 1]
            IMSf[iZ + 1::, :] = tmp

    IMSf[IMSf < 0] = 0

    # change the dynamic range for 16bit dicom output
    Rg = np.percentile(IMSf[1::20], [5, 99.99])
    IMSf[IMSf < Rg[0]] = Rg[0]
    IMSf[IMSf > Rg[1]] = Rg[1]
    IMSf = np.uint16((IMSf - Rg[0]) / (Rg[1] - Rg[0])*65535) # to 16bit

    return IMSf

def tail_reduction_parallel(img_flow, BitDepth=16, amp=0.07, dec=0.012, rg_L =5, rg_H=99.99):
    """
    :param img_stru: structural volume
    :param img_flow: flow volume
    :param BitDepth: 16 for lab built system dicom file
    :return:
    """
    # normalize the volume to 0-1
    bitrange = 2 ** BitDepth
    IMSf = img_flow/bitrange

    # for 3d image
    if img_flow.ndim==3:
        nZ, nX, nY= IMSf.shape

        vz = np.arange(0, nZ)

        ffilter = amp * np.exp(-dec * vz).reshape(-1, 1)

        # IMSf = IMS
        print('Calculating the tail free volume')

        # filter the image
        def process_frame(iY):
            print('frame = ', iY)
            for iZ in range(0, nZ-1):
                tmp = IMSf[iZ + 1::, :, iY] - IMSf[iZ, :, iY] * ffilter[:nZ - iZ - 1, :]
                IMSf[iZ + 1::, :, iY] = tmp

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            executor.map(process_frame, range(nY))

    # for 2d slice
    elif img_flow.ndim==2:
        nZ, nX = IMSf.shape

        vz = np.arange(0, nZ)
        ffilter = amp * np.exp(-dec * vz).reshape(-1, 1)

        print('Calculating the tail free image')

        # filter the image
        for iZ in range(0, nZ - 1):
            tmp = IMSf[iZ + 1::, :] - IMSf[iZ, :] * ffilter[:nZ - iZ - 1]
            IMSf[iZ + 1::, :] = tmp

    IMSf[IMSf < 0] = 0

    # change the dynamic range for 16bit dicom output
    Rg = np.percentile(IMSf[1::5], [rg_L, rg_H])
    IMSf[IMSf < Rg[0]] = Rg[0]
    IMSf[IMSf > Rg[1]] = Rg[1]
    IMSf = np.uint16((IMSf - Rg[0]) / (Rg[1] - Rg[0])*65535) # to 16bit

    return IMSf

# testing
"""
folder_path = 'D:/oof3response/E236'
stru_file = '/P0015-1.7.1.31492-200kHzComp-646_Angio (12mmx12mm)_10-3-2018_11-15-30_OD_sn0000_cube_z.img'
flow_file = '/P0015-1.7.1.31492-200kHzComp-646_Angio (12mmx12mm)_10-3-2018_11-15-30_OD_sn0000_FlowCube_z.img'

img_obj = Oct3Dimage()    # create the instance
img_obj.read_stru_data(folder_path + stru_file)
img_obj.read_flow_data(folder_path + flow_file)    # read the flow data

img_stru = img_obj.stru3d[:,:,0:10]
img_flow = img_obj.flow3d[:,:,0:10]

import cProfile
import pstats

# Your existing code here

# Wrap your function call in cProfile
# cProfile.run('result = tail_reduction(img_stru, img_flow, BitDepth=16, amp=0.07, dec=0.012)', 'profile_stats')
# cProfile.run('result = tail_reduction_new(img_flow, BitDepth=16, amp=0.07, dec=0.012)', 'profile_stats')
cProfile.run('result = tail_reduction_parallel(img_flow, BitDepth=16, amp=0.07, dec=0.012)', 'profile_stats')

# Create a pstats.Stats object
stats = pstats.Stats('profile_stats')

# Print the statistics
stats.strip_dirs().sort_stats('cumulative').print_stats()

img_obj.save_video(result, 'D:/oof3response/E236/tail_free_ori_parallel.dcm')

plt.imshow(result[:,:,3])
plt.show()
# vdata = permute(uint8(IMSf * 255), [1, 2, 4, 3])


plt.imshow(one_image)
plt.show()
#change the contrast

# plt.imshow(IMSf[:,:,0])
# plt.show()
# plt.imshow(img_flow[:,:,0])
# plt.show()
"""