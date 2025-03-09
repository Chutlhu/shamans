import sofar as sfr
import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

import sofar as sfr
import pyroomacoustics as pra



# load my own ATF with sofar
path_to_sofa = Path("./data/SPEAR_Kemar_rir2.sofa")

# The desired reverberation time and dimensions of the room
rt60 = 0.2  # seconds
room_dim = [5, 5, 5]  # meters

energy_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
max_order = 0

room = pra.ShoeBox(
    p=[5, 3, 3],
    materials=pra.Material(energy_absorption),
    fs=16000,
    max_order=max_order,
)

# # add a dummy head receiver from the MIT KEMAR database
orientation = pra.directivities.DirectionVector(azimuth=0, colatitude=90, degrees=True)
spear_array = pra.directivities.MeasuredDirectivityFile(
    path=path_to_sofa,  # SOFA file is in the database
    fs=room.fs,
    interp_order=12,  # interpolation order
    interp_n_points=1000,  # number of points in the interpolation grid
)
# Create a rotation object to orient the microphones.
rot_obj = pra.direction.Rotation3D([0, 0, 0], "xyz", degrees=True)

dir_obj_m1 = spear_array.get_mic_directivity(0, orientation=rot_obj)
dir_obj_m2 = spear_array.get_mic_directivity(1, orientation=rot_obj)
dir_obj_m3 = spear_array.get_mic_directivity(2, orientation=rot_obj)
dir_obj_m4 = spear_array.get_mic_directivity(3, orientation=rot_obj)
dir_obj_m5 = spear_array.get_mic_directivity(4, orientation=rot_obj)

# plot
# azimuth = np.linspace(start=0, stop=360, num=361, endpoint=True)
# colatitude = np.linspace(start=0, stop=180, num=181, endpoint=True)
# # colatitude = None   # for 2D plot
# dir_obj.plot(freq_bin=50)
# plt.show()

# for a head-related transfer function, the microphone should be co-located
mic_pos = [1.05, 1.74, 1.81]
room.add_microphone(mic_pos, directivity=dir_obj_m1)
room.add_microphone(mic_pos, directivity=dir_obj_m2)
room.add_microphone(mic_pos, directivity=dir_obj_m3)
room.add_microphone(mic_pos, directivity=dir_obj_m4)
room.add_microphone(mic_pos, directivity=dir_obj_m5)

# provide the source position with respect to the microphone center
azimuth = -90
colatitude = 90
distance = 1
src_pos = pra.doa.utils.spher2cart(azimuth, colatitude, distance, degrees=True)
src_pos += mic_pos

room.add_source(src_pos)


room.compute_rir()


room.plot_rir()
plt.show()