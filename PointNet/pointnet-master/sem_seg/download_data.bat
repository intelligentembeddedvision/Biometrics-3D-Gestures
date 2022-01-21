@echo off

wget https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip
Expand-Archive -Path indoor3d_sem_seg_hdf5_data.zip -DestinationPath ./
rm indoor3d_sem_seg_hdf5_data.zip

