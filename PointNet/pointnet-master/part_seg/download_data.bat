@echo off

# Download original ShapeNetPart dataset (around 1GB)
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip
Expand-Archive -Path shapenetcore_partanno_v0.zip -DestinationPath ./
rm shapenetcore_partanno_v0.zip

# Download HDF5 for ShapeNet Part segmentation (around 346MB)
wget https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zipp
Expand-Archive -Path shapenet_part_seg_hdf5_data.zip -DestinationPath ./
rm shapenet_part_seg_hdf5_data.zip
