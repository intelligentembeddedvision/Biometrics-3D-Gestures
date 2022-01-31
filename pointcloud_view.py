# this is a helper tool to visualize RealSense-calculated point clouds
# also permits to verify if depth frames are correct

from open3d.cpu.pybind.io import read_point_cloud
from open3d.cpu.pybind.visualization import draw_geometries


def main():
    # Read the point cloud
    cloud = read_point_cloud("pointcloud.ply")
    # Visualize the point cloud
    draw_geometries([cloud])


if __name__ == "__main__":
    main()
