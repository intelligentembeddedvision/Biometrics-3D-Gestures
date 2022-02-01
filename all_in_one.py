import os

if __name__ == "__main__":
    print('Launching acquisition...')
    os.system('python acquisition.py')
    print('Showing point cloud result of hand...')
    os.system('python pointcloud_view.py')
    print('Pre-processing hand...')
    os.system('python preprocessing.py')