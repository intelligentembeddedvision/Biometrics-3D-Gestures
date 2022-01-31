# this is a helper tool made to export point cloud data
# point cloud calculation is made via the RealSense library.

import pyrealsense2 as rs

# init realsense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# init point cloud
pc = rs.pointcloud()
points = rs.points()

# colorize depth (and point cloud in the same instance)
colorizer = rs.colorizer()

# Start streaming
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
count = 0

try:
    # wait for good frames to come in
    while count < 2000:
        count += 1

    # Wait for the next set of frames from the camera
    frames = pipeline.wait_for_frames()
    colorized = colorizer.process(frames)

    # Create save_to_ply object
    ply = rs.save_to_ply("pointcloud.ply")

    # Set options to the desired values
    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    ply.set_option(rs.save_to_ply.option_ply_normals, True)

    print("Saving point cloud...")
    # Apply the processing block to the frameset which contains the depth frame and the texture
    ply.process(colorized)
    print("Done")

finally:
    pipeline.stop()
