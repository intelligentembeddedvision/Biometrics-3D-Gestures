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

frameset = []
try:
    for i in range(100):
        # wait for camera to give a signal that frames are available
        # apply a series of filters for better depth image quality
        frames = pipeline.wait_for_frames()
        frameset.append(frames.get_depth_frame())
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        hole_filling = rs.hole_filling_filter()
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        frame = frameset[i]
        frame = spatial.process(frame)
        frame = temporal.process(frame)
        frame = hole_filling.process(frame)

    # Create save_to_ply object
    ply = rs.save_to_ply("pointcloud.ply")

    # Set options to the desired values
    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    ply.set_option(rs.save_to_ply.option_ply_normals, True)

    print("Saving point cloud...")
    # Apply the processing block to the frameset which contains the depth frame and the texture
    ply.process(frame)
    print("Done")

finally:
    pipeline.stop()
