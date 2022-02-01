import numpy as np
import pyrealsense2 as rs

# according to https://roboticsknowledgebase.com/wiki/sensing/realsense/ this is the optimal depth resolution
W = 848
H = 480

pipeline = rs.pipeline()
# initialize camera configuration
print("[INFO] begin initialization of RS camera")
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# launch camera
profile = pipeline.start(config)

# init point cloud
pc = rs.pointcloud()
points = rs.points()

# Set high accuracy for depth sensor
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(
    rs.option.visual_preset, 3
)

align_to = rs.stream.color
align = rs.align(align_to)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# colorize depth frame
colorizer = rs.colorizer()

print("[INFO] launching RS camera, taking pictures")
frameset = []
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
# threshold = rs.threshold_filter()
# threshold.set_option(rs.option.min_distance, 0.1)
# threshold.set_option(rs.option.max_distance, 0.5)
# take a series of frames for better filtering
for i in range(100):
    frames = pipeline.wait_for_frames()
    frameset.append(frames.get_depth_frame())
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    # wait for camera to give a signal that frames are available
    # apply a series of filters for better depth image quality
    decimation = rs.decimation_filter()
    hole_filling = rs.hole_filling_filter()
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    frame = frameset[i]
    frame = decimation.process(frame)
    frame = depth_to_disparity.process(frame)
    frame = spatial.process(frame)
    frame = temporal.process(frame)
    frame = disparity_to_depth.process(frame)
    frame = hole_filling.process(frame)
    # frame = threshold.process(frame)

if not frame or not color_frame:
    raise RuntimeError("[ERROR] Could not acquire depth or color frame.")

print("[INFO] done dealing with pictures, closing camera...")
# close RS camera
pipeline.stop()

# Create point cloud to save_to_ply object
ply = rs.save_to_ply("pointcloud.ply")

# Set options to the desired values
ply.set_option(rs.save_to_ply.option_ply_binary, False)
ply.set_option(rs.save_to_ply.option_ply_normals, True)

print("Saving point cloud...")
# Apply the processing block to the frameset which contains the depth frame and the texture
ply.process(frame)

print("Saving data...")
# Correct depth frame by **actually** giving depths distances
depth_image = np.asanyarray(frame.get_data())
scaled_depth_image = depth_image * depth_scale
# save depths in numpy file, to be used in a file.
np.save("depth.npy", np.array(scaled_depth_image))
