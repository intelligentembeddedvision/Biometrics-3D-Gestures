import cv2
import pyrealsense2 as rs
import numpy as np

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

depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(
    rs.option.visual_preset, 3
)  # Set high accuracy for depth sensor

# align depth stream to color stream
align_to = rs.stream.color
align = rs.align(align_to)

# colorize depth frame
colorizer = rs.colorizer(2)

print("[INFO] launching RS camera, taking pictures")
# take a picture (depth and color)
frameset = []
# just wait for the camera to have a real good frame for preprocessing
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

    if not frame or not color_frame:
        raise RuntimeError("[ERROR] Could not acquire depth or color frame.")

print("[INFO] done dealing with pictures, closing camera and exporting pictures")
# close RS camera
pipeline.stop()
# transform frames to usable arrays for both opencv and the hand extraction program
# the colorizer.colorize function will be used to colorize the depth frame
depth_image = np.asanyarray(colorizer.colorize(frame).get_data())
color_image = np.asanyarray(color_frame.get_data())

# save images
cv2.imwrite('color.jpg', color_image)
cv2.imwrite('depth.jpg', depth_image)
