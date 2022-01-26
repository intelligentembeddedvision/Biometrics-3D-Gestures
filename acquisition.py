import cv2
import pyrealsense2 as rs
import numpy as np

# import hand extractor for second part
from hand_extractor import hand_extractor, visualizer

# reimport those from the hand extractor
SHOW_IMAGES = True
CLI_VERBOSE = True

# can be changed depending on the resolutions of the D435i
W = 640
H = 480

pipeline = rs.pipeline()
# initialize camera configuration
print("[INFO] begin initialization of RS camera")
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Align depth frame to color frame
align_to = rs.stream.color
align = rs.align(align_to)


# take a picture (depth and color)
def record_rgbd():
    # this function will be used to colorize the depth frame
    # older dataset use colorized depth frames for preprocessing
    # it will also permit to have a b&w picture for hand extraction
    colorizer = rs.colorizer()

    # just wait for the camera to have a real good frame for preprocessing
    for i in range(100):
        # wait for camera to give a signal that frames are available
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_color_frame = colorizer.process(aligned_depth_frame)

    # transform frames to usable arrays for both opencv and the hand extraction program
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 153
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.07), cv2.COLORMAP_INFERNO)

    if not depth_color_frame or not color_frame:
        raise RuntimeError("[ERROR] Could not acquire depth or color frame.")

    pipeline.stop()

    return bg_removed, depth_colormap


if __name__ == "__main__":
    # main routine
    print("start taking picture")
    color_image, depth_image = record_rgbd()
    print("successfully took picture")
    print("plotting images")
    cv2.imshow('Depth Image', depth_image)
    cv2.imshow('Colored image - Background removed', color_image)
    cv2.waitKey(0)
    print("beginning preprocessing phase")
    # Run hand extractor on sample
    out_hand, out_img, success = hand_extractor.run_on_sample(depth_image)
    visualizer.imshow(out_img, False, False)
