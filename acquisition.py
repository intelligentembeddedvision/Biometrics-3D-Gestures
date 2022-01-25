import cv2
import imageio
import pyrealsense2 as rs
import numpy as np

# import hand extractor for second part
from hand_extractor import hand_extractor, visualizer

# can be changed depending on the resolutions of the D435i
W = 480
H = 270

# take a picture of both depth and color frame
def record_rgbd():
    pipeline = rs.pipeline()

    # initialize camera configuration
    print("[INFO] begin initialization of RS camera")
    config = rs.config()
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)

    pipeline.start(config)

    # this function will be used to colorize the depth frame
    # older dataset use colorized depth frames for preprocessing
    colorizer = rs.colorizer()

    # just wait for the camera to have a real good frame for preprocessing
    for i in range(100):
        # wait for camera to give a signal that frames are available
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_color_frame = colorizer.process(depth_frame)

        if not depth_color_frame:
            raise RuntimeError("[ERROR] Could not acquire depth or color frames.")

    # transform frames to usable arrays for both opencv and the hand extraction program
    depth_image = np.asanyarray(depth_color_frame.get_data())
    pipeline.stop()

    return depth_image


if __name__ == "__main__":
    # main routine
    print("start taking picture")
    depth_image = record_rgbd()
    print("successfully took picture")
    # Hand extractor only takes monochrome (b/w) images, reflect it
    grayImage = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    print("plotting images")
    cv2.imshow('Depth Image', depth_image)
    cv2.imshow('Grayed image for extraction', grayImage)
    cv2.waitKey(0)
    print("beginning preprocessing phase")
    # Run hand extractor on sample
    results = hand_extractor.run_on_sample(grayImage)
    # Visualise hand data
    visualizer.plot(results)