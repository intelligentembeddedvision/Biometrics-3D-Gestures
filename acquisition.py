import imageio
import pyrealsense2 as rs
import numpy as np

# import hand extractor for second part
from hand_extractor import hand_extractor, visualizer

# can be changed depending on the resolutions of the D431i
W = 848
H = 480

# take picture of both depth and color frame
def record_rgbd():
    pipeline = rs.pipeline()

    # initialize camera configuration
    print("[INFO] begin initialization of RS camera")
    config = rs.config()
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, W, H, rs.format.rgb8, 30)

    pipeline.start(config)

    # align depth frame to color frame
    align_to = rs.stream.color
    align = rs.align(align_to)

    # this function will be used to colorize the depth frame
    # older dataset use colorized depth frames for preprocessing
    colorizer = rs.colorizer()

    try:
        # just wait for the camera to have a real good frame for preprocessing
        for i in range(100):
            print(i)
            # wait for camera to give a signal that frames are available
            frames = pipeline.wait_for_frames()
            # align depth to color
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            # colorize depth frame
            depth_color_frame = colorizer.colorize(aligned_depth_frame)
            color_frame = aligned_frames.get_color_frame()

            if not depth_color_frame or not color_frame:
                raise RuntimeError("[ERROR] Could not acquire depth or color frames.")

            # transform frames to usable arrays for both opencv and the hand extraction program
            depth_image = np.asanyarray(depth_color_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            imageio.imsave('color.jpg', color_image)
            imageio.imsave('depth.jpg', depth_image)

    finally:
        pipeline.stop()

    return color_image, depth_image

if __name__ == "__main__":
    # main routine
    print("start taking picture")
    color_image, depth_image = record_rgbd()
    print("successfully took picture, launching hand extractor")
    # Run hand extractor on sample
    scaled_image = hand_extractor.run_on_sample(depth_image)
    # Visualise hand data
    visualizer.plot(scaled_image)