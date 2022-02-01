import numpy as np

from hand_extractor import hand_extractor, visualizer

print("beginning preprocessing phase")
image = np.load('depth.npy')

print("hand extraction begins...")
out_hand, out_img, success = hand_extractor.run_on_sample(image)
visualizer.plot3D(out_hand)