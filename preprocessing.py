import cv2

from hand_extractor import hand_extractor, visualizer

print("beginning preprocessing phase")
img = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)
# Invert colors (old dataset used to have black hand gesture on white background)
img_inverted = cv2.bitwise_not(img)
img2 = cv2.imread('color.jpg')

print("plotting image before hand extraction")
cv2.imshow("Depth", img)
cv2.waitKey(0)

print("hand extraction begins...")
out_hand, out_img, success = hand_extractor.run_on_sample(img, img2)
visualizer.imshow(out_img, False, False)