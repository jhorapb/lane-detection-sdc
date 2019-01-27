
import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def canny(image):
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(grayscale_img, (5, 5), 0)
    return cv2.Canny(blur_img, 50, 150)


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intercept = params[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

# Array of pixels
# img = cv2.imread('img/lanes5.jpg')
# lane_img = np.copy(img)
# canny_img = canny(lane_img)
# cropped_img = region_of_interest(canny_img)
#
# # (img, resolution, degree presition, threshold, placeholder array, length of line, max distance in px between
# # segmented lines wich will allow to be connected into single line (not breaking up))
# lines = cv2.HoughLinesP(cropped_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#
# average_lines = average_slope_intercept(lane_img, lines)
# line_img = display_lines(lane_img, average_lines)
# combo_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)
# cv2.imshow('showpic', combo_img)
# cv2.waitKey(0)

# Optional plot:
#   plt.imshow(cropped_img)
#   plt.show()

cap = cv2.VideoCapture('img/lanes_video.mp4')
while (cap.isOpened()):
    _, frame = cap.read()
    canny_img = canny(frame)
    cropped_img = region_of_interest(canny_img)
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    average_lines = average_slope_intercept(frame, lines)
    line_img = display_lines(frame, average_lines)
    combo_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    cv2.imshow('showpic', combo_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()