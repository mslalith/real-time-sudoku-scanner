import cv2
import tensorflow as tf
import numpy as np
import time
from sudoku_solver import Sudoku

WARP_IMAGE_WIDTH = 891
WARP_IMAGE_HEIGHT = 891
INPUT_SIZE = 32


#
# convert approx_poly to 1D array.
#
def get_corners(approx_poly):
    edges = []
    for pt in approx_poly:
        for p in pt:
            edges.append(p)
    return get_corners_in_order(edges)


#
# return the points in "Top Left, Top Right, Bottom Left, Bottom Right" order.
#
def get_corners_in_order(points):
    mean = np.mean(points)
    top_left = [mean, mean]
    top_right = [mean, mean]
    bottom_left = [mean, mean]
    bottom_right = [mean, mean]
    for pt in points:
        x = pt[0]
        y = pt[1]
        if x < mean and y < mean:
            if x < top_left[0]:
                top_left[0] = x
            if y < top_left[1]:
                top_left[1] = y

        if x > mean > y:
            if x > top_right[0]:
                top_right[0] = x
            if y < top_right[1]:
                top_right[1] = y

        if x > mean and y > mean:
            if x > bottom_right[0]:
                bottom_right[0] = x
            if y > bottom_right[1]:
                bottom_right[1] = y

        if x < mean < y:
            if x < bottom_left[0]:
                bottom_left[0] = x
            if y > bottom_left[1]:
                bottom_left[1] = y

    return top_left, top_right, bottom_left, bottom_right


#
# checks for a number. If exists, there will be more black pixel count.
#
def is_cell_empty(cel):
    offset = int(cel.shape[0] / 6)
    temp = cel[offset:4 * offset, offset:4 * offset]
    _, temp = cv2.threshold(temp, 125, 255, cv2.THRESH_BINARY)
    return sum(temp.flatten())


#
# predicts number from image.
#
def predict(image):
    image = image.astype('float32') / 255
    prediction = model.predict(image)
    return prediction.argmax()


#
# get the Sudoku cell at r'th row and c'th column.
#
def get_cell(image, r, c):
    x = [c + r * cell_size, r + c * cell_size]
    y = [c + r * cell_size + cell_size, r + c * cell_size + cell_size]
    cel = image[x[0]:y[0], x[1]:y[1]]
    cel = cv2.cvtColor(cel, cv2.COLOR_BGR2GRAY)
    return cv2.resize(cel, (INPUT_SIZE, INPUT_SIZE))


#
# get the (x, y) coordinate to place the text depending on row and column.
#
def get_text_point(r, c):
    size = int(WARP_IMAGE_WIDTH / 9)
    x = size * c + int(size / 3)
    y = size * r + 3 * int(size / 4)
    return x, y


#
# warp the Sudoku square to face straight.
#
def warp_image(image, pts1, width=WARP_IMAGE_WIDTH, height=WARP_IMAGE_WIDTH):
    pts2 = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype='float32')
    perspective = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image.copy(), perspective, (width, height))


#
# warp the Sudoku square to be placed in the original image.
#
def unwarp_image(org_img, img_warp, org_img_corners):
    src_pts = np.array([[0, 0],
                        [img_warp.shape[0], 0],
                        [0, img_warp.shape[1]],
                        [img_warp.shape[0], img_warp.shape[1]]], dtype=np.float32)

    # warp the image to the perspective of original image
    shape = (org_img.shape[1], org_img.shape[0])
    H = cv2.getPerspectiveTransform(src_pts, org_img_corners)
    warp_org = cv2.warpPerspective(img_warp, H, shape)
    return mask_image(org_img, warp_org)


#
# Mask the perspective image with original image.
#
def mask_image(org_img, warp_org):
    grayscale = cv2.cvtColor(warp_org, cv2.COLOR_BGR2GRAY)
    _, grayscale = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY)
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 4)
    grayscale_inv = cv2.bitwise_not(grayscale)
    img_src_final = cv2.bitwise_and(org_img, org_img, mask=grayscale_inv)
    warp_org_final = cv2.bitwise_and(warp_org, warp_org, mask=grayscale)
    final_image = img_src_final + warp_org_final
    return final_image


#
# solve the given 9x9 Sudoku matrix.
#
def solve_sudoku(matrix):
    sudoku_solver = Sudoku(matrix)
    is_solved = sudoku_solver.solve("A", "0", time.time(), False)
    return sudoku_solver.grid if is_solved else None

#
# draw the corners of the Sudoku grid (for debugging).
#
def draw_corners(pts):
    cv2.circle(original, tuple(pts[0]), 2, (0, 0, 255), -1)
    cv2.circle(original, tuple(pts[1]), 2, (0, 255, 0), -1)
    cv2.circle(original, tuple(pts[2]), 2, (255, 0, 0), -1)
    cv2.circle(original, tuple(pts[3]), 2, (0, 255, 255), -1)


#
# entry point of the program.
#
if __name__ == '__main__':

    # load our trained model and initialize the camera.
    model = tf.keras.models.load_model('digit_classifier.h5')
    videoCapture = cv2.VideoCapture(0)

    while True:
        # read the image frame from the camera feed.
        _, frame = videoCapture.read()
        original = frame.copy()

        # convert to grayscale and remove noise to better understand the lines in the image.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        denoise = cv2.fastNlMeansDenoising(gaussian_blur, None, 10, 7, 21)
        threshold = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3)

        # to find our Sudoku grid, find all the contours and select square with biggest area.
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        corners = np.array(get_corners(approx), dtype='float32')
        # draw_corners(corners)

        if len(approx) == 4:
            # warp the skewed image.
            warp = warp_image(original, corners)

            # iterate over each cell and predict the number.
            cell_size = int(WARP_IMAGE_WIDTH / 9)
            sudoku_matrix = []
            for row in range(9):
                sudoku_row = []
                for col in range(9):
                    cell = get_cell(warp, row, col)
                    if is_cell_empty(cell) > 52000:
                        sudoku_row.append(0)
                    else:
                        cell = cell.reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)
                        value = predict(cell)
                        sudoku_row.append(value)
                sudoku_matrix.append(sudoku_row)

            # solve the Sudoku.
            # print(sudoku_matrix)

            sudoku_solved = solve_sudoku(sudoku_matrix)
            print(sudoku_solved)

            # if Sudoku is solved, write the number text on the warped image.
            if sudoku_solved is not None:
                for row in range(9):
                    for col in range(9):
                        if sudoku_matrix[row][col] == 0:
                            center = get_text_point(row, col)
                            value = str(sudoku_solved[row][col])
                            cv2.putText(warp, value, center, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            # warp the image to its original position and display.
            frame = unwarp_image(original, warp, corners)

        cv2.imshow('frame', frame)

        # press 'q' to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the camera.
    videoCapture.release()
    cv2.destroyAllWindows()
