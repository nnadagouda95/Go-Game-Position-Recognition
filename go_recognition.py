import cv2
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
from itertools import combinations
import scipy.io as sio

##############################################################################################
""" Functions """
##############################################################################################
#Generate the slope and intercept from rho and theta
def generate_slope_intercept(rho, theta, pred):
    intercept = np.zeros(pred)
    slope = np.zeros(pred)
    # TODO : setup the case where the slope can be infinity
    for i in range(0, pred):
        a = math.cos(theta[i])
        b = math.sin(theta[i])
        x0 = a * rho[i]
        y0 = b * rho[i]
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

        # If slope is infinity, assign a large value as a placeholder to return
        if pt2[0] == pt1[0]:
            slope[i] = 1000000
            # If line is parallel to y-axis, then set intercept as the x-coordinate
            intercept[i] = pt2[0] 
        else:
            slope[i] = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
            intercept[i] = pt1[1] - (slope[i]) * pt1[0]

    return slope, intercept


# function called for skew correction
def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


# This is also called from skew_correction()
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct a "birds eye view",
    #i.e. top-down view) of the image, specifying pointsin the top-left, top-right,
    # bottom-right, and bottom-left using the speciifed order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

####################################### Sharpening of the image ########################################################

def sharpening(img, num_contrast, contrast_method):
    tile_size = 10  # Number of pixels in the window for performing CLAHE
    for i in range(0, num_contrast):
        if contrast_method == 1:
            # Converting image to LAB Color model
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            # cv2.imshow("lab",lab)

            # Splitting the LAB image to different channels
            l, a, b = cv2.split(lab)
            # cv2.imshow('l_channel', l)
            # cv2.imshow('a_channel', a)
            # cv2.imshow('b_channel', b)

            # Applying CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_size, tile_size))
            cl = clahe.apply(l)
            # cv2.imshow('CLAHE output', cl)

            # Merge the CLAHE enhanced L-channel with the a and b channel
            limg = cv2.merge((cl, a, b))
            # cv2.imshow('limg', limg)

            # Converting image from LAB Color model to RGB model
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            img = final
        # cv2.imshow('final', final)

        elif contrast_method == 0:
            img = img[:, :, 0]
            #specifications of CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_size, tile_size))
            cl = clahe.apply(img)
            final = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
            img = final
        else:
            continue
        cv2.imwrite('gen/sharpened.jpg', img)
    return img


##################################### Threshold based Segmentation #######################################################
def threshold_segmentation(img):
    # window size to get the rgb intensity averages averages
    window = 20
    #image shape specified in channels l and w 
    l = img.shape[0]
    w = img.shape[1]

    # Area of the image we definitely know is the background
    def_bg = img[0:window, 0:window]
    b_b = int(np.mean(def_bg[:, :, 0]))
    g_b = int(np.mean(def_bg[:, :, 1]))
    r_b = int(np.mean(def_bg[:, :, 2]))
    # print(b_b, g_b, r_b, def_bg.shape)


    # Area of the image we definitely know is the foreground (board)
    def_fg = img[l // 2 - window:l // 2 + window, w // 2 - window:w // 2 + window]
    b_f = int(np.mean(def_fg[:, :, 0]))
    g_f = int(np.mean(def_fg[:, :, 1]))
    r_f = int(np.mean(def_fg[:, :, 2]))
    # print(b_f, g_f, r_f, def_fg.shape)

    # Tolerance while comparing the intensity values
    tol = 5
    mask = np.zeros(img.shape[:2], np.uint8)
    for i in range(0, l):
        for j in range(0, w):
            x = img[i, j, :]
            # if x[0] < b_f + tol and x[0] < g_f + tol and x[0] < r_f + tol:
            if x[0] > b_f + tol and x[0] > g_f + tol and x[0] > r_f + tol:
                mask[i, j] = 1
            else:
                mask[i, j] = 0

    # Viewing and saving the mask
    mask = mask * 255
    # plt.imshow(mask), plt.show()

    # Converting to BGR format and saving the image
    image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('gen/thresholded_mask.jpg', image)

    # Performing erosion to fill in the gaps in the thresholded mask
    kernel = np.ones((5, 5), np.uint8)

    #erosion done usigng dilation of the image
    erosion = cv2.dilate(image, kernel, iterations=5)
    print (erosion.shape)
    cv2.imwrite('gen/eroded.jpg', erosion)
    return image, erosion
    # plt.imshow(erosion), plt.show()


###################################### GrabCuts based segmentation #####################################################
def grab_cuts_segmentation(img):
    # All probable background pixels are multiplyied with cv2.GC_PR_BGD
    mask = np.ones(img.shape[:2], np.uint8) * cv2.GC_PR_BGD

    # Background and foreground models for grabcut segmentation
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # corner(x,y)followed by x and y lengths for the rectangle
    # Everything outside the rectangle is taken as the background
    corner = 20
    rect = (corner, corner, img.shape[0] - corner, img.shape[1] - corner)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

   #applying foreground mask 
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask3 = mask2 * 255
    cv2.imwrite('gen/foreground_only_mask.png', mask3)
    img = img * mask2[:, :, np.newaxis]
    cv2.imwrite('gen/foreground_img.jpg', img)
    # plt.imshow(mask3), plt.colorbar(), plt.show()
    mask3 = cv2.cvtColor(mask3, cv2.COLOR_GRAY2BGR)
    return img, mask3

############################################### Hough Transform ####################################################
# To obtain the rho and theta values for the lines present on the edges of the quadrilateral
def hough_transform(img):
    src = img
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Median Blur
    src = cv2.medianBlur(src, 5)

    # Copy edges to the images that will display the results in BGR
    dst = cv2.Canny(src, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    # Hough lines applied on the images 
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 100, None, 0, 0)
    rho = np.array(np.zeros(len(lines)))
    theta = np.array(np.zeros(len(lines)))
    if lines is not None:
        for i in range(0, len(lines)):
            # for i in range(8, 9):
            rho[i] = lines[i][0][0]
            theta[i] = lines[i][0][1]
            a = math.cos(theta[i])
            b = math.sin(theta[i])
            x0 = a * rho[i]
            y0 = b * rho[i]
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        # print('rho - ', rho, '\ntheta - ', theta * (180 / np.pi), len(lines))
        cv2.imwrite('gen/hough_transform.jpg', cdst)
    else:
        # If we are unable to fit any lines, then exit the program
        print('No lines detected')
        exit(0)

    return rho, theta


# Obtain the final four coordinates from all the lines detected by Hough transform
def pairwise_intersect(slope, intercept, filename):
    # We need to intersect all lines pairwise.

    # Intersecting lines with infinite slope
    L1 = list(range(0, slope.size))

    # number of lines
    n = slope.size
    # Lines taken two at a time
    r = 2
    # total number of combinations computed
    comb = np.math.factorial(n) / (
        np.math.factorial(n - r) * np.math.factorial(r))  # Getting the number of combinations
    comb = int(comb)

    x = np.zeros(comb)
    y = np.zeros(comb)

    count = 0
    for pair in combinations(L1, 2):
        p1 = pair[0]
        p2 = pair[1]
        # print(p1, p2)

        if slope[p1] != 1000000 and slope[p2] != 1000000:
            # When neither line is parallel to y-axis
            x[count] = (intercept[p2] - intercept[p1]) / (slope[p1] - slope[p2])
            y[count] = slope[p1] * (x[count]) + intercept[p1]
        elif slope[p1] == 1000000 and slope[p2] == 1000000:
            # When both the lines are parallel to y-axis, we set the point of intersection 
            # as a very high value so that it lies outside the image and can be disregarded
            x[count] = 100000
            y[count] = 100000
        elif slope[p1] == 1000000:
            # When first line is parallel to y-axis
            x[count] = intercept[p1]
            y[count] = slope[p2]*x[count] + intercept[p2]
        elif slope[p2] == 1000000:
            # When second line is parallel to y-axis
            x[count] = intercept[p2]
            y[count] = slope[p1]*x[count] + intercept[p1]

        # incrementing the counter
        count += 1

    x = (np.round(x)).astype(int)
    y = (np.round(y)).astype(int)
    # print ('x - ', x, '\ny - ', y)

    final_x = np.zeros(4)
    final_y = np.zeros(4)

    count = 0
    # print (l,w)
    # From all possible combinations, only obtain the ones which are the four corners (meaning, inside the grid)
    for i in range(0, x.size):
        if x[i] >= 0 and x[i] <= l:
            if y[i] >= 0 and y[i] <= w:
                final_x[count] = x[i]
                final_y[count] = y[i]
                count += 1
                # print(count)

    # print ('final x - ', final_x, '\nfinal y - ', final_y)

    # The final 4 coodinates
    final_x = final_x.astype(int)
    final_y = final_y.astype(int)

    # Plotting the points on the image.
    img = cv2.imread(filename)
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
    for i in range(0, 4):
        cv2.line(img, (final_x[i], final_y[i]), (final_x[i] + 1, final_y[i] + 1), (0, 0, 255), 5)
    cv2.imwrite('gen/points_plotted.jpg', img)
    # print('final_x - ', final_x, '\nfinal_y - ', final_y)

    return final_x, final_y



#######################################################################################################################
""" Various options for running """
#######################################################################################################################

thresh_method = 1  # 0 for threshold based segmenation; 1 for grabCuts
hough = True  # To perform Hough transform

#######################################################################################################################
""" Main method declared"""
#######################################################################################################################
# Load the image

input_files = 1 # 1 for first video, 2 for second video

if input_files == 1:
    fname = 'video1.list'
    loc = 'video1/'
    gt_loc = 'video1_mat/'
    savename = 'video1_mat.list'
elif input_files == 2:
    fname = 'video2.list'
    loc = 'video2/'
    gt_loc = 'video2_mat/'
    savename = 'video2_mat.list'

with open(fname) as f:
    content = f.readlines()

content = [x.strip() for x in content] 

with open(savename) as f:
    saving = f.readlines()

saving = [x.strip() for x in saving] 
print ('Entering the loop')
# print (content, saving)

overall_count = 0



for ii in range(0, len(content)):
    # print (content[ii])
    filename = loc + content[ii] + '.png'
    # filename = content[ii]
    print ('file name is ', filename)
    # for ii in 
    img = cv2.imread(filename)
    # print (img.shape)

    l, w = (int(img.shape[0]), int(img.shape[1]))

    # If the image size is too large, reshape it to a quarter of original size
    if l > 1000 or w > 1000:
        img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)

    # print(img.shape)
    l, w = (int(img.shape[0]), int(img.shape[1]))
    #rewrite resized image
    resized_orig = img
    cv2.imwrite('gen/resized_maintaining_aspect.jpg', img)

    ############################## Preprocessing the image ################################################################
    # To pre-process the image, we perform image sharpening using CLAHE 

    # Sharpening is performed once
    num_contrast = 1

    # Methods dedined: 1 for CLAHE in LAB and 0 - CLAHE in RGB and 2 - for no sharpening
    contrast_method = 1

    # Performing the sharpening
    img = sharpening(img, num_contrast, contrast_method)

    ################### Threshold based image Segmentation ##########################################################
    # To separate the foreground (the board) from the background (rest of the image).

    
    if thresh_method == 0:
        # thresholding based segmentation
        img, segment = threshold_segmentation(img)
    else:
        # grabCuts based segmentation
        img, segment = grab_cuts_segmentation(img)


    ############################# Hough transform to obtain the lines #####################################################
    #To find intersetion points on the grid by fitting lines along the corners using Hough Transform
    if hough:
        rho, theta = hough_transform(segment)
    # print ('rho - ', rho, '\ntheta - ', theta*(180/np.pi))

    ###################################### K- Means CLustering #############################################################
    # K means clustering is applied to find 4 clusters each representing an edge line of the square for each grid block
    #rho and theta are made into pairs as polar coordinates 
    X = np.column_stack((rho, theta))
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    centres = kmeans.cluster_centers_
    # print (centres, centres.shape)
    r_center, theta_center = centres[:,0], centres[:,1]
    # print (r_center, theta_center)
    # print (kmeans.labels_)
    # predicted = kmeans.labels_

    # The slopes and intercepts are obtained for the 4 edge lines when the slope is infinity
    slope, intercept = generate_slope_intercept(r_center, theta_center, 4)
    # print ('Final slopes - ', slope, '\nFinal intercepts - ', intercept)

    # The 4 slopes and intercepts of the lines need to be intersected with lines having large differnce in slopes such that
    #parallel lines won`t be intersected  
    final_x, final_y = pairwise_intersect(slope, intercept, filename)


    ################################################# Skew correction #####################################################
    #Skew correctio done after corner pixels are obtained and plotted 
    image = resized_orig[:,:,0]
    pts = np.column_stack((final_x, final_y))

    # applying the four point tranform to obtain a "birds eye view" of the image
    warped = four_point_transform(image, pts)

    # show the original and warped images
    img = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("gen/skew_corrected.jpg", img)
    # print (img.shape)

    #Reshaping the size of our image frame and placing it on top of the already loaded 600x600 grid in order to detect the 
    #intersection points for stone detection accurately
    grid_locs = sio.loadmat('XY.mat')

    x = grid_locs['X']
    y = grid_locs['Y']

    x = np.floor(x).astype(int)
    y = np.floor(y).astype(int)

    # Loading the resized image
    skew = cv2.imread('gen/skew_corrected.jpg')
    l_grid, w_grid = 650, 650

    skew_resized = cv2.resize(skew, (l_grid, w_grid))
    cv2.imwrite('gen/skew_resized.jpg', skew_resized)
    # print(skew_resized.shape)

    for i in range(1, x.shape[0] - 1):
        for j in range(1, x.shape[0] - 1):
            # print (i,j) for the resized skewed image 
            cv2.line(skew_resized, (x[i, j], y[i, j]), (x[i, j] + 1, y[i, j] + 1), (255, 0, 0), 5)

    cv2.imwrite('gen/showing_points.jpg', skew_resized)
    #show final resized grid frame


    #-------------------------------------------------------------------------------------------
    # Starting the accuracy computation

    # First, let us compute the average pixel value at the locations specified by the grid.
    # These will be used to classify them as black, white or no stone
    window = 3
    pixel_values = np.zeros(19*19)
    count = 0
    for i in range(1, x.shape[0] - 1):
        for j in range(1, x.shape[0] - 1):
            # print (x[i]-window)
            pixel_values[count] = np.mean(skew_resized[x[i,j]-window:x[i,j]+window, \
                y[i,j]-window:y[i,j]+window])
            count += 1

    # print ('count = ', pixel_values.shape)

    # Let us cluster these by k-means and see what their labels are
    X = np.reshape(pixel_values, [pixel_values.size, 1])
    k_pixel = KMeans(n_clusters=3, random_state=0).fit(X)
    center_pixel = k_pixel.cluster_centers_
    center_pixel = np.reshape(center_pixel, [3,])
    # print ('center pixel - ', center_pixel, center_pixel.shape)
    # print (k_pixel.labels_.shape)

    # Plotting the cluster plot
    classes = ['white stone', 'black stone', 'no stone']
    plt.scatter(X, k_pixel.labels_, c = k_pixel.labels_)
    plt.savefig('scatter_plot.eps', format='eps', dpi=1000)
    plt.title("Scatter plot of the average pixel values")
    plt.xlabel("Pixel number")
    plt.ylabel("Class label")
    plt.legend()

    # Our labels are : 0 - no stone, 1 - black and 2 - white. We need to map these labels 
    # with the ones from k means 
    mapped_labels = np.zeros(361)
    # print ('center_pixel - ', center_pixel)
    for i in range(0,3):
        if center_pixel[i] == np.min(center_pixel):
            min_pix = i
        elif center_pixel[i] == np.max(center_pixel):
            max_pix = i
        else:
            mid_pix = i

    # print (min_pix, max_pix, mid_pix)

    map_locs = np.where(k_pixel.labels_ == min_pix)
    mapped_labels[map_locs] = 1
    map_locs = np.where(k_pixel.labels_ == max_pix)
    mapped_labels[map_locs] = 2
    map_locs = np.where(k_pixel.labels_ == mid_pix)
    mapped_labels[map_locs] = 0
    # print (mapped_labels)

    # Let us load the ground truth
    gt_name = gt_loc + saving[ii] + '.mat'
    # print ('gt_name - ', gt_name)
    gt_labels = sio.loadmat(gt_name)
    gt_labels = gt_labels[saving[ii]]

    # reshaping it into 1D vector
    gt_labels_vec = np.reshape(gt_labels, [361,])

    count = 0

    # No of white stones guessed correctly
    for k in range(0, gt_labels_vec.size):
        if mapped_labels[k] == gt_labels_vec[k]:
            count += 1
    
    # print ('correct guesses', count)
    accuracy = count / 361
    print ('accuracy for image ', ii, '- ', accuracy*100)
    overall_count += count

print ('Overall accuracy - ', overall_count/(361*len(content)))