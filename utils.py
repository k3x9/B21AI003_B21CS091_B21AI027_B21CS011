import shutil
import matplotlib.patches as patches
import os
import time
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def decode_boxes(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if (score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    return [detections, confidences]

def get_hist(img):
    bins = np.arange(0, 300, 10)
    bins[26] = 255
    hp = np.histogram(img, bins)
    return hp

def get_hr(hp, sqrt_hw):
    for i in range(len(hp[0])):
        if hp[0][i] > sqrt_hw:
            return i * 10
        
def get_CEI(img, hr, c):
    CEI = (img - (hr + 50 * c)) * 2
    CEI[np.where(CEI > 255)] = 255
    CEI[np.where(CEI < 0)] = 0
    return CEI
                
def draw(img):
    tmp = img.astype(np.uint8)
    cv2.imshow('image',tmp)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def scale(img):
   s = np.max(img) - np.min(img) 
   res = img / s
   res -= np.min(res)
   res *= 255
   return res

def get_th(img, bins):
    hist = np.histogram(img,bins)
    peak_1_index = np.argmax(hist[0])
    peak_2_index = 0
    if peak_1_index == 0:
        peak_2_index += 1
    for i in range(len(hist[0])):
        if hist[0][i] > hist[0][peak_2_index] and i != peak_1_index:
            peak_2_index = i
    peak_1 = hist[1][peak_1_index]
    peak_2 = hist[1][peak_2_index]
    return ((peak_1 + peak_2) / 2), hist

def get_th2(img, bins):
    num = img.shape[0] * img.shape[1]
    hist = np.histogram(img, bins)
    cdf = 0
    for i in range(len(hist[0])):
        cdf += hist[0][i]
        if cdf / num > 0.85:
            return hist[1][i]

def img_threshold(th, img, flag):
    h = img.shape[0]
    w = img.shape[1]
    new_img = np.zeros((h,w))
    if flag == "H2H":
        new_img[np.where(img >= th)] = 255
    elif flag == "H2L":
        new_img[np.where(img < th)] = 255
    return new_img

def merge(edge, cei):
    h = edge.shape[0]
    w = edge.shape[1]
    new_img = 255 * np.ones((h,w))

    new_img[np.where(edge == 255)] = 0
    new_img[np.where(cei == 255)] = 0
    return new_img

def find_end(tli, x, y):
    i = x
    while(i < tli.shape[0] and tli[i][y] == 0):
        i += 1
    return i - 1

def find_mpv(cei, head, end, y):
    h = []
    e = []
    for k in range(5):
        if head - k >= 0:
            h.append(cei[head-k][y])
        if end + k < cei.shape[0]:
            e.append(cei[end + k][y])
    return np.max(h), np.max(e)
    
def set_intp_img(img, x, y, tli, cei):
    head = x
    end = find_end(tli, x, y)
    n = end - head + 1
    if n > 30:
        return end
    mpv_h, mpv_e = find_mpv(cei, head, end, y)
    for m in range(n):
        img[head+m][y] = mpv_h + (m + 1) * ((mpv_e - mpv_h) / n) 
    return end



def text_enhancement(FILE_NAME, FORMAT):
    c = 0.4
    bl = 260
    FILE_NAME_OUT = 'output_images/' + FILE_NAME + '/' + FILE_NAME
    os.makedirs(f'output_images/{FILE_NAME}', exist_ok=True)
    FILE_NAME = 'input_images/' + FILE_NAME
    shutil.copy(FILE_NAME + FORMAT, FILE_NAME_OUT + FORMAT)

    im = cv2.imread(FILE_NAME + FORMAT)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)

    width = gray.shape[1]
    height = gray.shape[0]

    #STEP: contrast enhancement
    print("Enhancing Contrast")
    hp = get_hist(im)
    sqrt_hw = np.sqrt(height * width)
    hr = get_hr(hp, sqrt_hw)
    cei = get_CEI(gray, hr, c)
    cv2.imwrite(FILE_NAME_OUT + "_Cei" + FORMAT, cei)

    #STEP: Edge detection
    print("Edge Detection")
    # build four filters
    m1 = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape((3,3))
    m2 = np.array([-2,-1,0,-1,0,1,0,1,2]).reshape((3,3))
    m3 = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape((3,3))
    m4 = np.array([0,1,2,-1,0,1,-2,-1,0]).reshape((3,3))

    eg1 = np.abs(cv2.filter2D(gray, -1, m1))
    eg2 = np.abs(cv2.filter2D(gray, -1, m2))
    eg3 = np.abs(cv2.filter2D(gray, -1, m3))
    eg4 = np.abs(cv2.filter2D(gray, -1, m4))
    eg_avg = scale((eg1 + eg2 + eg3 + eg4) / 4)

    bins_1 = np.arange(0, 265, 5) 
    eg_bin = img_threshold(30, eg_avg,"H2H")
    cv2.imwrite(FILE_NAME_OUT + "_EdgeBin" + FORMAT, eg_bin)


    #STEP: Text location
    print("Locating the Text")
    bins_2 = np.arange(0, 301, 40)
    cei_bin = img_threshold(60, cei, "H2L")
    cv2.imwrite(FILE_NAME_OUT + "_CeiBin" + FORMAT, cei_bin)
    tli = merge(eg_bin, cei_bin)
    cv2.imwrite(FILE_NAME_OUT + "_TLI" + FORMAT, tli)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(tli,kernel,iterations = 1)
    cv2.imwrite(FILE_NAME_OUT + "_TLI_erosion" + FORMAT, erosion)

    #STEP: Light distribution
    print("Estimate Light Distribution")
    int_img = np.array(cei)
    ratio = int(width / 20)
    t1 = time.time()
    for y in range(width):
        for x in range(height):
            if erosion[x][y] == 0:
                x = set_intp_img(int_img, x, y, erosion, cei)

            print("Progress: {:.2f}%".format((y * height + x) / (width * height) * 100), end='\r')
    

    mean_filter = 1 / 121 * np.ones((11,11), np.uint8)
    ldi = cv2.filter2D(scale(int_img), -1, mean_filter)
    cv2.imwrite(FILE_NAME_OUT + "_LDI" + FORMAT, ldi)


    #STEP: Light Balancing
    print("Balancing Light and Generating Result")
    result = np.divide(cei, ldi, out=np.zeros_like(cei), where=ldi!=0) * bl
    result[np.where(erosion != 0)] *= 1.5

    cv2.imwrite(FILE_NAME_OUT + "_r" + FORMAT, result)

def adaptive_thresholdMean(img, block_size, c):
    assert block_size % 2 == 1 and block_size > 0, "block_size must be an odd positive integer"
    height, width = img.shape
    binary = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            x_min = max(0, i - block_size // 2)
            y_min = max(0, j - block_size // 2)
            x_max = min(height - 1, i + block_size // 2)
            y_max = min(width - 1, j + block_size // 2)
            block = img[x_min:x_max+1, y_min:y_max+1]
            thresh = np.mean(block) - c
            if img[i, j] >= thresh:
                binary[i, j] = 1

    return binary

def get_line_points(x1, y1, x2, y2):
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    
    if x1 == x2:
        points = []
        if y1 > y2:
            y1, y2 = y2, y1
        for i in range(y1, y2 + 1):
            if i < 0 or i >= 320:
                continue
            points.append((x1, i))
        return points
    else:
        slope = (y2 - y1) / (x2 - x1)
        points = []
        for i in range(x1, x2 + 1):
            if i < 0 or i >= 320:
                continue
            j = int(y1 + (i - x1) * slope)
            if j < 0 or j >= 320:
                continue
            points.append((i, j))
        return points
    
def mask_image(img, boxes, indices):
    main_arr = []
    flag = False

    if len(indices) == 0:
        return np.ones_like(img)

    for i in indices:
        vertices = np.array(cv2.boxPoints(boxes[i]))
        vertices = vertices.astype(int)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [vertices], (255, 255, 255))
        masked_image = cv2.bitwise_and(img, mask)
        mask = cv2.bitwise_not(mask)
        masked_image = cv2.bitwise_or(masked_image, mask)
        masked_image = adaptive_thresholdMean(masked_image, 19, 10)
        total = 0
        black = 0

        # Line 1
        points = get_line_points(vertices[0][0], vertices[0][1], vertices[1][0], vertices[1][1])
        for x, y in points:
            total += 1
            if x < 0 or x >= 320 or y < 0 or y >= 320:
                continue
            if masked_image[y][x] == 0:
                black += 1

        # Line 2
        points = get_line_points(vertices[1][0], vertices[1][1], vertices[2][0], vertices[2][1])
        for x, y in points:
            total += 1
            if x < 0 or x >= 320 or y < 0 or y >= 320:
                continue
            if masked_image[y][x] == 0:
                black += 1

        # Line 3
        points = get_line_points(vertices[2][0], vertices[2][1], vertices[3][0], vertices[3][1])
        for x, y in points:
            total += 1
            if x < 0 or x >= 320 or y < 0 or y >= 320:
                continue
            if masked_image[y][x] == 0:
                black += 1

        # Line 4
        points = get_line_points(vertices[3][0], vertices[3][1], vertices[0][0], vertices[0][1])
        for x, y in points:
            total += 1
            if x < 0 or x >= 320 or y < 0 or y >= 320:
                continue
            if masked_image[y][x] == 0:
                black += 1

        if black > 0.5 * total:
            flag = True
        
        main_arr.append(masked_image)
    
    masked_image = main_arr[0]
    for i in range(1, len(main_arr)):
        masked_image = cv2.bitwise_and(masked_image, main_arr[i])

    if flag:
        temp = np.ones_like(masked_image) * 2
        for i in indices:
            vertices = np.array(cv2.boxPoints(boxes[i]))
            vertices = vertices.astype(int)
            cv2.fillPoly(temp, [vertices], (1, 1, 1))
        
        masked_image = temp - masked_image

    return masked_image

def combined_collage(images, file_paths, FORMAT):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            if i == 2 and j == 2:
                axs[i][j].axis('off')
            else:
                axs[i][j].imshow(images[i * 3 + j], cmap='gray', vmin=0, vmax=1)
                axs[i][j].axis('off')
                rect = patches.Rectangle((0, 0), 320, 320, linewidth=2, edgecolor='r', facecolor='none')
                axs[i][j].add_patch(rect)
    
    # Save
    file_name = file_paths[0].split('/')[-1].split('.')[0]
    plt.savefig(f'output_images/{file_name}/{file_name}_collage{FORMAT}')

    plt.tight_layout()
    plt.show()

def otsu_thresholding(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    hist = cv2.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    
    bins = np.arange(256)
    
    fn_min = np.inf
    thresh = -1
    
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) 
        q1,q2 = Q[i],Q[255]-Q[i]
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i])
        
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i

    # Threshold the image
    new_img = np.zeros_like(img)
    new_img[img > thresh] = 1
    return new_img
    
def sauvola_threshold(img, window_size, k):
    # Check that the window size is odd and nonnegative
    assert window_size % 2 == 1 and window_size > 0, "window_size must be an odd positive integer"

    # Calculate the local threshold for each pixel
    height, width = img.shape
    binary = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            # Calculate the local threshold using a square neighborhood centered at (i, j)
            x_min = max(0, i - window_size // 2)
            y_min = max(0, j - window_size // 2)
            x_max = min(height - 1, i + window_size // 2)
            y_max = min(width - 1, j + window_size // 2)
            block = img[x_min:x_max+1, y_min:y_max+1]
            thresh = np.mean(block) * (1 + k * (np.std(block) / 128 - 1))
            if img[i, j] >= thresh:
                binary[i, j] = 1

    return binary