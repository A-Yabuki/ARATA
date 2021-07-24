# coding: utf-8

import copy
import cv2
import numpy as np

from functools import partial


def timeit(func):
    import time
    def wrapper(*args, **kwargs):
            
        s = time.time()
        res = func(*args, **kwargs)
        e = time.time()
        print(e - s)
        
        return res

    return wrapper


class Library():



    @staticmethod
    def window(cnt):
        # return x, y, w, h
        return cv2.boundingRect(cnt)


    @staticmethod
    def minimum_area_rectangular(cnt):
        
        # Fit a minimum circumscribed rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        return box


    @staticmethod
    def cut_window(img, pts):

        x, y, w, h = pts
        return img[y:y+h, x:x+x+w]


    @staticmethod
    def convertContourArray2BasicArray(cnt):

        return np.array([pts[0] for pts in cnt])


    @staticmethod
    def convertBasicArray2ContourArray(arr):

        return np.array([np.array([pts]) for pts in arr])


    @staticmethod
    def cosine_similarity(x, y):

        return np.dot(x.T, y) / (np.dot(x.T, x) * np.dot(y.T, y))


    @staticmethod
    def angle_calculator(pt1, pt2):

        x1, y1 = np.array(pt1).T
        x2, y2 = np.array(pt2).T

        angles = np.arctan2(x1 - x2, y1 - y2) * (180 / np.pi)
        
        angles[angles<0] += 360

        return angles

    
    @staticmethod
    def bincount_angles(angles):

        bincount = np.bincount(np.int16(angles), minlength=360)

        majority = np.argmax(bincount)

        ranged_majority = 0

        lower = majority - 2
        upper = majority + 2

        if lower < 0:
            ranged_majority += np.sum(bincount[lower + 360:])
            ranged_majority += np.sum(bincount[:upper])

        elif upper >= 360:
            ranged_majority += np.sum(bincount[lower:])
            ranged_majority += np.sum(bincount[:upper-360])
        
        else:
            ranged_majority += np.sum(bincount[lower:upper])

    
        return ranged_majority / len(angles)  / (1+np.exp(-0.5*(len(angles)-4))), majority


    @classmethod
    def majority_rate_of_angles(self, kp1, kp2):

        if len(kp1) == 0 or len(kp2) == 0:
            return 0, 0

        angles = self.angle_calculator(kp1, kp2)
        
        majority_rate, majority = self.bincount_angles(angles)

        return majority_rate, majority





    @staticmethod
    def cam_shift(img_prev, img_current, cnts_prev, key):

        a = np.uint8((2 << (2*key+1) % 8) + (2**key + 5*key + 3)) 
        b = np.uint8((2 << (2*key + 3) % 8) + (2**key + 7*key - 1))
        c = np.uint8((2 << (2*key + 5) % 8) + (2**key - 9*key + 9))
        
        
        x1, y1, x2, y2 = 10 ** 5, 10 ** 5, 0, 0

        for cnt in cnts_prev:
            x_, y_, w_, h_ = Library.window(cnt)

            x1 = min(x1, x_)
            y1 = min(y1, y_)
            x2 = max(x2, x_ + w_)
            y2 = max(y2, y_+h_)

        print(a,b,c)
        # vertex coodinates of a track window
        track_window = (x1, y1, x2-x1, y2-y1)

        # set up the ROI for tracking
        roi = img_prev[y1: y2, x1: x2]

        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

        # calcHist... arguments = images, target ch, mask, binSize, value range of pixels to calculate hist
        roi_hist=cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

        # normalize... arguments = image, dst, alpha(norm value or lower range boundary), beta(upper range boundary), normType, dtype, mask
        # normType... NORM_INF=norm normalization (alpha only), NORM_MINMAX=range_normalization(alpha and beta)
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


        img_current = cv2.cvtColor(img_current, cv2.COLOR_BGR2HSV)  
        
        # calcBackProject = calc back projection of a histogram
        # arguments = arrays, channels, hist, ranges, scale
        # ヒストグラムで表現された経験的確率分布における，各要素値の確率を求める関数
        dst = cv2.calcBackProject([img_current], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        # CamShift... arguments = probImage, window, stop criteria
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
                
        # Draw it on image
            
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)

        img2 = cv2.polylines(img_current, [pts], True, list(map(int, (a,b,c))), 2)
        
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            pass
        else:
            cv2.imwrite(chr(k) + ".jpg", img2)
            
        return img2, pts


    @staticmethod
    def orb_bfmatcher(img1, img2):

        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # Draw first 10 matches.
        #img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,flags=2)
        #cv2.imshow("w", img3)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return [m.distance for m in matches][:20]


    @staticmethod
    def akaze_bfmatcher(img1, img2):

        # Initiate AKAZE matcher
        akaze = cv2.AKAZE_create()

        # find the keypoints and descriptors with AKAZE
        kp1, des1 = akaze.detectAndCompute(img1, None)
        kp2, des2 = akaze.detectAndCompute(img2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher()

        # Match descriptors.
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        ratio = 0.75
        good = []
        for m,n in matches:
            if m.distance < ratio*n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        #img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=0)
        
        #Library.showImage("akaze", img3)

        good = sorted(good, key=lambda x: x[0].distance)

        img1_pt = [list(map(int, kp1[m[0].queryIdx].pt)) for m in good]
        img2_pt = [list(map(int, kp2[m[0].trainIdx].pt)) for m in good]

        return good, img1_pt, img2_pt
        

    @staticmethod
    def orb_flann(img1, img2):

        # Initiate AKAZE detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with AKAZE
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        
        
        # FLANN parameters
        FLANN_INDEX_LSH = 0
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
        search_params = dict(checks=50)  # or pass empty dictionary
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(np.float32(des1),np.float32(des2),k=2)

        # Need to draw only good matches, so create a mask
        #matchesMask = [[0,0] for i in range(len(matches))]

        good = []

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                #matchesMask[i] = [1, 0]
                good.append(m.distance)

        """
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = 0)

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        Library.showImage("flann", img3)
        """
        return sorted(good)


    @staticmethod
    def akaze_flann(img1, img2):

        # Initiate AKAZE detector
        akaze = cv2.AKAZE_create()

        # find the keypoints and descriptors with AKAZE
        kp1, des1 = akaze.detectAndCompute(img1,None)
        kp2, des2 = akaze.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(np.float32(des1),np.float32(des2),k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        good = []
        
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i] = [1, 0]
                good.append([m])
        """
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = 0)
        
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

        Library.showImage("flann", img3)
        """

        return sorted(good, key=lambda m:m.distance)


    @staticmethod
    def match_template(tmp, img):

        y_tmp, x_tmp = tmp.shape[:2]
        y_img, x_img = img.shape[:2]

        height = max(y_tmp, y_img)
        width = max(x_tmp, x_img) 

        canvas1 = np.zeros((height, width, 3), dtype=np.uint8)

        canvas2 = copy.deepcopy(canvas1)

        canvas1[:y_tmp,:x_tmp, :] = tmp
        canvas2[:y_img,:x_img, :] = img

        # apply template matching
        res = cv2.matchTemplate(canvas1, canvas2, cv2.TM_CCOEFF_NORMED)

        # returned values... min_val, max_val, min_loc, max_loc
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        return max_val



    @classmethod
    def compare_matcher(self, temp_img, temp_img2):

        match = timeit(self.orb_bfmatcher)(temp_img, temp_img2)
        akaze = timeit(self.akaze_bfmatcher)(temp_img, temp_img2)
        akaze_flann = timeit(self.akaze_flann)(temp_img, temp_img2)
        orb_flann = timeit(self.orb_flann)(temp_img, temp_img2)

        matching_list = [match, akaze, akaze_flann, orb_flann]

        min_d = list(map(np.min, matching_list))
        avg_d = list(map(np.mean, matching_list))
        sum_d = list(map(np.sum, matching_list))
        mean_top_d = list(map(np.mean, [m[:10] for m in matching_list]))
        mean_top_d2 = list(map(np.mean, [m[:20] for m in matching_list]))

        return matching_list, min_d, avg_d, sum_d, mean_top_d, mean_top_d2


    @staticmethod
    def drawMultiContours(canvas, cnts, color, thickness):

        [cv2.drawContours(canvas, [cnt], -1, color, thickness) for cnt in cnts]

        return canvas
        
    @staticmethod
    def showImage(window_name, img):
        
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




class RootIdentifier():

    def __init__(self, img_paths, label_paths):

        self.img_paths = img_paths
        self.label_paths = label_paths
        self.root_dict = RootDict()
        self.root_count = 0
        

    def run(self):

        for i, j in zip(self.img_paths, self.label_paths):

            self.img1 = cv2.copyMakeBorder(cv2.imread(i, cv2.IMREAD_UNCHANGED),10,10,10,10,cv2.BORDER_CONSTANT, value=0)
            lab = cv2.imread(j, cv2.IMREAD_UNCHANGED)
            
            bin_lab = np.where((lab[:,:, 0] == 255) | (lab[:,:, 2] == 255), 255, 0).astype(np.uint8)

            cnts, hierarchy = cv2.findContours(bin_lab, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            
            # 親を持たない輪郭＝最外輪郭のみのindexを取り出す
            cnts_id = np.where((np.asarray(hierarchy[0], dtype=np.int32))[:, 3] == -1)

            # 親を持つ輪郭 = 内部輪郭のみのindex を取り出す
            cnts_id_internal = np.where((np.asarray(hierarchy[0], dtype=np.int32))[:, 3] != -1)

            cnts = np.array(
            [np.asarray(cnt, dtype=np.int32) for cnt in cnts])

            # 内部輪郭で囲まれる領域のマスク
            cnts_internal = cnts[cnts_id_internal[0]]
            self.mask_internal_cnts = Library.drawMultiContours(np.zeros((bin_lab.shape), dtype=np.uint8), cnts_internal, 1, -1) 


            # 最外輪郭のみ取り出す 内側の輪郭は面積計算時に元ラベルとのマッチングで消しましょう
            cnts = cnts[cnts_id[0]]
            
            ret = self.register_roots(cnts, minimum_size=300, img_size=bin_lab.shape, matcher_type='akaze')

            self.img2 = self.img1
            
            
            print("name: %s total root num: %s current root num: %s" % (i, self.root_count, len(self.root_dict)), ret)
            
            mask = np.zeros((bin_lab.shape), dtype=np.uint8)
            mask = self.root_dict.drawRoots(mask)

            test = np.copy(self.img1)

            test[mask == 0] = 0
            
            Library.showImage(i, test)
                




    def register_roots(self, cnts, minimum_size, img_size, matcher_type='orb'):

        matcher = Library.orb_bfmatcher if matcher_type == "orb" else Library.akaze_bfmatcher
        rect_drawer = partial(cv2.rectangle, color=1, thickness=-1)
        cnt_drawer = partial(cv2.drawContours, contourIdx=-1, color=1, thickness=-1)
        multi_cnt_drawer = partial(Library.drawMultiContours, color=1, thickness=-1)

            

        def register(apex, cnt, size):

            # rectangular shape mask
            
            canvas = np.zeros(size, dtype=np.uint8)
            x, y, w, h = apex
            canvas = rect_drawer(canvas, (x, y), (x + w, y + h))
            #cnt_drawer(canvas, [apex])

            # root contour mask
            mask = np.zeros(size, dtype=np.uint8)
            cnt_drawer(mask, [cnt])

            # Temporary image for measuring similarity based on ORB
            temp_img = np.copy(self.img1)

            # apply root contour mask
            temp_img[mask == 0] = 0

            if self.root_dict:

                for no, values in self.temp_dict.items():

                    # for rectangular shape mask
                    canvas2 = np.zeros(size, dtype=np.uint8)

                    # for root contour mask
                    mask2 = np.zeros(size, dtype=np.uint8)

                    # Temporary image for measureing similarity based on ORB
                    temp_img2 = np.copy(self.img2)

                    # get stored values about roots in previous images
                    pts, cnts2, gone_count, status, overlap_count = values

                    #Library.cam_shift(self.img2, self.img1, cnts2, no)

                    if status:
                        
                        # draw rectangle mask
                        for i in pts:
                            x, y, w, h = i
                            canvas2 = rect_drawer(canvas2, (x, y), (x + w, y + h))
                            #cnt_drawer(canvas2, [i])

                        # calculate IoU
                        # strength: calculation cost
                        # weakness: movement, rotation 
                        IoU = np.sum(cv2.bitwise_and(canvas, canvas2)) / np.sum(cv2.bitwise_or(canvas, canvas2))


                        ### judge same root or not

                        # regard as the same root
                        if IoU >= 0.75:

                            # renew root location and other information
                            self.root_dict[no] = [[apex], [cnt], 0, status, overlap_count]
                            
                            # set key as the evidence that the root still exists.
                            self.key_set.add(no)

                            return True
                        
                        
                        # candidate of the same root
                        if IoU > 0:
                            
                            # mask for extracting a root from the original image.
                            mask2 = multi_cnt_drawer(mask2, cnts2)
                                
                            # remove internal area
                            mask2[self.mask_internal_cnts != 0] = 0
                                
                            # extract a root from the original image.
                            temp_img2[mask2 == 0] = 0
                            
                            # brute force matching
                            match, kp1, kp2 = matcher(temp_img, temp_img2)
                            
                            # get distance between key points
                            distance = [m[0].distance for m in match]
                            
                            # if no counterpart keypoints
                            if len(distance) == 0:
                                # not same root
                                continue

                            # matching score
                            
                            # objective: to evaluate the similarity of keypoints
                            # strength: able to evaluate the similarity of the root shape
                            # weakenss: Because of the size effect, simple thresholds make the results unstable
                            least_distance = np.min(distance)
                            least_20_mean_distance = np.mean(distance[:20])
                            
                            #match_length_per_cnt = len(distance) / sum(list(map(len, cnts)))
                            
                            # calculate the sum of distance / root area
                            # objective: the number of keypoints detected by akaze depends on the object scale in the image.
                            #            so,  as the root area increases, more points are detected and the sum of distances increases.
                            #            this value can remove the effect of the size of area
                            # strength: robust to "size effect"
                            # weakness: 

                            mean_distance_area_base = np.sum(distance) / np.min([np.sum(mask), np.sum(mask2)])
                            
                            # calculate the number of major angle / len(angles)
                            # objective: to evaluate the similarity of angles between two key points 
                            # strength: robust to rotation, movement
                            # weakness: often returns low value, even if target is same root
                            majority_rate, majority = Library.majority_rate_of_angles(kp1, kp2)

                            # template matching
                            # objective: to evaluate color and shape similarity
                            # strength: able to evaluate color similarity
                            # weakness: vulnerable to rotation
                            tm = Library.match_template(Library.cut_window(self.img1, apex), Library.cut_window(self.img2, pts[0]))
                            score = 0


                            if IoU >= 0.5:
                                score += 5

                            elif IoU >= 0.3:
                                score += 3

                            else:
                                score -= 1


                            if least_distance <= 50:
                                score += 5

                            elif least_distance <= 100:
                                score += 3

                            elif least_distance <= 200:
                                score += 1

                            elif least_20_mean_distance > 300:
                                score -= 1


                            # distance
                            if least_20_mean_distance <= 100:
                                score += 5

                            elif least_20_mean_distance <= 200:
                                score += 3

                            elif least_20_mean_distance <= 300:
                                score += 1

                            elif least_20_mean_distance > 300:
                                score -= 1


                            # distance / area
                            if mean_distance_area_base >= 1.0:
                                score += 5

                            elif mean_distance_area_base >= 0.5:
                                score += 3
                            
                            elif mean_distance_area_base >= 0.3:
                                score += 1

                            elif mean_distance_area_base < 0.1:
                                score -= 3


                            # angle
                            if (majority <= 45) or (majority >= 315) or ((majority >= 135) and (majority <= 225)):
                                
                                if majority_rate >= 0.5:
                                    score += 3

                                elif majority_rate >= 0.2:
                                    score += 1

                            else:
                                score -= 1


                            if tm >= 0.7:
                                score += 5

                            elif tm >= 0.4:
                                score += 3

                            elif tm >= 0.1:
                                score += 1

                            elif tm < 0.1:
                                score -= 3
                            


                            #condition1 = least_distance <= 50
                            #condition2 = least_20_mean_distance <= 200
                            #condition3 = mean_distance_area_base >= 0.3
                            #condition4 = majority_rate >= 0.2

                            # regard as the same root
                            if score >= 5:#(condition1 or condition2 or condition3 or condition4):
                                
                                # renew root location and other information
                                self.root_dict[no] = [[apex], [cnt], 0, status, overlap_count]
                                
                                # set key as the evidence that the root still exists.
                                self.key_set.add(no)

                                #if score >= 10:
                                    #del self.temp_dict[no]

                                return [True, no, IoU, tm, score, least_distance, least_20_mean_distance, mean_distance_area_base, majority, majority_rate]

                            print([False, no, IoU, tm, score, least_distance, least_20_mean_distance, mean_distance_area_base,  majority, majority_rate])
                            #return [False, no, IoU, tm, score, least_distance, least_20_mean_distance, mean_distance_area_base,  majority, majority_rate]


                            
                # 同一根無し

                self.root_dict[self.root_count] = [[apex], [cnt], 0, True, 0]
                self.root_count += 1
                                

                
                return False
            
            # 根無し
            else:

                self.root_dict[self.root_count] = [[apex], [cnt], 0, True, 0]
                self.root_count += 1

                return None
                

        self.temp_dict = copy.deepcopy(self.root_dict)
        
        ret_list = []
        self.key_set = set()
        
        for cnt in cnts:

            if len(cnt) < minimum_size:
                continue


            #area, length = self.calc_parameters(cnt)
            apex = Library.window(cnt)
            #apex = Library.minimum_area_rectangular(cnt)
    
            ret = register(apex, cnt, img_size)

            ret_list.append(ret)
            #print(ret)


        # overlap数のリセット
        self.root_dict.reset_overlap_count()


        for key in self.temp_dict.keys():

            if not(key in self.key_set):
                count = self.root_dict[key][2]

                count += 1

                if count >= 3:
                    #self.root_dict[key][3] = False
                    del self.root_dict[key]

                else:
                    self.root_dict[key][2] = count
                 

        return ret_list


class RootDict():

    def __init__(self):

        self.dictionary = {}


    def __getitem__(self, key):

        return self.dictionary[key]


    def __setitem__(self, key, values):

        # 重複値 += 1  
        values[4] += 1

        # 既出根の場合
        if (key in self.dictionary.keys()) and (self.dictionary[key][4] != 0):    
            self.dictionary[key][0].append(values[0][0])
            self.dictionary[key][1].append(values[1][0])
            

        # 新しい根 or 重複なしの場合
        else:
            self.dictionary[key] = values


    def __delitem__(self, key):

        del self.dictionary[key]


    def __len__(self):

        return len(self.dictionary)


    def keys(self):

        return self.dictionary.keys()


    def values(self):

        return self.dictionary.values()


    def items(self):

        return self.dictionary.items()


    def reset_overlap_count(self):

        for key in self.dictionary.keys():
            
            self.dictionary[key][4] = 0


    def drawRoots(self, canvas):

        for key, values in self.dictionary.items():

            cnts = values[1]
            Library.drawMultiContours(canvas, cnts, 1, -1)

            # calculate suitable location to put a text to recognize each root.
            pts = [i[0] for i in cnts[0]]
            pts = sorted(pts, key=lambda x: x[0])

            mid_x = len(pts) // 2
            
            # add key name near the target root
            # img, text, org, font_style, font_scale, color, thickness, linetype

            cv2.putText(canvas, str(key), tuple(pts[mid_x]),
               cv2.FONT_HERSHEY_PLAIN, 6,
               255, 3, cv2.LINE_AA)
  
        return canvas




            
"""
メモ書き

根の合体問題の解決方法

道筋：同一の根が分離合体する場合、自身の存在していた場所内で起こるはず(例外あり)
解決策：個別の根の最大範囲を記録しておく

道筋：異なる根が合体する場合、その前の状態で、異なる物として記録されているはず
解決策：前に記録された根と二つ以上マッチする場合、合体していると判定。伸長方向と根の輪郭より分離位置を推定。
"""



import glob

img_paths = glob.glob("test/img1/*.*")
label_paths = glob.glob("test/label1/*.png")

RI = RootIdentifier(img_paths, label_paths)

RI.run()

cv2.destroyAllWindows()