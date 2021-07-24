# coding: utf-8

import cv2
import os
from typing import List, Tuple

import exifread
import numpy as np

from arata.common.constants import ResourcePathConst


def make_diff_images(src_paths: List[str], increment_dst_path: str, decrement_dst_path: str) -> None:
    
    r""" 
    Take the difference of images that are related before and after by given path.  
    
    When taking the difference, values under 0 are assumed to be zero.

    Args:
        src_paths (List[str]): paths of source images
        increment_dst_path (str): save location of increment results. (calculated by this formula: result = next - prev )
        decrement_dst_path (str): save location of decrement results. (calculated by this formula: result = prev - next)
    """

    for i, _ in enumerate(src_paths[1:], start=1):

        next_img = cv2.imread(src_paths[i], cv2.IMREAD_COLOR).astype(np.int32)
        prev_img = cv2.imread(
            src_paths[i-1], cv2.IMREAD_COLOR).astype(np.int32)

        h, w = next_img.shape[:2]
        increment = next_img - prev_img
        decrement = prev_img - next_img

        src_name_1 = os.path.split(src_paths[i])[1]

        if i == 1:
            src_name_2 = os.path.split(src_paths[i-1])[1]

            cv2.imwrite(increment_dst_path +
                        '/{}.png'.format(src_name_1), increment)
            cv2.imwrite(increment_dst_path +
                        '/{}.png'.format(src_name_2), np.zeros((h, w, 3)))

            cv2.imwrite(decrement_dst_path +
                        '/{}.png'.format(src_name_1), decrement)
            cv2.imwrite(decrement_dst_path +
                        '/{}.png'.format(src_name_2), np.zeros((h, w, 3)))

        else:
            cv2.imwrite(increment_dst_path +
                        '/{}.png'.format(src_name_1), increment)
            cv2.imwrite(decrement_dst_path +
                        '/{}.png'.format(src_name_1), decrement)


def binarize(src, bgr_lower, bgr_upper):
    return cv2.inRange(src, bgr_lower, bgr_upper)


# BGRで特定の色を抽出する関数
def extract_color(src, bgr_lower, bgr_upper):
    gray = binarize(src, bgr_lower, bgr_upper) # BGRからマスクを作成
    result = cv2.bitwise_and(src, src, mask=gray) # 元画像とマスクを合成
    return img_bin


class PhaseOnlyCorrelation():

    r"""
    Based on the Phase Only Correlation method,
    relocates the center position of an image to become a similar arrangement of the objects in the base image. 
    """

    @staticmethod
    def relocate(actor: 'np.ndarray[np.uint8]', stage: 'np.ndarray[np.uint8]') -> Tuple['np.ndarray[np.uint8]', Tuple[int, int]]:

        """ position shift correction by phase only correlation """

        gray_actor = cv2.cvtColor(actor, cv2.COLOR_BGR2GRAY)
        gray_stage = cv2.cvtColor(stage, cv2.COLOR_BGR2GRAY)
        
        rows, cols = gray_stage.shape
        height, width = gray_actor.shape
        
        y = rows if rows < height else height
        x = cols if cols < width else width
        
        clipped_actor = gray_actor[0:y, 0:x]
        clipped_stage = gray_stage[0:y, 0:x]
        
        try:
            distance, etc = cv2.phaseCorrelate(clipped_actor.astype(np.float), clipped_stage.astype(np.float))

        except:
            error_message = "Operation [Phase only correlation] is failed. "\
                            "Image size... img 1: %d, %d , img 2: %d, %d"%(rows, cols, height, width)
            
            raise(ArithmeticError(error_message))

        else:
            dy, dx = distance
            M = np.float32([[1, 0, dy], [0, 1, dx]])
            acted = cv2.warpAffine(actor, M, (cols, rows))

            return acted, (int(dy), int(dx))


########################## Control Adjustment Tools ################################

class ContrastControler():

    r"""
    Method Aggregation Class of Control Contrast
    """

    @staticmethod
    def gamma_transformation(img, gamma):

        LUT_G = np.zeros((256, 1), dtype = np.uint8 )
        for i in range(256):
            LUT_G[i, :] = 255 * pow(float(i) / 255, 1.0 / gamma)

        #LUT_G = np.broadcast_to(LUT_G[np.newaxis,:], (3, 256))
        gamma_img = cv2.LUT(img.astype(np.uint8), LUT_G.astype(np.uint8))
        
        return gamma_img

    @staticmethod
    def high_contrast(img, min_table, max_table):

        min_table = int(min_table)
        max_table = int(max_table)
        LUT_HC = np.zeros((256, 1), dtype = np.uint8 )
        diff_table = max_table - min_table
        # ハイコントラストLUT作成
        for i in range(0, min_table):
            LUT_HC[i, :] = 0
        for i in range(min_table, max_table):
            LUT_HC[i, :] = 255 * (i - min_table) / diff_table
        for i in range(max_table, 255):
            LUT_HC[i, :] = 255
        # 変換
        #LUT_HC = np.broadcast_to(LUT_HC[np.newaxis,:], (3, 256))
        high_cont_img = cv2.LUT(img.astype(np.uint8), LUT_HC.astype(np.uint8))
        return high_cont_img

    @staticmethod
    def low_contrast(img, min_table, max_table):

        min_table = int(min_table)
        max_table = int(max_table)
        LUT_LC = np.zeros((256, 1), dtype = np.uint8 )
        
        diff_table = max_table - min_table
        # ローコントラストLUT作成
        for i in range(256):
            LUT_LC[i, :] = min_table + i * (diff_table) / 255

        # 変換
        #LUT_LC = np.broadcast_to(LUT_LC[np.newaxis,:], (3, 256))
        low_cont_img = cv2.LUT(img.astype(np.uint8), LUT_LC.astype(np.uint8))
        return low_cont_img 

    @staticmethod
    def clahe(img, clipLimit=2, tileGridSize=(8,8)):

        img = np.asarray(img, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        for ch in range(3):
            img[:,:,ch] = clahe.apply(img[:,:,ch])

        return img

############################################################################

########################## FFT Tools ################################

class FFT():

    @staticmethod
    def apply_fft_filter(src, a):

        # 高速フーリエ変換(2次元)
        src = np.fft.fft2(src)
        
        # 画像サイズ
        h, w = src.shape
    
        # 画像の中心座標
        cy, cx =  int(h/2), int(w/2)
        
        # フィルタのサイズ(矩形の高さと幅)
        rh, rw = int(a*cy), int(a*cx)

        # 第1象限と第3象限、第1象限と第4象限を入れ替え
        fsrc =  np.fft.fftshift(src)  

        # 入力画像と同じサイズで値0の配列を生成
        fdst = np.zeros(src.shape, dtype=complex)
        #outFilter = fdst.real
        
        # 中心の縦と横の値を抽出
        fdst[:, cx - rw : cx + rw] = fsrc[:, cx - rw : cx + rw]
        fdst[cy - rh : cy + rh, :] = fsrc[cy - rh : cy + rh, :]

        # 第1象限と第3象限、第1象限と第4象限を入れ替え(元に戻す)
        fdst =  np.fft.fftshift(fdst)

        # 高速逆フーリエ変換 
        dst = np.fft.ifft2(fdst)
        
        # 実部の値のみを取り出し、符号なし整数型に変換して返す
        return np.uint8(dst.real) #, np.uint8(outFilter)

    @classmethod
    def remove_stripe_like_noise(cls, img, a):

        h, w = img.shape[:2]
        img = cv2.copyMakeBorder(img, h % 2, 0, w % 2, 0, cv2.BORDER_REPLICATE)
        filtered = np.zeros((h + h % 2, w + w % 2, 3), dtype=np.uint8)
        
        for i in range(3):
            filtered[:, :, i] = cls.apply_fft_filter(img[:, :, i], a)

        ret = img.astype(np.int32) - filtered.astype(np.int32) + np.mean(img, axis=(0, 1), dtype=np.int32)
        ret = np.clip(ret, 0, 255).astype(np.uint8)
        return np.uint8(ret[h % 2 :, w % 2 :])


    @staticmethod
    def apply_mean_filter(img, ksize, subtraction = True):

        # 元画像 - 平滑化画像 + 平均値
        smoothed = cv2.blur(img, (ksize, ksize))
        mean = np.mean(img, axis=(0, 1))

        if subtraction:
            img = img.astype(np.float32) - smoothed.astype(np.float32) + mean.astype(np.float32) 
        
        else:
            img = (img.astype(np.float32) / smoothed.astype(np.float32)) * mean.astype(np.float32) 

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img  #, smoothed

############################################################################

########################## Random Image Transformation Tools ################################

class Transformer():

    @staticmethod
    def rotate_randomly(img, lbl):

        theta = np.random.choice(range(-15,15))
        
        # 回転角度・拡大率
        p = np.random.uniform(low=1.0, high=1.25, size=(1))
        scale = p
                    
        # 画像の中心座標
        oy, ox = int(img.shape[0]/2), int(img.shape[1]/2)

        # 方法2(OpenCV)
        R = cv2.getRotationMatrix2D((ox, oy), theta, scale)    # 回転変換行列の算出
            
        img = cv2.warpAffine(img, R, img.shape[:2], flags=cv2.INTER_CUBIC)    # アフィン変換
        lbl = cv2.warpAffine(lbl, R, lbl.shape[:2], flags=cv2.INTER_NEAREST)    # アフィン変換
            
        return img, lbl

    
    @staticmethod
    def flip_randomly(*args):

        p = np.random.randint(-1, 2, size=(1))

        return [cv2.flip(img, p) for img in args]


class CutMix():

    def __init__(self):
        self.pos1 = None
        self.pos2 = None
        self.cut_img = None
        self.cut_lbl = None
        self.rng = np.random.default_rng()


    def _sort(self, a, b):

        if a <= b:
            return a, b

        else:
            return b, a

    def cut_or_mix(self, img, lbl):

        if (self.cut_img is None or self.cut_lbl is None):
            return self.mix(img, lbl)

        else:
            self.cut(img, lbl)


    def cut(self, img, lbl):

        h, w = img.shape[:2]
        
        y1, y2 = self.rng.integers(0, h, 2)
        x1, x2 = self.rng.integers(0, w, 2)

        x_min, x_max = self._sort(x1, x2)
        y_min, y_max = self._sort(y1, y2)

        cut_img = img[ y_min : y_max, x_min : x_max ]
        cut_lbl = lbl[ y_min : y_max, x_min : x_max ]

        self.pos1 = (x_min, y_min)
        self.pos2 = (x_max, y_max)
        self.cut_img = cut_img
        self.cut_lbl = cut_lbl


    def mix(self, img, lbl):

        if (self.cut_img is None or self.cut_lbl is None):
            return img, lbl

        h, w = img.shape[:2]
        x3 = self.rng.integers(0, self.pos1[0], 1)[0] if self.pos1[0] != 0 else 0
        y3 = self.rng.integers(0, self.pos1[1], 1)[0] if self.pos1[1] != 0 else 0

        cut_width = self.pos2[0] - self.pos1[0]
        cut_height = self.pos2[1] - self.pos1[1]

        img[ y3 : y3 + cut_height, x3 : x3 + cut_width ] = self.cut_img
        lbl[ y3 : y3 + cut_height, x3 : x3 + cut_width ] = self.cut_lbl

        self.cut_img = None
        self.cut_lbl = None
        return img, lbl

############################################################################

########################## Random Noise Creation Tools ################################

class NoiseCreator():

    rng = np.random.default_rng()
    
    def __init__():
        pass

    @classmethod
    def add_point_noise(cls, img, strength = 0.2, possibility = 0.5):

        """
        Add random values to random pixcels
        """

        noise = np.ones_like(img) * strength * (img.max() - img.min())
        
        noise[cls.rng.random(size=noise.shape) > possibility] *= -1
        
        img = img + noise
        
        img[img > 255] = 255
        
        img[img < 0] = 0
        
        return img


    @staticmethod
    def add_bokeh(img, scale1, scale2, scale3):

        """
        Applying 3 smoothing filters of specified scales. 

        and

        Averaging those outputs
        
        """
        
        blur1 = cv2.blur(img, (scale1, scale1)).astype(np.int32)
        blur2 = cv2.blur(img, (scale2, scale2)).astype(np.int32)
        blur3 = cv2.blur(img, (scale3, scale3)).astype(np.int32)

        img = ((blur1 + blur2 + blur3) // 3).astype(np.uint8)
        
        return img


    @classmethod
    def draw_marker_line(cls, img, p1: Tuple[int, int], p2: Tuple[int, int] , thickness: int, linecolor: str="black"):

        """
        input: A BGR image
        output: A BGR image added line drown with ink 
        """

        h, w = img.shape[:2]
        
        # インクのカスレ具合を表現する為の直線に濃淡をつけるマスク画像
        mask = cv2.imread(ResourcePathConst.BLUR_MASK, cv2.IMREAD_GRAYSCALE)
        
        mask_h, mask_w = mask.shape

        # マスク画像のどの部分を取るかランダムに決める
        maskX = cls.rng.integers(low=0, high=mask_h-h, size=1)[0]
        maskY = cls.rng.integers(low=0, high=mask_w-w, size=1)[0]
        mask = mask[maskX : maskX + h, maskY : maskY + w]

        # 白地キャンバス
        white = np.ones((h, w), dtype=np.uint8)*255

        cv2.line(white, p1, p2, color=(0,0,0), thickness=thickness)
            
        # 黒線をぼかす
        white = cv2.GaussianBlur(white, (9, 9), sigmaX=5, sigmaY=5)

        # 白地を黒くする
        black = np.where(white>=255, 0, white).astype(np.int32)

        # 上の操作で線の端の部分が最も白くなり線の中心が黒くなる
        
        # マスクを0~1の値にして、黒地白線画像にかけ合わせる

        masked = (mask.astype(np.float32)/255) * black

        # 線の部分以外を黒に
        masked = np.where(black <= 0, 0, masked)

        if linecolor == "black":
            
            masked = np.broadcast_to(masked[:,:,np.newaxis], (h, w, 3))

            img = img.astype(np.int32) - masked

            img[img>255] = 255
            img[img<0] = 0


        elif linecolor == "waterblack":

            
            img = np.multiply(img.astype(np.int32), (white/255)[:,:,np.newaxis])



        elif linecolor == "waterwhite":

            # 255-imgで255までの余裕を測る
            # それに黒地に白線のマスクをかけることでimgとの合計が255を超えないようにできる
            img = img +  np.multiply((255 - img.astype(np.int32)), (1 - white/255)[:,:,np.newaxis])


        # 赤色は未完成　
        elif linecolor == "red":
            
            red_mean = np.mean(img.astype(np.int32)[:,:,2])

            red = (255 - red_mean) * masked

            red = img.astype(np.int32)[:,:,2] + red

            red[red>255] = 255
            red[red<0] = 0
            
            red = red.astype(np.uint8)

            img = np.concatenate((img[:,:,0:2], red[:,:,np.newaxis]), axis=2)

        return img.astype(np.uint8)


    @staticmethod
    def make_periphery_darker(img, strength, grad_type="circle"):

        """
        Imitating darker area surrounding image edges

        input: A BGR image
        output: A BGR image whose periphery is darker than what it was
        """

        h, w = img.shape[:2]
        
        distantFromCenter = np.zeros((h,w))

        if grad_type=="circle":
            # 1 - 画素ごとの画像中心からの正規化した距離 = 中心から遠いほど０に近い
            for i in range(h):
                for j in range(w):
                    distantFromCenter[i, j] = 1 - (((h/2) - i)**2 + ((w/2) - j)**2 )/ ((h/2)**2+(w/2)**2)

        if grad_type=="linear":
            for j in range(w):
                distantFromCenter[:, j] = 1 - (((w/2) - j)**2 /  (w/2)**2)

        distantFromCenter = np.broadcast_to(
                                distantFromCenter[:,:,np.newaxis], (h, w, 3))
        
        # 外縁部が黒くなる
        distant_img = np.multiply(img, distantFromCenter**strength)
        
        img = distant_img
        
        img = img.astype(np.uint8)

        return img


    @staticmethod
    def add_salt_pepper(src, salt=True):

        """
        Adding salt or pepper
        """

        row,col,ch = src.shape
        s_vs_p = 0.5
        amount = 0.004
        sp_img = src.copy()

        if salt:
            # 塩モード
            num_salt = np.ceil(amount * src.size * s_vs_p)
            coords = [np.random.randint(0, i-1 , int(num_salt)) for i in src.shape]
            sp_img[coords[:-1]] = (255,255,255)

        else:
            # 胡椒モード
            num_pepper = np.ceil(amount* src.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in src.shape]
            sp_img[coords[:-1]] = (0,0,0)

        return sp_img


    @staticmethod
    def apply_unsharpmask(img_src, k=2.0):
        
        # k..シャープの度合い

        # シャープ化するためのオペレータ
        sharp_operator = np.array([[-k/9.0, -k/9.0, -k/9.0],
                    [-k/9.0, 1 + (8 * k)/9.0, -k/9.0],
                    [-k/9.0, -k/9.0, -k/9.0]])
    
        # 作成したオペレータを基にシャープ化
        img_tmp = cv2.filter2D(img_src, -1, sharp_operator)
        img_sharp = cv2.convertScaleAbs(img_tmp)
    
        return img_sharp


class RandomNoiseCreator(NoiseCreator):

    def __init__(self):
        super().__init__(self)

    @classmethod
    def apply_random_unsharpmask(cls, img_src):

        value = cls.rng.integers(low=0, high=4, size=1)[0]

        img_sharp = cls.apply_unsharpmask(img_src, value)
        
        return img_sharp


    @classmethod
    def add_random_point_noise(cls, img):

        strength = cls.rng.uniform(0, 0.1, 1)
        possibility = cls.rng.uniform(0.4, 0.6, 1)

        img = cls.add_point_noise(img, strength, possibility)

        return img


    @classmethod
    def add_random_bokeh(cls, img):

        scale1, scale2, scale3 = cls.rng.integers(low=0, high=16, size=3)

        img = cls.add_bokeh(img, scale1, scale2, scale3)

        return img


    @classmethod
    def add_random_marker_line(cls, img):
        
        h, w = img.shape[:2]
        num_iter = cls.rng.integers(low=0, high=3, size=1)[0]

        for i in range(num_iter):
            vertical = np.random.choice((0, 1))

            if vertical:
                x_coods = cls.rng.integers(low=0, high=w, size=1)[0]
                p1 = (x_coods, 0)
                p2 = (x_coods, h)

            else:
                y_coods = cls.rng.integers(low=0, high=h, size=1)[0]
                p1 = (0, y_coods)
                p2 = (w, y_coods)

            thickness = cls.rng.integers(low=1, high=10)
            img = cls.draw_marker_line(img, p1, p2, thickness)

        return img


    @classmethod
    def make_periphery_darker_randomly(cls, img):

        strength = cls.rng.uniform(0, 1.5)
        img = cls.make_periphery_darker(img, strength)

        return img


############################################################################


########################## exif information ################################

class ExifReader():

    @staticmethod
    def read_exif(img_path: str, read_orientation: bool=True, read_dpi: bool=True) -> Tuple[int, int]:

        with open(img_path, 'rb') as f:

            tags = exifread.process_file(f, details=False)
            
            if read_orientation and "Image Orientation" in tags.keys():
                orientation = tags["Image Orientation"].values[0]

            else:
                orientation = 1

            if read_dpi and "Image XResolution" in tags.keys():
                dpi = int(str(tags["Image XResolution"].values[0]))

            else:
                dpi = -1

        return orientation, dpi


    @classmethod
    def restore_original_orientation(cls, img_path, img=None):

        convert_image = {
            1: lambda img: img,
            2: lambda img: cv2.flip(img, 1),                              # 左右反転
            3: lambda img: cv2.rotate(img, cv2.ROTATE_180),                                   # 180度回転
            4: lambda img: cv2.flip(img, 0),                              # 上下反転
            5: lambda img: cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_90_COUNTERCLOCKWISE),    # 左右反転＆反時計回りに90度回転
            6: lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),                                   # 反時計回りに270度回転
            7: lambda img: cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_90_CLOCKWISE), # 左右反転＆反時計回りに270度回転
            8: lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),                                    # 反時計回りに90度回転
        }


        orientation, _ = cls.read_exif(img_path, read_dpi=False)

        if img is None:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        img = convert_image[orientation](img)

        return img


    @classmethod
    def rescale(cls, img_path:str, target_dpi: int, img=None):

        _, dpi = cls.read_exif(img_path, read_orientation=False)

        if img is None:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if (dpi > 0 and target_dpi > 0):
            img = cv2.resize(img, None, fx=target_dpi/dpi, fy=target_dpi/dpi, interpolation=cv2.INTER_CUBIC)
            
        return img

############################################################################

################### Image Converter ########################################

class ImageConverter():

    def __init__(self):
        pass

    @staticmethod
    def cvtColor2BGR(img, color_space):
            
        """
        RGB以外の色空間で保存されているものを変換する。
        変換はopencvに準拠するが、
        各色空間の値の取る範囲は媒体により異なるため、
        opencvの方式に合わせる必要がある。

        画像はRGBで保存しておくことを勧める。
        """

        if color_space=='RGB':
            """
            opencvではRGB画像をBGRで読みこむのでそのまま
            """
            return img


        elif color_space=='BGR':
            """
            上記より、BGRで保存されているとRGBで読みこまれてしまう
            """
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)



        elif color_space=='HSV':
            """
            h(0~360)：色相（色彩を表すパラメータ）、h=0(赤),120(緑),225(青)
            Rが最大値の場合 色相 H = 60 × ((G – B) ÷ (R – G,Bの最小値))
            Gが最大値の場合 色相 H = 60 × ((B – R) ÷ (G – R,Bの最小値)) +120
            Bが最大値の場合 色相 H = 60 × ((R – G) ÷ (B – R,Gの最小値)) +240
            3つとも同じ値の場合 色相 H = 0

            s(0~255)：彩度＝(R,G,Bの最大値 – R,G,Bの最小値) ÷ R,G,Bの最大値
            v(0~255)：明度 V = R,G,Bのうち最大値が反映される
            """
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        elif color_space=='HLS':

            """
            h(0~360)：色相（色彩を表すパラメータ）、h=0(赤),120(緑),225(青)
            Rが最大値の場合 色相 H = 60 × ((G – B) ÷ (R – G,Bの最小値))
            Gが最大値の場合 色相 H = 60 × ((B – R) ÷ (G – R,Bの最小値)) +120
            Bが最大値の場合 色相 H = 60 × ((R – G) ÷ (B – R,Gの最小値)) +240
            3つとも同じ値の場合 色相 H = 0

            l(0~255)：輝度
            (R,G,Bの最大値 +R,G,Bの最小値) ÷ 2

            s(0~255)：彩度
            収束値 CNT = (RGBの最大値 +R,G,Bの最小値) ÷ 2
            収束値 CNTが127以下の場合 彩度 S = (R,G,Bの最大値 -R,G,Bの最小値) ÷ (RGBの最大値 +R,G,Bの最小値)
            収束値 CNTが128以上の場合 彩度 S = (R,G,Bの最大値 -R,G,Bの最小値) ÷ (510 -RGBの最大値 -R,G,Bの最小値)
            """
            return cv2.cvtColor(img, cv2.COLOR_HLS2BGR)

        elif color_space=='LAB':
            """
            X=XXn             ※Xn=0.950456
            Z=ZZn             ※Zn=1.088754
            Y>0.08856の場合
            L= 116Y1/3−16
            f(t) = t1/3

            Y≤0.08856の場合
            L= 903.3Y
            f(t) = 7.787t + 16116

            a=500(f(X)-f(Y)) + delta
            b=200(f(Y)-f(Z)) + delta

            8bitイメージ： delta=128
            16bitイメージ： サポートなし
            浮動小数点型のイメージ： delta=0
            """
            return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

        elif color_space=='YCrCb':
            """
            Y = 0.299R+0.587G+0.114B
            Cr =0.713 (R-Y) + delta
            Cb =0.564 (B-Y) + delta
            8bitイメージ： delta=128
            16bitイメージ： delta=32768
            浮動小数点型のイメージ： delta=0.5
            """
            return cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

        elif color_space=='XYZ':
            """
            X = 0.4124R + 0.3576G + 0.1805B
            Y = 0.2126R + 0.7152G + 0.0722B
            Z = 0.0193R + 0.1192G + 0.9505B
            """
            return cv2.cvtColor(img, cv2.COLOR_XYZ2BGR)