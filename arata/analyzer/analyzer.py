# coding: utf-8

import cv2

from arata.common.constants import AnalysisConfigJsonConst, DetailedErrorMessages, ImageExtensionConst, TrainingConfigJsonConst, ColumnHeaders
from arata.common.enum import ErrorType
from arata.common.error_handler import ErrorType
from arata.common.image_tools import make_diff_images
from arata.common.path_utils import get_file_names, get_image_paths
from arata.common.wrapper_collection import watch_error
from arata.common.configurations import AnalysisConfig, TrainingConfig
from arata.nn.param_setter import create_predictor
from arata.nn.predictor import Predictor

from .calculator import dt, imgcalc
from .tracer import preprocessor, root_painter, postprocessor

class Analyzer():

    r"""
    Analyzes roots in images in the specified folder.
    """

    def __init__(self) -> None:

        super().__init__()

        self.analysis_conf = AnalysisConfig()
        self.analysis_conf.load()

        self.training_conf = TrainingConfig()
        self.training_conf.load()

        # Creates folders to save results.
        self.analysis_conf.make_folders()

        # Calculation target region of all images
        self.__boundary = ((0, 0), (0, 0))
        
        # Few image flag
        self.__image_number_flag = True


    @watch_error(
        DetailedErrorMessages.get_critical_error_message("preprocessing"), 
        ErrorType.CRITICAL_ERROR)
    def preprocesses(self) -> None:
        
        r"""
        Preprocesses images in the source folder.
        """

        src_paths = get_image_paths(
            self.analysis_conf.config[AnalysisConfigJsonConst.SOURCE_LOCATION],
            self.analysis_conf.config[AnalysisConfigJsonConst.IMAGE_FORMAT])
            

        # If images are few, starts analysis with few image mode.
        if len(src_paths) <= 2:
            self.__image_number_flag = False

        # Preprocess
        preprocess = preprocessor.ImagePreprocessor(
)

        # 位置合わせは二枚より多い時でないと機能しない
        condition = self.__image_number_flag & \
            self.analysis_conf.config[AnalysisConfigJsonConst.POSITION_FIXING]
            
        # 位相相関による位置合わせ
        preprocess.preprocess(src_paths, 
            self.analysis_conf.result_main_path,
            self.analysis_conf.config[AnalysisConfigJsonConst.DPI], 
            self.analysis_conf.config[AnalysisConfigJsonConst.AUTO_DPI],
            self.analysis_conf.config[AnalysisConfigJsonConst.IMAGE_FORMAT], 
            self.analysis_conf.config[AnalysisConfigJsonConst.COLOR_SPACE],
            condition)

        # 対象画像全ての画面内に写っている部分だけ計算するか
        if condition & self.analysis_conf.config[AnalysisConfigJsonConst.INSCREEN_CALCULATION_MODE]:
            self.__boundary = preprocess.find_position_constant_area()


    @watch_error(
        DetailedErrorMessages.get_critical_error_message("tracing roots"), 
        ErrorType.CRITICAL_ERROR)
    def paints(self) -> None:

        r"""
        Extracts roots in images and output result images.
        """

        src_paths = get_image_paths(
            self.analysis_conf.result_main_path, self.analysis_conf.config[AnalysisConfigJsonConst.IMAGE_FORMAT])
            
        predictor = create_predictor(self.training_conf.config, self.training_conf.class_info)
            
        predictor.initialize_network(self.analysis_conf.config[AnalysisConfigJsonConst.MODEL_PATH])

        painter = root_painter.RootPainter()

        painter.run(            
            src_paths, 
            self.analysis_conf.result_probability_path,
            predictor,
            self.analysis_conf.config[AnalysisConfigJsonConst.CROP_SIZE],
            self.analysis_conf.config[AnalysisConfigJsonConst.DPI],
            self.analysis_conf.config[AnalysisConfigJsonConst.MULTISCALE_PREDICTION],
            self.__boundary,
            self.training_conf.class_info.class_dict
            )


    @watch_error(
        DetailedErrorMessages.get_critical_error_message("postprocessing"), 
        ErrorType.CRITICAL_ERROR
        )
    def postprocesses(self) -> None:
        
        r"""
        Postprocesses root extraction results.
        """

        src_paths = get_image_paths(
            self.analysis_conf.result_probability_path,
            ImageExtensionConst.INTERNAL_IMAGE_EXTENSIONS)

        postprocess = postprocessor.Postprocessor(src_paths, self.analysis_conf.result_postprocess_path)
        postprocess.run()


    @watch_error(
        DetailedErrorMessages.get_critical_error_message("creating difference images"), 
        ErrorType.CRITICAL_ERROR
        )
    def creates_diff_images(self) -> None:
        
        r"""
        Creates difference images.
        """

        # If image number < 2, cannot make difference. 
        if not (self.__image_number_flag and self.analysis_conf.config[AnalysisConfigJsonConst.DIFFERENCE_CALCULATION]):
            return False
        
        src_paths = get_image_paths(self.analysis_conf.result_postprocess_path, ImageExtensionConst.INTERNAL_IMAGE_EXTENSIONS)

        # Makes diffrence images
        make_diff_images(src_paths, self.analysis_conf.result_increment_path, self.analysis_conf.result_decrement_path)


    @watch_error(
        DetailedErrorMessages.get_critical_error_message("calculating results"), 
        ErrorType.CRITICAL_ERROR)
    def calculates(self) -> None:

        r"""
        Calculates values from root extraction results.
        """

        if not self.analysis_conf.config[AnalysisConfigJsonConst.ENABLE_CALCULATION]:
            return False

        src_paths = get_image_paths(self.analysis_conf.result_postprocess_path, ImageExtensionConst.INTERNAL_IMAGE_EXTENSIONS)
        img_names = get_file_names(src_paths)
        
        table = dt.DataTable()
        table.create_table({ 'Name': img_names })

        for obj_name, class_info in self.training_conf.class_info.class_dict.items():

            if obj_name == 'root':

                current_areas = [imgcalc.calc_area(cv2.imread(src_path, cv2.IMREAD_COLOR), class_info.color) 
                            for src_path in src_paths]      

                table.add_numeric_data_column(
                    ColumnHeaders.create_header(ColumnHeaders.ROOT, ColumnHeaders.CURRENT, ColumnHeaders.AREA), 
                    current_areas)

                current_lengths = [imgcalc.calc_skeletonlen(cv2.imread(src_path, cv2.IMREAD_COLOR), class_info.color) 
                            for src_path in src_paths]

                table.add_numeric_data_column(
                    ColumnHeaders.create_header(ColumnHeaders.ROOT, ColumnHeaders.CURRENT, ColumnHeaders.LENGTH),                     
                    current_lengths)


                if self.analysis_conf.config[AnalysisConfigJsonConst.DIFFERENCE_CALCULATION]:

                    increment_src_paths = get_image_paths(self.analysis_conf.result_increment_path, ImageExtensionConst.INTERNAL_IMAGE_EXTENSIONS)
                    increment_areas = [
                        imgcalc.calc_area(cv2.imread(src_path, cv2.IMREAD_COLOR), class_info.color) 
                        for src_path in increment_src_paths]   

                    table.add_numeric_data_column(
                        ColumnHeaders.create_header(ColumnHeaders.ROOT, ColumnHeaders.INCREMENT, ColumnHeaders.AREA), 
                        increment_areas)

                    increment_lengths = [
                        imgcalc.calc_skeletonlen(cv2.imread(src_path, cv2.IMREAD_COLOR), class_info.color) 
                        for src_path in increment_src_paths]   

                    table.add_numeric_data_column(
                        ColumnHeaders.create_header(ColumnHeaders.ROOT, ColumnHeaders.INCREMENT, ColumnHeaders.LENGTH), 
                        increment_lengths)

                    decrement_src_paths = get_image_paths(self.analysis_conf.result_decrement_path, ImageExtensionConst.INTERNAL_IMAGE_EXTENSIONS)
                    decrement_areas = [
                        imgcalc.calc_area(cv2.imread(src_path, cv2.IMREAD_COLOR), class_info.color) 
                        for src_path in decrement_src_paths]   

                    table.add_numeric_data_column(
                        ColumnHeaders.create_header(ColumnHeaders.ROOT, ColumnHeaders.DECREMENT, ColumnHeaders.AREA), 
                        decrement_areas)

                    decrement_lengths = [
                        imgcalc.calc_skeletonlen(cv2.imread(src_path, cv2.IMREAD_COLOR), class_info.color) 
                        for src_path in decrement_src_paths]   

                    table.add_numeric_data_column(
                        ColumnHeaders.create_header(ColumnHeaders.ROOT, ColumnHeaders.DECREMENT, ColumnHeaders.LENGTH), 
                        decrement_lengths)

        table.write_csv(self.analysis_conf.result_table_path)


    def outputs_analysis_log(self) -> None:
        
        r"""
        Outputs an analysis log.
        """

        if (self.analysis_conf.config[AnalysisConfigJsonConst.AUTO_LOG]):
            lines = []
            log_path = self.analysis_conf.result_log_path

            for key, value in self.analysis_conf.config.items():
                value = str(value)
                lines.append("{0}: {1}\n".format(key, value))

            with open(log_path, 'w') as f:
                f.writelines(lines)


def analyze():

    r"""
    Analyzes images in the specified folder and save results in bellow folders.
    
    Directory Configuration...

    root folder (= selected by user)
    |---sub folder (= image source folder name)
        |------ save folder (= the date you started this analysis)
                |-------results
                        |-----------probability (= store results which show class probability)
                        |-----------postprocess (= postprocessed results stored in the "probability" folder)
                        |-----------difference (= difference of two images)
                                    |--- increment
                                    |--- decrement
                        |-----------calcuration results
    
    """

    # Initialize Analyzer
    AZ = Analyzer()

    # Preprocess
    AZ.preprocesses()

    # Detect roots
    AZ.paints()    

    # Cancel noises in model outputs
    AZ.postprocesses()

    # Create images showing production & gone 
    AZ.creates_diff_images()

    # Calcurate parameters
    AZ.calculates()

    # Log analysis information
    AZ.outputs_analysis_log()