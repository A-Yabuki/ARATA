# coding: utf-8


class DetailedErrorMessages():
    
    """Error messages"""

    MEMORY_READ_ERROR = "Failed to read a memory mapped file."
    MEMORY_WRITE_ERROR = "Failed to write message into a memory mapped file."

    DECODE_ERROR = "Memory mapped file's value is invalid."
    ENCODE_ERROR = "Cannot encode string to byte array."

    MESSAGE_SEND_ERROR = "Failed to send messages to GUI."

    __CRITICAL_ERROR = "Critical error is occured in {0}"

    @classmethod
    def get_critical_error_message(cls, process_name: str) -> str:

        return str.format(cls.__CRITICAL_ERROR, process_name)


class DisplayErrorMessages():

    """Error messages to display"""

    COMMUNICATION_FAILED = \
        """Communication between GUI and background processes were failed."""
    
    MEMORY_OVERFLOW = \
        """Memory overflow was occured. It may be resolved by following ways...
        1. Reduce the size of window size
        2. Reduce the size of batch size
        3. Reduce the number of the DeepLabv3+'s middle layer
        4. Reduce the scale of the DeepLabv3+'s middle layer
        """

    INVALID_INPUT_VALUE = \
        """Input value [%s: %s] is invalid."""

    OUTPUT_FAILED = \
        """Failed to output a file. [file type:%s, path:%s]"""

    UNEXPECTED_ERROR = \
        """Unexpected error occured. Please report the situation where this error occured to the developper."""


class MemoryMappedFileName():
    
    r"""
    Defines memory mapped file names.
    """

    INTERRUPTER = "interrupter"
    REPORTER = "reporter"


class ColumnHeaders():
    
    r"""
    Defines table column heades of calcuration results.
    """

    NAME = "Name"
    ROOT = "Root"
    HYPHA = "Hypha"

    CURRENT = "Current"
    INCREMENT = "Increment"
    DECREMENT = "Decrement"
    
    AREA = "Area (mm\u00b2)"
    LENGTH = "Length (mm)"
    HEADER_FRM = "{0} {1} : {2}" 

    @classmethod
    def create_header(cls, obj_name: str, amount_kind: str, calc_target: str):
        return str.format(cls.HEADER_FRM, obj_name, amount_kind, calc_target)
        


class ResourcePathConst():

    r"""
    Defines relative paths of resource files.
    """
    
    RESOURCES_PATH = './resources'
    BLUR_MASK = RESOURCES_PATH + '/blurmask.png'
    ANALYSIS_CONFIG_PATH = RESOURCES_PATH + '/analysis_configuration.json'
    TRAINING_CONFIG_PATH = RESOURCES_PATH + '/training_configuration.json'
    COMMON_CONFIG_PATH = RESOURCES_PATH + '/constants.json'
    LOG_OUTPUT_PATH = './log'
    APP_LOG = 'App.log'
    ERROR_LOG = 'Error.log'


class ImageExtensionConst():

    r"""
    Defines available image extensions
    """

    JPG_EXTENSIONS = ['jpg', 'jpeg']#, 'JPG', 'JPEG']
    TIF_EXTENSIONS = ['tif', 'tiff']#, 'TIF', 'TIFF']
    PNG_EXTENSIONS = ['png']#, 'PNG']
    BMP_EXTENSIONS = ['bmp']  #, 'BMP']
    
    INTERNAL_IMAGE_EXTENSIONS = PNG_EXTENSIONS

    IMAGE_EXTENSIONS = \
        [
            JPG_EXTENSIONS,
            TIF_EXTENSIONS,
            PNG_EXTENSIONS,
            BMP_EXTENSIONS 
        ]


class ClassColorJsonConst():

    r"""
    Defines json keys handling colors.
    """

    White = "White"
    Yellow = "Yellow"
    Pink = "Pink"
    Aqua = "Aqua"
    Red = "Red"
    Green = "Green"
    Blue = "Blue"
    Black = "Black"

    KEY_LIST = [ White, Yellow, Pink, Aqua, Red, Green, Blue, Black ]


class LossFuncJsonConst():

    r"""
    Defines json keys about loss function settings.
    """
    
    CrossEntropy = "Cross Entropy"
    WeightedCE = "Weighted Cross Entropy"
    FocalCE = "Focal Cross Entropy"
    DiceLoss = "Generalized Dice Similarity"

    KEY_LIST = [ CrossEntropy, WeightedCE, FocalCE, DiceLoss ]


class OptimizerJsonConst():

    r"""
    Defines json keys of optimizer settings.
    """

    SGD  = "Stochastic Gradient Descent"
    NesterovAG = "Nesterov Accelerated Gradient"
    AdaBound = "AdaBound"

    KEY_LIST = [ SGD, NesterovAG, AdaBound ]


class MiddleLayerJsonConst():
    
    r"""
    Defines json keys of middle layer settings.
    """

    Eight = "Eight"
    Sixteen = "Sixteen"
    ThirtyTwo = "ThirtyTwo"

    KEY_LIST = [ Eight, Sixteen, ThirtyTwo ]


class ActivationFuncJsonConst():

    r"""
    Defines json keys of activateion fuction settings.
    """

    ReLU = "Rectified Linear Unit"
    LeakyReLU = "Learky ReLU"
    HardSwish = "Hard Swish"
    TanhExp = "Exponential Hypabolic Tangent"

    KEY_LIST = [ ReLU, LeakyReLU, HardSwish, TanhExp ]


class NormalizationMethodJsonConst():
    
    r"""
    Defines json keys of normalization method settings.
    """

    Batch = "Batch Normalization"
    Layer = "Layer Normalization"
    Instance = "Instance Normalization"

    KEY_LIST = [ Batch, Layer, Instance ]


class JsonItem():
    
    r"""
    Defines json keys of basal items.
    """

    IS_BOOL = "isBool"
    IS_NUMERIC = "isNumeric"
    VALUE = "value"


class TrainingConfigJsonConst():

    r"""
    Defines json keys of training settings.
    """

    IMAGE_SOURCE = 'ImageSource'
    LABEL_SOURCE = 'LabelSource'
    OUTPUT_DESTINATION = 'OutputDestination'


    MIDDLE_LAYER_NUM = 'MiddleLayerNum'
    MIDDLE_LAYER_SCALE = 'MiddleLayerScale'
    NORMALIZER = 'Normalizer'
    ACTIVATOR = 'Activator'
    
    
    ADD_FLIP_AND_ROTATION = 'AddFlipAndRotation'
    ADD_RANDOM_NOISE = 'AddRandomNoise'
    APPLY_CLAHE = 'ApplyClahe'
    CUT_MIX = 'CutMix'
    OVERSAMPLING = 'OverSampling'


    SET_CLASSINFO_AUTO ='SetClassInfoAutomatically'
    CLASS1_COLOR = 'Class1Color'
    CLASS1_IGNORE = 'Class1Ignore'
    CLASS1_WEIGHT = 'Class1Weight'
    CLASS2_COLOR = 'Class2Color'
    CLASS2_IGNORE = 'Class2Ignore'
    CLASS2_WEIGHT = 'Class2Weight'
    CLASS3_COLOR = 'Class3Color'
    CLASS3_IGNORE = 'Class3Ignore'
    CLASS3_WEIGHT = 'Class3Weight'
    CLASS4_COLOR = 'Class4Color'
    CLASS4_IGNORE = 'Class4Ignore'
    CLASS4_WEIGHT = 'Class4Weight'
    CLASS5_COLOR = 'Class5Color'
    CLASS5_IGNORE = 'Class5Ignore'
    CLASS5_WEIGHT = 'Class5Weight'


    INITIAL_MODEL_PATH = 'InitialModelPath'
    EPOCH_NUM = 'EpochNum'
    BATCH_SIZE= 'BatchSize'
    LOSS_FUNC = 'LossFunction'
    OPTIMIZER = 'Optimizer'   
    INITIAL_LEARNING_RATE = 'InitialLearningRate'
    FINAL_LEARNING_RATE = 'FinalLearningRate'
    WEIGHT_DECAY = 'WeightDecay'
    VALIDATION_RATIO = 'ValidationRatio'
    SCHEDULAR_STEP_RATE = 'SchedularStepRate'
    SCHEDULAR_STEP_SIZE = 'SchedularStepSize'


    OUTPUT_GRAPH_LOG = 'OutputGraphAndLog'
    OUTPUT_GRAPH_INTERVAL = 'OutputGraphAndLogInterval'
    OUTPUT_IMAGE = 'OutputImage'
    OUTPUT_IMAGE_INTERVAL = 'OutputImageInterval'
    TAKE_SNAPSHOT = 'TakeSnapshot'
    TAKE_SNAPSHOT_INTERVAL = 'TakeSnapshotInterval'


    KEY_LIST = \
        [
            IMAGE_SOURCE, LABEL_SOURCE, OUTPUT_DESTINATION,
            SET_CLASSINFO_AUTO, CLASS1_COLOR, CLASS1_IGNORE, CLASS1_WEIGHT,
            CLASS2_COLOR, CLASS2_IGNORE, CLASS2_WEIGHT,
            CLASS3_COLOR, CLASS3_IGNORE, CLASS3_WEIGHT,
            CLASS4_COLOR, CLASS4_IGNORE, CLASS4_WEIGHT,
            CLASS5_COLOR, CLASS5_IGNORE, CLASS5_WEIGHT,
            ADD_FLIP_AND_ROTATION, ADD_RANDOM_NOISE, APPLY_CLAHE, CUT_MIX, OVERSAMPLING,
            MIDDLE_LAYER_NUM, MIDDLE_LAYER_SCALE, NORMALIZER, ACTIVATOR,
            INITIAL_MODEL_PATH, BATCH_SIZE, EPOCH_NUM,
            LOSS_FUNC, OPTIMIZER, INITIAL_LEARNING_RATE, FINAL_LEARNING_RATE,
            WEIGHT_DECAY, VALIDATION_RATIO, SCHEDULAR_STEP_RATE, SCHEDULAR_STEP_SIZE,
            OUTPUT_GRAPH_LOG, OUTPUT_GRAPH_INTERVAL,
            OUTPUT_IMAGE, OUTPUT_IMAGE_INTERVAL,
            TAKE_SNAPSHOT, TAKE_SNAPSHOT_INTERVAL
        ]


class AnalysisConfigJsonConst():
    
    r"""
    Defines json keys of analysis settings
    """

    MODEL_PATH = 'ModelPath'
    TRAINING_SETTINGS_PATH = 'NetArcPath'
    
    AUTO_LOG = 'AutoLog'
    MULTISCALE_PREDICTION = 'MultiScalePrediction'
    DPI = 'Resolution'
    AUTO_DPI = 'IndefiniteResolution'
    CROP_SIZE = 'WindowSize'
    COLOR_SPACE = 'ColorSpace'
    IMAGE_FORMAT = 'ImageFormat'
    SOURCE_LOCATION = 'SourcePath'
    SAVE_LOCATION = 'SavePath'
    POSITION_FIXING = 'PositionFixing'
    PROBABILITY_ENHANCEMENT = 'ProbabilityEnhancement'
    REAL_TIMESCALE = 'RealTimeScale'
    TIME_RANGE = 'TimeRange'
    ENABLE_CALCULATION = 'Calculation'
    HYPHAE_CALCULATION = 'HyphaeCalculation'
    DIFFERENCE_CALCULATION = 'DifferenceCalculation'
    INSCREEN_CALCULATION_MODE = 'InScreenCalculationMode'
    
    KEY_LIST = \
        [
            MODEL_PATH, TRAINING_SETTINGS_PATH,
            AUTO_LOG, MULTISCALE_PREDICTION, DPI, AUTO_DPI, CROP_SIZE, COLOR_SPACE, IMAGE_FORMAT,
            SOURCE_LOCATION, SAVE_LOCATION, POSITION_FIXING, PROBABILITY_ENHANCEMENT,
            REAL_TIMESCALE, TIME_RANGE, ENABLE_CALCULATION, HYPHAE_CALCULATION,
            DIFFERENCE_CALCULATION, INSCREEN_CALCULATION_MODE
        ]