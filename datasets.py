import numpy as np

# Configurations for different datasets
CONFIG = {
    'cityscapes': {
        'classes': 19,
        'weights_file': 'data/pretrained_dilation_cityscapes.pickle',
        'input_shape': (1396, 1396, 3),
        'output_shape': (1024, 1024, 19),
        'mean_pixel': (72.39, 82.91, 73.16),
        'palette': np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [255, 255, 255],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]], dtype='uint8'),
        'zoom': 1,
        'conv_margin': 186
    },
    'camvid': {
        'classes': 11,
        'weights_file': 'data/pretrained_dilation_camvid.pickle',
        'input_shape': (900, 1100, 3),
        'output_shape': (66, 91, 11),
        'mean_pixel': (110.70, 108.77, 105.41),
        'palette': np.array([[128, 0, 0],
                             [128, 128, 0],
                             [128, 128, 128],
                             [64, 0, 128],
                             [192, 128, 128],
                             [128, 64, 128],
                             [64, 64, 0],
                             [64, 64, 128],
                             [192, 192, 128],
                             [0, 0, 192],
                             [0, 128, 192]], dtype='uint8'),
        'zoom': 8,
        'conv_margin': 186
    }
}
