import numpy as np

configs = {
    'dim_study':
        {'d': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         'regressor': ['signature', 'spline'],
         },

    'dim_study_max':
        {'d': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         'regressor': ['signature', 'spline'],
         'Y_type': ['max']
         },

    'dim_study_max_X_dependent':
        {'d': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         'regressor': ['signature', 'spline'],
         'X_type': ['dependent'],
         'Y_type': ['max']
         },

    'test':
        {'d': [2],
         'regressor': ['signature', 'spline'],
         'X_type': ['independent'],
         'Y_type': ['max'],
         'npoints': [1000],
         'noise_X_std': [5]
         },
}

