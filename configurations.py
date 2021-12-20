
configs = {
    'estimator_convergence':
        {'d': [2],
         'regressor': ['signature'],
         'X_type': ['smooth'],
         'Y_type': ['sig'],
         'ntrain': [10, 50, 100, 500],
         'Kpen': [20]
        },

    'test':
        {'d': [3],
         'regressor': ['signature', 'fPCA', 'bspline', 'fourier'],
         'X_type': ['smooth_independent'],
         'Y_type': ['mean']
         },

    'air_quality':
        {'regressor': ['signature', 'fPCA', 'bspline', 'fourier'],
         'X_type': ['air_quality'],
         'selection_method': ['estimation'],
         'Kpen': [10 ** (-2)],
         'univariate_air_quality': [True, False]
         },

    'dim_study':
        {'d': [1, 3, 5, 7, 9, 11],
         'regressor': ['signature', 'fPCA', 'bspline', 'fourier'],
         'X_type': ['smooth'],
         'Y_type': ['mean']
         },

    'dim_study_gp':
        {'d': [1, 3, 5, 7, 9, 11],
         'regressor': ['signature', 'fPCA', 'bspline', 'fourier'],
         'X_type': ['gp'],
         },
}

