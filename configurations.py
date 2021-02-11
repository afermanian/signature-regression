
configs = {
    'estimator_convergence':
        {'d': [2],
         'regressor': ['signature'],
         'X_type': ['independent'],
         'Y_type': ['sig'],
         'ntrain': [10, 50, 100, 500],
         'Kpen': [20]
        },

    'test':
        {'d': [3],
         'regressor': ['signature', 'fPCA', 'bspline', 'fourier'],
         'X_type': ['independent'],
         'Y_type': ['mean']
         },

    'dim_study':
        {'d': [1, 3, 5, 7, 9, 11],
         'regressor': ['signature', 'fPCA', 'bspline', 'fourier'],
         'X_type': ['dependent', 'independent'],
         'Y_type': ['max', 'mean']
         },

    'dim_study_gp':
        {'d': [1, 3, 5, 7, 9, 11],
         'regressor': ['signature', 'fPCA', 'bspline', 'fourier'],
         'X_type': ['gp_independent'],
         },

    'weather_estimation':
        {'regressor': ['signature', 'fPCA', 'bspline', 'fourier'],
         'X_type': ['weather'],
         'selection_method': ['estimation'],
         'Kpen': [0.02]
         },

    'electricity_loads':
        {'regressor': ['signature', 'fPCA', 'bspline', 'fourier'],
         'X_type': ['electricity_loads'],
         'selection_method': ['cv'],
         'nclients': [1, 5, 10, 15, 20],
         },
}

