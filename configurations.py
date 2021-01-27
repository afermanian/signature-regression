
configs = {
    'dim_study':
        {'d': [1, 3, 5, 7, 9, 11],
         'regressor': ['signature', 'bspline', 'fourier'],
         'X_type': ['dependent', 'independent'],
         'Y_type': ['max', 'mean']
         },

    'dim_study_gp':
        {'d': [1, 3, 5, 7, 9, 11],
         'regressor': ['signature', 'bspline', 'fourier'],
         'X_type': ['gp_independent'],
         },

    'weather_estimation':
        {'regressor': ['signature', 'bspline', 'fourier'],
         'X_type': ['weather'],
         'selection_method': ['estimation'],
         'Kpen': [0.02]
         },

    'electricity_loads':
        {'regressor': ['signature', 'bspline', 'fourier'],
         'X_type': ['electricity_loads'],
         'selection_method': ['cv'],
         'nclients': [1, 5, 10, 15, 20],
         },
}

