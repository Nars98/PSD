import seaborn as sns

TEST_NAMES = [ 'RFSD', 'RFSD(alpha)',
              'IMQ KSD', 'Gauss KSD', 'Gauss FSSD-opt', 'PSD r1', 'PSD r2','PSD r3','PSD r4',
                ]

ORDERED_TEST_NAMES = ['RFSD',
                      'RFSD(alpha)', 'RFSD(RBM)',
                      'Gauss KSD', 'IMQ KSD', 'Gauss FSSD-opt', 'PSD r1', 'PSD r2', 'PSD r3', 'PSD r4',
                       ]

def test_name_colors_dict():
    return dict(zip(TEST_NAMES, sns.color_palette(n_colors=len(TEST_NAMES))))
