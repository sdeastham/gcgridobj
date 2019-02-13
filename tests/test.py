from gcgridobj import gc_vertical

def test_g5_size():
    '''
    Make sure that the GEOS vertical grid is present and correctly sized.
    '''

    
    assert gc_vertical.G5_AP.size == 73, 'Bad GEOS vertical grid data'
