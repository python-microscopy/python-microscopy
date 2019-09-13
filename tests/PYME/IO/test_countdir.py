
def test_dirsize():
    from PYME.IO import countdir
    import os

    assert countdir.dirsize(os.curdir) == len(os.listdir(os.curdir)) + 2  # os.listdir does not count '.' and '..'