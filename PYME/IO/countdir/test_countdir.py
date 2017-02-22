from nose.tools import assert_equals

def test_dirsize():
    import countdir
    import os

    assert_equals(countdir.dirsize(os.curdir), len(os.listdir(os.curdir)) + 2) #os.listdir does not count '.' and '..'