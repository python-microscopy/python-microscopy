import numpy as np

def c_mul(a, b):
    return eval(hex((long(a) * b) & 0xFFFFFFFFL)[:-1])


def hashString32(s):
    if not s:
        return 0 # empty
    value = np.array([ord(s[0]) << 7], 'int32')
    for char in s:
        value = (value*1000003) ^ ord(char)
    value = value ^ len(s)
    value = int(value)
    if value == -1:
        value = -2
    return value