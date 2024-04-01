import numpy as np

def term_generate_1():

    long_term = [168]
    mid_term = [48, 72, 96, 120]
    short_term = [6, 12, 18, 24]

    return  long_term, mid_term, short_term

def term_generate_2():

    long_term = [336]
    mid_term = [48, 72, 96, 120]
    short_term = [6, 12, 18, 24]

    return  long_term, mid_term, short_term


def term_generate_3():
    long_term = [504]
    mid_term = [48, 72, 96, 120]
    short_term = [6, 12, 18, 24]

    return long_term, mid_term, short_term


def term_generate_4():
    long_term = [607]
    mid_term = [48, 72, 96, 120]
    short_term = [6, 12, 18, 24]

    return long_term, mid_term, short_term

if __name__ == '__main__':
    long_term, mid_term, short_term = term_generate_1()
    long_term, mid_term, short_term = term_generate_2()
    long_term, mid_term, short_term = term_generate_3()
    long_term, mid_term, short_term = term_generate_4()
    print(long_term)
    print(mid_term)
    print(short_term)