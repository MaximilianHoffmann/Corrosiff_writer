import siffpy

from local_consts import small_path, large_path, rclone_path

TEST_RCLONE = False

def test_siffio_small_alone():
    sr = siffpy.SiffReader(backend = 'siffreadermodule')

    sr.siffio.open(small_path)

def test_siffio_large_alone():
    sr = siffpy.SiffReader(backend = 'siffreadermodule')

    sr.siffio.open(large_path)

def test_open_small_siff():
    """Stupid test function"""
    siffpy.SiffReader(small_path, backend = 'siffreadermodule')

def test_open_large_siff():
    siffpy.SiffReader(large_path, backend = 'siffreadermodule')

def test_open_rclone_siff():
    siffpy.SiffReader(rclone_path, backend = 'siffreadermodule')

if __name__ == '__main__':
    import timeit
    
    if TEST_RCLONE:
        print(
            "Rclone file:\n",
            timeit.timeit("test()",
                setup="from __main__ import test_open_rclone_siff as test",
                number = 10,
            )/10 , "sec per iter"
        )

    print(
        "Siffio alone, small file:\n",
        timeit.timeit("test()",
            setup="from __main__ import test_siffio_small_alone as test",
            number = 10,
        )/10 , "sec per iter"
    )

    print(
        "Siffio alone, large file:\n",
        timeit.timeit("test()",
            setup="from __main__ import test_siffio_large_alone as test",
            number = 10,
        )/10 , "sec per iter"
    )
    
    print(
        "SiffReader, small file:\n",
        timeit.timeit("test()",
            setup="from __main__ import test_open_small_siff as test",
            number = 10,
        )/10 , "sec per iter"
    )
    print(
        "SiffReader, large file:\n",
        timeit.timeit("test()",
            setup="from __main__ import test_open_large_siff as test",
            number = 10,
        )/10, "sec per iter"
    )
