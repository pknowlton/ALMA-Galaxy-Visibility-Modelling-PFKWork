import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('dynesty_version_fixed/input_dir'))

import prior_tform as pt_orig
import prior_tform_streamline as pt_stream

def test_streamline_parity():
    np.random.seed(42)
    test_u = np.random.uniform(0.001, 0.999, size=(50, 20))
    test_cases = [
        ('twod_gaussring_ptform', 7),
        ('twod_gauss1blob_ptform', 11),
        ('twod_gauss1blob_2peak_ptform', 13),
        ('twod_gauss1blob_2peak_dp_ptform', 15),
        ('twod_gauss3blob_ptform', 19),
        ('simgauss_ptform', 7),
    ]

    for func_name, ndim in test_cases:
        func_orig = getattr(pt_orig, func_name)
        func_stream = getattr(pt_stream, func_name)
        for i in range(len(test_u)):
            u_vec = test_u[i, :ndim].copy()
            res_orig = func_orig(u_vec.copy())
            res_stream = func_stream(u_vec.copy())
            np.testing.assert_allclose(
                res_stream, res_orig, rtol=1e-12, atol=1e-12,
                err_msg=f"Parity mismatch for {func_name} at sample {i}"
            )
        print(f"PASS: Parity verified for {func_name} ({ndim} params).")

if __name__ == '__main__':
    test_streamline_parity()
    print("\nALL PARITY TESTS PASSED!")
