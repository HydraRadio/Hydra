import numpy as np

def zernike(coeffs, x, y):
        """
        Zernike polynomials (up to degree 66) on the unit disc.
        This code was adapted from:
        https://gitlab.nrao.edu/pjaganna/zcpb/-/blob/master/zernikeAperture.py
        Parameters
        ----------
        coeffs : array_like
            Array of real coefficients of the Zernike polynomials, from 0..66.
        x, y : array_like
            Points on the unit disc.
        Returns
        -------
        zernike : array_like
            Values of the Zernike polynomial at the input x,y points.
        """
        # Coefficients
        assert len(coeffs) <= 66, "Max. number of coeffs is 66."
        c = np.zeros(66)
        c[: len(coeffs)] += coeffs

        # Precompute powers of x and y
        x2, x3, x4, x5, x6, x7, x8, x9, x10 = tuple(x ** idx for idx in range(2, 11))
        y2, y3, y4, y5, y6, y7, y8, y9, y10 = tuple(y ** idx for idx in range(2, 11))

        # Setting the equations for the Zernike polynomials
        # r = np.sqrt(powl(x,2) + powl(y,2))
        Z = {
            1: c[0] * np.ones(len(x)),  # m = 0    n = 0
            2: c[1] * x,  # m = -1   n = 1
            3: c[2] * y,  # m = 1    n = 1
            4: c[3] * 2 * x * y,  # m = -2   n = 2
            5: c[4] * (2 * x2 + 2 * y2 - 1),  # m = 0  n = 2
            6: c[5] * (-1 * x2 + y2),  # m = 2  n = 2
            7: c[6] * (-1 * x3 + 3 * x * y2),  # m = -3     n = 3
            8: c[7] * (-2 * x + 3 * (x3) + 3 * x * (y2)),  # m = -1   n = 3
            9: c[8] * (-2 * y + 3 * y3 + 3 * (x2) * y),  # m = 1    n = 3
            10: c[9] * (y3 - 3 * (x2) * y),  # m = 3 n =3
            11: c[10] * (-4 * (x3) * y + 4 * x * (y3)),  # m = -4    n = 4
            12: c[11] * (-6 * x * y + 8 * (x3) * y + 8 * x * (y3)),  # m = -2   n = 4
            13: c[12]
            * (
                1 - 6 * x2 - 6 * y2 + 6 * x4 + 12 * (x2) * (y2) + 6 * y4
            ),  # m = 0  n = 4
            14: c[13] * (3 * x2 - 3 * y2 - 4 * x4 + 4 * y4),  # m = 2    n = 4
            15: c[14] * (x4 - 6 * (x2) * (y2) + y4),  # m = 4   n = 4
            16: c[15] * (x5 - 10 * (x3) * y2 + 5 * x * (y4)),  # m = -5   n = 5
            17: c[16]
            * (
                4 * x3 - 12 * x * (y2) - 5 * x5 + 10 * (x3) * (y2) + 15 * x * y4
            ),  # m =-3     n = 5
            18: c[17]
            * (
                3 * x
                - 12 * x3
                - 12 * x * (y2)
                + 10 * x5
                + 20 * (x3) * (y2)
                + 10 * x * (y4)
            ),  # m= -1  n = 5
            19: c[18]
            * (
                3 * y
                - 12 * y3
                - 12 * y * (x2)
                + 10 * y5
                + 20 * (y3) * (x2)
                + 10 * y * (x4)
            ),  # m = 1  n = 5
            20: c[19]
            * (
                -4 * y3 + 12 * y * (x2) + 5 * y5 - 10 * (y3) * (x2) - 15 * y * x4
            ),  # m = 3   n = 5
            21: c[20] * (y5 - 10 * (y3) * x2 + 5 * y * (x4)),  # m = 5 n = 5
            22: c[21]
            * (6 * (x5) * y - 20 * (x3) * (y3) + 6 * x * (y5)),  # m = -6 n = 6
            23: c[22]
            * (
                20 * (x3) * y - 20 * x * (y3) - 24 * (x5) * y + 24 * x * (y5)
            ),  # m = -4   n = 6
            24: c[23]
            * (
                12 * x * y
                + 40 * (x3) * y
                - 40 * x * (y3)
                + 30 * (x5) * y
                + 60 * (x3) * (y3)
                - 30 * x * (y5)
            ),  # m = -2   n = 6
            25: c[24]
            * (
                -1
                + 12 * (x2)
                + 12 * (y2)
                - 30 * (x4)
                - 60 * (x2) * (y2)
                - 30 * (y4)
                + 20 * (x6)
                + 60 * (x4) * y2
                + 60 * (x2) * (y4)
                + 20 * (y6)
            ),  # m = 0   n = 6
            26: c[25]
            * (
                -6 * (x2)
                + 6 * (y2)
                + 20 * (x4)
                - 20 * (y4)
                - 15 * (x6)
                - 15 * (x4) * (y2)
                + 15 * (x2) * (y4)
                + 15 * (y6)
            ),  # m = 2   n = 6
            27: c[26]
            * (
                -5 * (x4)
                + 30 * (x2) * (y2)
                - 5 * (y4)
                + 6 * (x6)
                - 30 * (x4) * y2
                - 30 * (x2) * (y4)
                + 6 * (y6)
            ),  # m = 4    n = 6
            28: c[27]
            * (-1 * (x6) + 15 * (x4) * (y2) - 15 * (x2) * (y4) + y6),  # m = 6   n = 6
            29: c[28]
            * (
                -1 * (x7) + 21 * (x5) * (y2) - 35 * (x3) * (y4) + 7 * x * (y6)
            ),  # m = -7    n = 7
            30: c[29]
            * (
                -6 * (x5)
                + 60 * (x3) * (y2)
                - 30 * x * (y4)
                + 7 * x7
                - 63 * (x5) * (y2)
                - 35 * (x3) * (y4)
                + 35 * x * (y6)
            ),  # m = -5    n = 7
            31: c[30]
            * (
                -10 * (x3)
                + 30 * x * (y2)
                + 30 * x5
                - 60 * (x3) * (y2)
                - 90 * x * (y4)
                - 21 * x7
                + 21 * (x5) * (y2)
                + 105 * (x3) * (y4)
                + 63 * x * (y6)
            ),  # m =-3       n = 7
            32: c[31]
            * (
                -4 * x
                + 30 * x3
                + 30 * x * (y2)
                - 60 * (x5)
                - 120 * (x3) * (y2)
                - 60 * x * (y4)
                + 35 * x7
                + 105 * (x5) * (y2)
                + 105 * (x3) * (y4)
                + 35 * x * (y6)
            ),  # m = -1  n = 7
            33: c[32]
            * (
                -4 * y
                + 30 * y3
                + 30 * y * (x2)
                - 60 * (y5)
                - 120 * (y3) * (x2)
                - 60 * y * (x4)
                + 35 * y7
                + 105 * (y5) * (x2)
                + 105 * (y3) * (x4)
                + 35 * y * (x6)
            ),  # m = 1   n = 7
            34: c[33]
            * (
                10 * (y3)
                - 30 * y * (x2)
                - 30 * y5
                + 60 * (y3) * (x2)
                + 90 * y * (x4)
                + 21 * y7
                - 21 * (y5) * (x2)
                - 105 * (y3) * (x4)
                - 63 * y * (x6)
            ),  # m =3     n = 7
            35: c[34]
            * (
                -6 * (y5)
                + 60 * (y3) * (x2)
                - 30 * y * (x4)
                + 7 * y7
                - 63 * (y5) * (x2)
                - 35 * (y3) * (x4)
                + 35 * y * (x6)
            ),  # m = 5  n = 7
            36: c[35]
            * (y7 - 21 * (y5) * (x2) + 35 * (y3) * (x4) - 7 * y * (x6)),  # m = 7  n = 7
            37: c[36]
            * (
                -8 * (x7) * y + 56 * (x5) * (y3) - 56 * (x3) * (y5) + 8 * x * (y7)
            ),  # m = -8  n = 8
            38: c[37]
            * (
                -42 * (x5) * y
                + 140 * (x3) * (y3)
                - 42 * x * (y5)
                + 48 * (x7) * y
                - 112 * (x5) * (y3)
                - 112 * (x3) * (y5)
                + 48 * x * (y7)
            ),  # m = -6  n = 8
            39: c[38]
            * (
                -60 * (x3) * y
                + 60 * x * (y3)
                + 168 * (x5) * y
                - 168 * x * (y5)
                - 112 * (x7) * y
                - 112 * (x5) * (y3)
                + 112 * (x3) * (y5)
                + 112 * x * (y7)
            ),  # m = -4   n = 8
            40: c[39]
            * (
                -20 * x * y
                + 120 * (x3) * y
                + 120 * x * (y3)
                - 210 * (x5) * y
                - 420 * (x3) * (y3)
                - 210 * x * (y5)
                - 112 * (x7) * y
                + 336 * (x5) * (y3)
                + 336 * (x3) * (y5)
                + 112 * x * (y7)
            ),  # m = -2   n = 8
            41: c[40]
            * (
                1
                - 20 * x2
                - 20 * y2
                + 90 * x4
                + 180 * (x2) * (y2)
                + 90 * y4
                - 140 * x6
                - 420 * (x4) * (y2)
                - 420 * (x2) * (y4)
                - 140 * (y6)
                + 70 * x8
                + 280 * (x6) * (y2)
                + 420 * (x4) * (y4)
                + 280 * (x2) * (y6)
                + 70 * y8
            ),  # m = 0    n = 8
            42: c[41]
            * (
                10 * x2
                - 10 * y2
                - 60 * x4
                + 105 * (x4) * (y2)
                - 105 * (x2) * (y4)
                + 60 * y4
                + 105 * x6
                - 105 * y6
                - 56 * x8
                - 112 * (x6) * (y2)
                + 112 * (x2) * (y6)
                + 56 * y8
            ),  # m = 2  n = 8
            43: c[42]
            * (
                15 * x4
                - 90 * (x2) * (y2)
                + 15 * y4
                - 42 * x6
                + 210 * (x4) * (y2)
                + 210 * (x2) * (y4)
                - 42 * y6
                + 28 * x8
                - 112 * (x6) * (y2)
                - 280 * (x4) * (y4)
                - 112 * (x2) * (y6)
                + 28 * y8
            ),  # m = 4     n = 8
            44: c[43]
            * (
                7 * x6
                - 105 * (x4) * (y2)
                + 105 * (x2) * (y4)
                - 7 * y6
                - 8 * x8
                + 112 * (x6) * (y2)
                - 112 * (x2) * (y6)
                + 8 * y8
            ),  # m = 6    n = 8
            45: c[44]
            * (
                x8 - 28 * (x6) * (y2) + 70 * (x4) * (y4) - 28 * (x2) * (y6) + y8
            ),  # m = 8     n = 9
            46: c[45]
            * (
                x9
                - 36 * (x7) * (y2)
                + 126 * (x5) * (y4)
                - 84 * (x3) * (y6)
                + 9 * x * (y8)
            ),  # m = -9     n = 9
            47: c[46]
            * (
                8 * x7
                - 168 * (x5) * (y2)
                + 280 * (x3) * (y4)
                - 56 * x * (y6)
                - 9 * x9
                + 180 * (x7) * (y2)
                - 126 * (x5) * (y4)
                - 252 * (x3) * (y6)
                + 63 * x * (y8)
            ),  # m = -7    n = 9
            48: c[47]
            * (
                21 * x5
                - 210 * (x3) * (y2)
                + 105 * x * (y4)
                - 56 * x7
                + 504 * (x5) * (y2)
                + 280 * (x3) * (y4)
                - 280 * x * (y6)
                + 36 * x9
                - 288 * (x7) * (y2)
                - 504 * (x5) * (y4)
                + 180 * x * (y8)
            ),  # m = -5    n = 9
            49: c[48]
            * (
                20 * x3
                - 60 * x * (y2)
                - 105 * x5
                + 210 * (x3) * (y2)
                + 315 * x * (y4)
                + 168 * x7
                - 168 * (x5) * (y2)
                - 840 * (x3) * (y4)
                - 504 * x * (y6)
                - 84 * x9
                + 504 * (x5) * (y4)
                + 672 * (x3) * (y6)
                + 252 * x * (y8)
            ),  # m = -3  n = 9
            50: c[49]
            * (
                5 * x
                - 60 * x3
                - 60 * x * (y2)
                + 210 * x5
                + 420 * (x3) * (y2)
                + 210 * x * (y4)
                - 280 * x7
                - 840 * (x5) * (y2)
                - 840 * (x3) * (y4)
                - 280 * x * (y6)
                + 126 * x9
                + 504 * (x7) * (y2)
                + 756 * (x5) * (y4)
                + 504 * (x3) * (y6)
                + 126 * x * (y8)
            ),  # m = -1   n = 9
            51: c[50]
            * (
                5 * y
                - 60 * y3
                - 60 * y * (x2)
                + 210 * y5
                + 420 * (y3) * (x2)
                + 210 * y * (x4)
                - 280 * y7
                - 840 * (y5) * (x2)
                - 840 * (y3) * (x4)
                - 280 * y * (x6)
                + 126 * y9
                + 504 * (y7) * (x2)
                + 756 * (y5) * (x4)
                + 504 * (y3) * (x6)
                + 126 * y * (x8)
            ),  # m = -1   n = 9
            52: c[51]
            * (
                -20 * y3
                + 60 * y * (x2)
                + 105 * y5
                - 210 * (y3) * (x2)
                - 315 * y * (x4)
                - 168 * y7
                + 168 * (y5) * (x2)
                + 840 * (y3) * (x4)
                + 504 * y * (x6)
                + 84 * y9
                - 504 * (y5) * (x4)
                - 672 * (y3) * (x6)
                - 252 * y * (x8)
            ),  # m = 3  n = 9
            53: c[52]
            * (
                21 * y5
                - 210 * (y3) * (x2)
                + 105 * y * (x4)
                - 56 * y7
                + 504 * (y5) * (x2)
                + 280 * (y3) * (x4)
                - 280 * y * (x6)
                + 36 * y9
                - 288 * (y7) * (x2)
                - 504 * (y5) * (x4)
                + 180 * y * (x8)
            ),  # m = 5     n = 9
            54: c[53]
            * (
                -8 * y7
                + 168 * (y5) * (x2)
                - 280 * (y3) * (x4)
                + 56 * y * (x6)
                + 9 * y9
                - 180 * (y7) * (x2)
                + 126 * (y5) * (x4)
                - 252 * (y3) * (x6)
                - 63 * y * (x8)
            ),  # m = 7     n = 9
            55: c[54]
            * (
                y9
                - 36 * (y7) * (x2)
                + 126 * (y5) * (x4)
                - 84 * (y3) * (x6)
                + 9 * y * (x8)
            ),  # m = 9       n = 9
            56: c[55]
            * (
                10 * (x9) * y
                - 120 * (x7) * (y3)
                + 252 * (x5) * (y5)
                - 120 * (x3) * (y7)
                + 10 * x * (y9)
            ),  # m = -10   n = 10
            57: c[56]
            * (
                72 * (x7) * y
                - 504 * (x5) * (y3)
                + 504 * (x3) * (y5)
                - 72 * x * (y7)
                - 80 * (x9) * y
                + 480 * (x7) * (y3)
                - 480 * (x3) * (y7)
                + 80 * x * (y9)
            ),  # m = -8    n = 10
            58: c[57]
            * (
                270 * (x9) * y
                - 360 * (x7) * (y3)
                - 1260 * (x5) * (y5)
                - 360 * (x3) * (y7)
                + 270 * x * (y9)
                - 432 * (x7) * y
                + 1008 * (x5) * (y3)
                + 1008 * (x3) * (y5)
                - 432 * x * (y7)
                + 168 * (x5) * y
                - 560 * (x3) * (y3)
                + 168 * x * (y5)
            ),  # m = -6   n = 10
            59: c[58]
            * (
                140 * (x3) * y
                - 140 * x * (y3)
                - 672 * (x5) * y
                + 672 * x * (y5)
                + 1008 * (x7) * y
                + 1008 * (x5) * (y3)
                - 1008 * (x3) * (y5)
                - 1008 * x * (y7)
                - 480 * (x9) * y
                - 960 * (x7) * (y3)
                + 960 * (x3) * (y7)
                + 480 * x * (y9)
            ),  # m = -4   n = 10
            60: c[59]
            * (
                30 * x * y
                - 280 * (x3) * y
                - 280 * x * (y3)
                + 840 * (x5) * y
                + 1680 * (x3) * (y3)
                + 840 * x * (y5)
                - 1008 * (x7) * y
                - 3024 * (x5) * (y3)
                - 3024 * (x3) * (y5)
                - 1008 * x * (y7)
                + 420 * (x9) * y
                + 1680 * (x7) * (y3)
                + 2520 * (x5) * (y5)
                + 1680 * (x3) * (y7)
                + 420 * x * (y9)
            ),  # m = -2   n = 10
            61: c[60]
            * (
                -1
                + 30 * x2
                + 30 * y2
                - 210 * x4
                - 420 * (x2) * (y2)
                - 210 * y4
                + 560 * x6
                + 1680 * (x4) * (y2)
                + 1680 * (x2) * (y4)
                + 560 * y6
                - 630 * x8
                - 2520 * (x6) * (y2)
                - 3780 * (x4) * (y4)
                - 2520 * (x2) * (y6)
                - 630 * y8
                + 252 * x10
                + 1260 * (x8) * (y2)
                + 2520 * (x6) * (y4)
                + 2520 * (x4) * (y6)
                + 1260 * (x2) * (y8)
                + 252 * y10
            ),  # m = 0    n = 10
            62: c[61]
            * (
                -15 * x2
                + 15 * y2
                + 140 * x4
                - 140 * y4
                - 420 * x6
                - 420 * (x4) * (y2)
                + 420 * (x2) * (y4)
                + 420 * y6
                + 504 * x8
                + 1008 * (x6) * (y2)
                - 1008 * (x2) * (y6)
                - 504 * y8
                - 210 * x10
                - 630 * (x8) * (y2)
                - 420 * (x6) * (y4)
                + 420 * (x4) * (y6)
                + 630 * (x2) * (y8)
                + 210 * y10
            ),  # m = 2  n = 10
            63: c[62]
            * (
                -35 * x4
                + 210 * (x2) * (y2)
                - 35 * y4
                + 168 * x6
                - 840 * (x4) * (y2)
                - 840 * (x2) * (y4)
                + 168 * y6
                - 252 * x8
                + 1008 * (x6) * (y2)
                + 2520 * (x4) * (y4)
                + 1008 * (x2) * (y6)
                - 252 * (y8)
                + 120 * x10
                - 360 * (x8) * (y2)
                - 1680 * (x6) * (y4)
                - 1680 * (x4) * (y6)
                - 360 * (x2) * (y8)
                + 120 * y10
            ),  # m = 4     n = 10
            64: c[63]
            * (
                -28 * x6
                + 420 * (x4) * (y2)
                - 420 * (x2) * (y4)
                + 28 * y6
                + 72 * x8
                - 1008 * (x6) * (y2)
                + 1008 * (x2) * (y6)
                - 72 * y8
                - 45 * x10
                + 585 * (x8) * (y2)
                + 630 * (x6) * (y4)
                - 630 * (x4) * (y6)
                - 585 * (x2) * (y8)
                + 45 * y10
            ),  # m = 6    n = 10
            65: c[64]
            * (
                -9 * x8
                + 252 * (x6) * (y2)
                - 630 * (x4) * (y4)
                + 252 * (x2) * (y6)
                - 9 * y8
                + 10 * x10
                - 270 * (x8) * (y2)
                + 420 * (x6) * (y4)
                + 420 * (x4) * (y6)
                - 270 * (x2) * (y8)
                + 10 * y10
            ),  # m = 8    n = 10
            66: c[65]
            * (
                -1 * x10
                + 45 * (x8) * (y2)
                - 210 * (x6) * (y4)
                + 210 * (x4) * (y6)
                - 45 * (x2) * (y8)
                + y10
            ),  # m = 10   n = 10
        }
        return Z

def apply_operator(inv_noise_var, coeff_cov_inv, zern_trans):
    """
    Apply LHS operator to vector of Zernike coefficients.

    Parameters:
        inv_noise_var (array_like): Inverse variance of same shape as vis.
            Assumes diagonal covariance matrix, which is true in practice.

        coeff_cov_inv (array_like, complex): Inverse prior covariance matrix for
            the Zernike coeffs.

        zern_trans: (array_like, complex): Matrix that, when applied to a vector
            of Zernike coefficients for one antenna, returns the visibilities
            associated with that antenna.
    """
    return

def construct_rhs(vis, inv_noise_var, coeff_mean, coeff_cov_inv_sqrt):
    """
    Construct the right-hand-side of the Gaussian Constrained Realization (GCR)
    equation.

    Parameters:
        vis (array_like, complex): Subset of visiblities belonging to the
            antenna for which the GCR is being set up.

        inv_noise_var (array_like): Inverse variance of same shape as vis.
            Assumes diagonal covariance matrix, which is true in practice.

        coeff_mean (array_like, complex): Prior mean for the Zernike
            coefficients.

        coeff_cov_inv_sqrt (array_like, complex): Square root of the prior
            inverse covariance matrix for the Zernike coefficients.

    """
    return