import numpy as np

import cloudmetrics


def test_large_uniform_circle_orientation():
    """
    Large uniform circle (should be 0)
    """
    # 1. One large, uniform circle
    t1 = np.ones((512, 512))
    mask = createCircularMask(512, 512)

    cloudmetrics.orientation()


def verify(self):
    """
    Verification based on three simple tests:
        1. Large uniform circle (should be 0)
        2. Randomly scattered points (should be 0)
        3. Vertical lines (should be 1)

    Returns
    -------
    veri : List of floats
        List containing metric(s) for verification case.

    """

    # 2. Randomly scattered points
    t1[~mask] = 0
    t2 = np.random.rand(512, 512)
    ind = np.where(t2 > 0.5)
    t2[ind] = 1
    ind = np.where(t2 <= 0.5)
    t2[ind] = 0

    # 3. Vertical lines
    t3 = np.zeros((512, 512))
    t3[:, 250:251] = 1
    tests = [t1, t2, t3]

    veri = []
    for i in range(len(tests)):
        orie = self.metric(tests[i])
        veri.append(orie)

    return veri
