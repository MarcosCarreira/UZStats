#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:48:07 2020

@author: marcoscscarreira
"""
import numpy as np
import pandas as pd
from tick.hawkes import (SimuHawkes, HawkesKernelTimeFunc, HawkesKernelExp,
                         HawkesEM)
from tick.base import TimeFunction
from tick.hawkes import HawkesConditionalLaw

run_time = 30000

t_values1 = np.array([0, 1, 1.5, 2., 3.5, 4.], dtype=float)
y_values1 = np.array([0.2, 0.3, 0., 0.1, 0.2, 0.], dtype=float)
tf1 = TimeFunction([t_values1, y_values1],
                   inter_mode=TimeFunction.InterConstRight, dt=0.1)
kernel1 = HawkesKernelTimeFunc(tf1)

baseline = np.array([0.1])

hawkes = SimuHawkes(baseline=baseline, end_time=run_time, verbose=False,
                    seed=2334)

hawkes.set_kernel(0, 0, kernel1)

hawkes.simulate()

em = HawkesEM(4, kernel_size=16, n_threads=8, verbose=False, tol=1e-3)
em.fit(hawkes.timestamps)

em_kernels = em.kernel

kern_d = np.linspace(0, 4, 16+1)

em2 = HawkesEM(kernel_discretization=kern_d, n_threads=8,
               verbose=False, tol=1e-3)
em2.fit(hawkes.timestamps)

em2_kernels = em2.kernel

kern_d3 = np.array([0, 1, 1.5, 2., 3.5, 4.], dtype=float)

em3 = HawkesEM(kernel_discretization=kern_d3, n_threads=8,
               verbose=False, tol=1e-3)
em3.fit(hawkes.timestamps)

em3_kernels = np.append(em3.kernel[0][0], 0.)
print(em3.get_kernel_norms())

df1 = pd.DataFrame({"orig": y_values1, "Learn": em3_kernels}, index=t_values1)

dfk1 = pd.DataFrame({
    'ksize=16': em_kernels[0][0],
    'kdisc': em2_kernels[0][0]})



# print(kern_d)

# [0.   0.25 0.5  0.75 1.   1.25 1.5  1.75 2.   2.25 2.5  2.75 3.   3.25
#  3.5  3.75 4.  ]

print(df1)

#      orig     Learn
# 0.0   0.2  0.196139
# 1.0   0.1  0.098355
# 1.5   0.0  0.004003
# 2.0   0.1  0.101001
# 3.5   0.2  0.213103
# 4.0   0.0  0.000000

# print(dfk1)

#     ksize=16     kdisc
# 0   0.180935  0.181033
# 1   0.203580  0.203602
# 2   0.188777  0.188813
# 3   0.194791  0.194700
# 4   0.310664  0.310658
# 5   0.303877  0.303795
# 6   0.011583  0.010918
# 7   0.004916  0.005844
# 8   0.113431  0.113512
# 9   0.089664  0.089500
# 10  0.106115  0.106188
# 11  0.094219  0.094594
# 12  0.113220  0.112034
# 13  0.093960  0.094659
# 14  0.196326  0.196315
# 15  0.225077  0.225080

cl = HawkesConditionalLaw(claw_method="log", delta_lag=0.1, min_lag=0.002,
                         max_lag=100, quad_method="log", n_quad=50,
                         min_support=0.002, max_support=10, n_threads=-1)
cl.fit(hawkes.timestamps)
print(cl.kernels_norms)
print(0.25*np.sum(em2_kernels[0][0]))
