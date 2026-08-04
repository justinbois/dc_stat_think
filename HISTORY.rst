=======
History
=======

0.1.0 (2017-07-20)
0.1.1 (2017-07-20)
0.1.2 (2017-07-24)
0.1.4 (2017-07-26)
0.1.5 (2017-08-17)
1.0.0 (2017-08-28)
1.0.1 (2019-02-19)
1.0.2 (2019-02-20)
1.1.0 (2020-07-14)
------------------

1.1.2 (2026-07-31)
------------------
* Replace removed ``np.float`` alias with the builtin ``float`` in
  ``_convert_data`` (both ``dc_stat_think`` and ``no_numba``). Restores
  compatibility with NumPy >= 1.24, where ``np.float`` was removed.
