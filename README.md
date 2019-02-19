# DataCamp Statistical Thinking utilities

[![version](https://img.shields.io/pypi/v/dc_stat_think.svg)](https://pypi.python.org/pypi/dc_stat_think) [![build status](https://img.shields.io/travis/justinbois/dc_stat_think.svg)](https://travis-ci.org/justinbois/dc_stat_think) 

Utility functions used in the DataCamp Statistical Thinking courses.
- [Statistical Thinking in Python Part I](https://www.datacamp.com/courses/statistical-thinking-in-python-part-1/)
- [Statistical Thinking in Python Part II](https://www.datacamp.com/courses/statistical-thinking-in-python-part-2/)
- [Case Studies in Statistical Thinking](https://www.datacamp.com/courses/case-studies-in-statistical-thinking/)


## Installation
dc_stat_think may be installed by running the following command.
```
pip install dc_stat_think
```

## Usage
Upon importing the module, functions from the DataCamp Statistical Thinking courses are available. For example, you can compute a 95% confidence interval of the mean of some data using the `draw_bs_reps()` function.

```python
>>> import numpy as np
>>> import dc_stat_think as dcst
>>> data = np.array([1.2, 3.3, 2.7, 2.4, 5.6, 
                     3.4, 1.3, 3.9, 2.9, 2.1, 2.7])
>>> bs_reps = dcst.draw_bs_reps(data, np.mean, size=10000)
>>> conf_int = np.percentile(bs_reps, [2.5, 97.5])
>>> print(conf_int)
[ 2.21818182  3.60909091]
```

## Implementation
The functions include in dc_stat_think are not *exactly* like those students wrote in the DataCamp Statistical Thinking courses. Notable differences are listed below.

+ The doc strings in dc_stat_think are much more complete.
+ The dc_stat_think module has error checking of inputs.
+ In most cases, especially those involving bootstrapping or other uses of the `np.random` module, dc_stat_think functions are more optimized for speed, in particular using [Numba](http://numba.pydata.org). Note, though, that dc_stat_think does not take advantage of any parallel computing.

If you do want to use functions *exactly* as written in the Statistical Thinking courses, you can use the `dc_stat_think.original` submodule.

```python
>>> import numpy as np
>>> import dc_stat_think.original
>>> data = np.array([1.2, 3.3, 2.7, 2.4, 5.6, 3.4, 1.3, 3.9, 2.9, 2.1, 2.7])
>>> bs_reps = dc_stat_think.original.draw_bs_reps(data, np.mean, size=10000)
>>> conf_int = np.percentile(bs_reps, [2.5, 97.5])
>>> print(conf_int)
[ 2.20909091  3.59090909]
```

## Credits
This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template and then modified.
