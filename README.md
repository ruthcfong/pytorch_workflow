# pytorch-workflow

This repo contains helper code for a research workflow using [PyTorch](http://pytorch.org/). Still a work in progress.

To use in any project, run `export PYTHONPATH="PATH/TO/THIS/REPO":$PYTHONPATH`, replacing `PATH/TO/THIS/REPO` with the absolute path of this repo (alternatively, this line can be included in your .bash_profile; see this [repo](https://github.com/ruthcfong/coding-tricks) for more details about setting up a .bash_profile script).

Then, in your new project, add `from utils import *` or `from utils import FUNCTION_A, FUNCTION_B, ETC` to python files in which you plan to use these helper functions and/or objects.
