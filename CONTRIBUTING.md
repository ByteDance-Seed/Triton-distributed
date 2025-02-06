<!-- #
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -->

# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Changes Accepted

Please file issues before doing substantial work; this will ensure that others
don't duplicate the work and that there's a chance to discuss any design issues.
Changes only tweaking style are unlikely to be accepted unless they are applied
consistently across the project. 
Most of the code style is derived from the
[Google Style Guides](http://google.github.io/styleguide/) for the appropriate
language and is generally not something we accept changes on (as clang-format
and clang-tidy handle that for us).
For Python code, we utilize `yapf` and `ruff`. 

- `yapf` is used to format Python code according to the Google Python Style Guide. It helps in making the code more readable and maintainable by applying consistent formatting rules.
- `ruff` serves as a fast Python code analysis and formatting tool. It combines multiple code - checking and formatting functionalities, and can quickly identify and fix style issues in Python code, ensuring it meets our style requirements.

The compiler portion of the project follows
[MLIR style](https://mlir.llvm.org/getting_started/DeveloperGuide/#style-guide).
Improvements to code structure and clarity are welcome but please file issues to
track such work first.


## AUTHORS file

If you would like to receive additional recognition for your contribution, you
may add yourself (or your organization) to the AUTHORS file. This keeps track of
those who have made significant contributions to the project. Please add the
entity who owns the copyright for your contribution. The source control history
remains the most accurate source for individual contributions.

##  Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

