"""Module 6 stage scripts, importable as a package.

This file exists so `scripts` is a REGULAR package rather than an implicit
namespace package. Without it, `from scripts.m6_common import ...` only
resolves when the project root already happens to be on `sys.path`, which is
true when running `python scripts/m6_01_encode_evidence.py` and false in a
fresh notebook -- producing `ModuleNotFoundError: No module named 'scripts'`
even though the file is plainly there.

Making the package explicit removes that ambiguity. It changes nothing about
running the stage scripts directly from the command line.

Deliberately empty otherwise: importing `scripts` must not pull in torch,
transformers, or any corpus.
"""
