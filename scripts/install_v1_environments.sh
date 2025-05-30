#!/bin/bash

git submodule add https://github.com/leanprover-community/mathlib4.git test-envs/minictx-v1/mathlib4
git submodule add https://github.com/teorth/pfr.git test-envs/minictx-v1/pfr
git submodule add https://github.com/AlexKontorovich/PrimeNumberTheoremAnd.git test-envs/minictx-v1/PrimeNumberTheoremAnd
git submodule add https://github.com/hanwenzhu/HTPILeanPackage4.7.git test-envs/minictx-v1/HTPILeanPackage4.7
git submodule add https://github.com/yangky11/miniF2F-lean4.git test-envs/minictx-v1/miniF2F-lean4
git submodule add https://github.com/hanwenzhu/HepLean-v4.7.git test-envs/minictx-v1/HepLean-v4.7
git submodule add https://github.com/lecopivo/SciLean.git test-envs/minictx-v1/SciLean

cd test-envs/minictx-v1/mathlib4
git checkout 85a47191abb7957cdc53c5c2b59aef219bd8f6d9
cd ../pfr
git checkout 6aeed6ddf7dd02470b3196e44527a6f3d32e54cf
cd ../PrimeNumberTheoremAnd
git checkout c2c4b1362152f60ef6a2c903c7a6b7e1dd013f51
cd ../HTPILeanPackage4.7
git checkout 8eeebaec8d7fa17b5fe9d97589839ca2560e3ce2
cd ../miniF2F-lean4
git checkout 24c9bbafa568a1dc08dcd286ef78380dba5a9377
cd ../HepLean-v4.7
git checkout 7448822afced644bd61f0bdcf4dc438f455f9890
cd ../SciLean
git checkout e03a8e439a558840118319a6c8a47106e1445e53

cd ../../..
git submodule update --init --recursive