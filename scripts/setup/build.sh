#!/bin/bash

git submodule update --init --recursive

export PROJECT_ROOT=$(pwd)

cd $PROJECT_ROOT/repls/repl-4.16.0
lake build

cd $PROJECT_ROOT/repls/repl-4.16.0-rc1
lake build

cd $PROJECT_ROOT/test-envs/minictx-v2/carleson
lake build

cd $PROJECT_ROOT/test-envs/minictx-v2/con-nf
lake build

cd $PROJECT_ROOT/test-envs/minictx-v2/FLT
lake build

cd $PROJECT_ROOT/test-envs/minictx-v2/Foundation
lake build

cd $PROJECT_ROOT/test-envs/minictx-v2/PhysLean
lake build

cd $PROJECT_ROOT/test-envs/minictx-v2/seymour
lake build

cd $PROJECT_ROOT/test-envs/minictx-v2/mathlib4
lake exe cache get
lake build