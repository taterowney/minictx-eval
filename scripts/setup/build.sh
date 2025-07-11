#!/bin/bash

git submodule update --init --recursive

export PROJECT_ROOT=$(pwd)

echo "Building Lean REPL 4.16.0"
cd $PROJECT_ROOT/repls/repl-4.16.0
lake build

echo "Building Lean REPL 4.16.0-rc1"
cd $PROJECT_ROOT/repls/repl-4.16.0-rc1
lake build

echo "Building carleson"
cd $PROJECT_ROOT/test-envs/minictx-v2/carleson
lake exe cache get
lake build

echo "Building con-nf"
cd $PROJECT_ROOT/test-envs/minictx-v2/con-nf
lake exe cache get
lake build

echo "Building FLT"
cd $PROJECT_ROOT/test-envs/minictx-v2/FLT
lake exe cache get
lake build

echo "Building Foundation"
cd $PROJECT_ROOT/test-envs/minictx-v2/Foundation
lake exe cache get
lake build

echo "Building PhysLean"
cd $PROJECT_ROOT/test-envs/minictx-v2/PhysLean
lake exe cache get
lake build

echo "Building seymour"
cd $PROJECT_ROOT/test-envs/minictx-v2/seymour
lake exe cache get
lake build

echo "Building mathlib4"
cd $PROJECT_ROOT/test-envs/minictx-v2/mathlib4
lake exe cache get
lake build