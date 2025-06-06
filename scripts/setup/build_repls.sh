#!/bin/bash

#git submodule init
#git submodule update

cd ./repls/repl-4.16.0
lake build

cd ../..
cp test-envs/minictx-v2/con-nf/lean-toolchain ./repls/repl-4.16.0-rc1/lean-toolchain
cd ./repls/repl-4.16.0-rc1
lake build