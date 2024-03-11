```
chmod u+x run_gin.sh
./run_gin.sh | tee ../../GINenergygrids/logs/out/hyperopt_GIN_all.out

export PYTORCH_ENABLE_MPS_FALLBACK=1

conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64
MACOSX_DEPLOYMENT_TARGET=12.5 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch torchvision torchaudio

$ MACOSX_DEPLOYMENT_TARGET=14.2 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+${cpu}.html
$ MACOSX_DEPLOYMENT_TARGET=14.2 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+${cpu}.html
$ MACOSX_DEPLOYMENT_TARGET=14.2 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+${cpu}.html
$ MACOSX_DEPLOYMENT_TARGET=14.2 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-geometric

```