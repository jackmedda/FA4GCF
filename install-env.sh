if [ -z "$1" ]
  then
    echo "No CUDA version supplied, e.g., cu121";
    exit 1
fi

pip install wheel --upgrade
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/$1 --no-cache-dir
pip install torch_scatter torch_sparse torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+$1.html --no-cache-dir
pip install gnnuers
pip install kmeans_pytorch