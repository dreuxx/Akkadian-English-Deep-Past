#!/bin/bash
sudo apt-get update
sudo apt-get install -y cmake python3.12-dev
pip3 install --user --break-system-packages eflomal
git clone https://github.com/clab/fast_align.git
cd fast_align && mkdir build && cd build && cmake .. && make
sudo cp atools /usr/local/bin/
cd ../..
echo "✅ Instalación completada"
