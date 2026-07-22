# SBLA_in_sim

# Installation notes

1. Download the code and enter its root directory. To keep with the latest version you might want to clone the repository from GitHub:
```
git clone https://github.com/iprafols/SBLA_in_sim.git
cd SBLA_in_sim
```

2. It is recomended to create a clean conda environment:
```
conda create --name sbla_in_sim python=3.13
conda activate sbla_in_sim
```

3. Install requirements:
```
pip install -r requirements.txt
```

3. Install DESI dependencies
```
cd ..
git clone https://github.com/desihub/desimodel.git
cd desimodel
pip install -e .
cd ..
git clone https://github.com/desihub/desiutil.git
cd desiutil
pip install -e .
cd ..
git clone https://github.com/desihub/desisim.git
cd desisim
pip install --no-build-isolation -e .
cd ..
git clone https://github.com/desihub/desispec.git
cd desispec
pip install -e .
cd ..
git clone https://github.com/desihub/desitarget.git
cd desitarget
pip install -e .
cd ..
cd SBLA_in_sim
```

4. Install code
```
pip install -e .
```