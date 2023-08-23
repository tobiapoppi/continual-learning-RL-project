# continual-learning-RL-project

## To prepare your environment you need to follow these instructions. Please, if you can, mind to the versions because it's easy to have incompatibilities.
1. create a new conda env
   - `conda create --name rl python=3.8`
   - `conda activate rl`
2. if you are on Windows AND you have a gpu and cuda installed (it's not mandatory!):

   for cuda <=11.7 (I have 11.2 and it's working fine):
   - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`
     
   for cuda 11.8:
   - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
4. if you don't have a gpu/cuda installed OR you are on Linux:
   - `pip install torch torchvision`
5. install other packets:
   - `pip install ray gym[atari]==0.23.1 gym[accept-rom-license]==0.23.1 opencv-python atari-py pygame numpy`
6. install avalanche 0.3.1 (note that you will need to change some files there, so I suggest you to install it manually with -e mode, and to modify the code as follows):
   - download this zip file: `https://github.com/ContinualAI/avalanche/archive/refs/tags/v0.3.1.zip`
   - unzip it and go with the terminal inside its folder
   - `pip install -e .`
   - go to `/avalanche/training/plugins/ewc.py` and substitute line __76__ and __116__ with this line: `exp_counter = strategy.training_exp_counter`
7. install avalanche-rl
   - `pip install git+https://github.com/ContinualAI/avalanche-rl.git`
  

### To test if your env is working fine just activate the conda env and run:

`python ewc_avalanche_load.py`

You should see the emulator opening the pong game failing miserably (but you should at lease see it playing). 
