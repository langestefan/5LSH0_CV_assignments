# Introduction
This is a simple implementation of LSTMs for music generation based on the
original (https://github.com/Skuldur/Classical-Piano-Composer). 
With the code presented here you'll be able to train and test a simple network 
for music generation. 
## Requirements
* Python 3.x
* Python packages:
    * Music21     
    * h5py
    * pyaml
    * Pytorch
    * tqdm
    
Install with conda:
``` 
conda install -c mutirri music21
conda install -c anaconda h5py 
conda install -c conda-forge pyaml
conda install -c conda-forge tqdm
``` 
    
# How to use
Unzip the provided files 'midi_out' and 'midi_songs'. All the configurable parameters
 are found on the configuration files ('./configs'). By changing the values of 
 the parameters in the config files you'll be able to run the main script for 
 both training and testing.

* Data_path should point to the 'midi_songs' folder.
* Note_file should point to the 'notes.h5' file.
* Out_path should point to the place where you want your outputs to be saved to.

## Training
If training from scratch it is recommended to use a new config file. Otherwise, 
modify one of the existing ones with the desired parameters e.g. 'batch_size',
'epochs', etc. Training one of the provided models for additional epochs is possible 
by turning on the 'resume' flag. If you dont have a GPU, change the device parameter
to 'cpu'. 
Then, change the 'main.py' script with the path to your configuration file and run 
it. It will print the loss at each epoch but no further logging is done.  

## Testing
Testing is performed automatically at various epochs after the network is done 
training. You can specify the epochs via the 'ckpt_epoch'  parameter.  

## Listening to outputs
The songs are saved in the midi format. Not all media players support this format
without installing additional plugins. VLC media player can play this type of 
file after proper configuration. For Ubuntu systems, Timidity is a simple midi 
player. If you cannot listen to the songs, make sure your chosen media player
supports the midi format. 