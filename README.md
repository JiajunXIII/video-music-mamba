# Video-Music-Mamba-VMM-
VMM: Video-Music Mamba for Generating Music from Videos

## Directory Structure

* `saved_models/`: saved model files
* `model/`
  * `video_music__mixer_mamba.py`: Video Music Mamba (VMM) model 
  * `video_regression.py`: Bi-GRU regression model used for predicting note density/loudness
  * `positional_encoding.py`: code for Positional encoding
  * `rpr.py`: code for RPR (Relative Positional Representation)
* `dataset/`
  * `vevo_dataset.py`: Dataset loader
* `generate.py`: inference script

## Dataset

* Obtain the dataset:
  * MuVi-Sync [(Link)](https://zenodo.org/records/10057093)

Pull the entire "dataset" directory into the local "dataset" folder.(`dataset/dataset/`)

## Pre-trained models

* Obtain the pre-trained models:
  * VMM and AMT [(Link)](https://drive.google.com/file/d/131kWNKMW6CZ_aEtPi-TS7T0TiATJVuTU/view?usp=sharing)

Download the `saved_models` directory which including  the processed training data `AMT.zip` and  our VMM pre-trained model.(`saved_models/AMT/` and ` saved_model/2025_1_VMM_contrastive_11_MuVi-Sync_best`/)

## Soundfont

Download the soundfont file `default_sound_font.sf2` from [HERE](https://drive.google.com/file/d/1PBMwfe36U6FKBHEfhN2HBB7y0Ek1QHN5/view?usp=sharing) and put the file directly under this folder (`soundfonts/`)

##  Inference

```python
python generate.py
```



## Training

Our training code will be released later~



## Acknowledgements

Our code is based on [Video2Music](https://github.com/AMAAI-Lab/Video2Music).
