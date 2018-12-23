# ThoughtViz
Implementation for the paper https://dl.acm.org/citation.cfm?doid=3240508.3240641

## EEG Data

* Download the EEG data from https://drive.google.com/open?id=1atP9CsjWIT-hg3fX--fcC1hg0uvg9bEH
* Extract it and place it in the project folder (.../ThoughtViz/data)

## Trained Models

* Download the trained EEG Classification and GAN models from https://drive.google.com/open?id=1AYff632-iwtkhGi6jWAiOkt-XLgthfNl 
* Extract it and place it in the project folder (.../ThoughtViz/models)

**NOTE** : Currently we have uploaded only one baseline model and our final model. Other baseline models will be updated soon. 

## Training

1. EEG Classification

2. GAN Training

## Testing

Run test.py to run the tests 

1. Baseline Evaluation

   * DeLiGAN : Uses 1-hot class label as conditioning with MoGLayer at the input.


2. Final Evaluation

   * Our Approach : Uses EEG encoding from the trained EEG classifier as conditioning. The encoding is used as weights in the MoGLayer

