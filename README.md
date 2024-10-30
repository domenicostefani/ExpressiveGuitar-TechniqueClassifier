# Expressive Guitar Technique classifier

Ph.D. research project of [Domenico Stefani](http://work.domenicostefani.com/)  
The Jupyter notebook loads a dataset of feature vectors extracted from **pitched** and **percussive** sounds recorded with many acoustic guitars.
___
The techniques/classes are:  

0.    **Kick**      (Palm on lower body)
1.    **Snare 1**   (All fingers on lower side)
2.    **Tom**       (Thumb on higher body)
3.    **Snare 2**   (Fingers on the muted strings, over the end
of the fingerboard)
___
4.    **Natural Harmonics** (Stop strings from playing the dominant frequency, letting harmonics ring)
5.    **Palm Mute** (Muting partially the strings with the palm
of the pick hand)
6.    **Pick Near Bridge** (Playing toward the bridge/saddle)
7.    **Pick Over the Soundhole** (Playing over the sound hole) (NEUTRAL NON-)TECHNIQUE 

___

## Content of the repository

- `data/`: Folder containing the links to the feature dataset files. Download them in the folder with the links file.
- `phase3results/`: Experimental results (see paper).
- `convert_to_script.py` : Script to convert the Colab/Jupyter notebook to a Python script.
- `expressive-technique-classifier-phase3.ipynb` : Jupyter notebook with the code to train and test the classifier.
- `guitarists_touch.ipynb`: Jupyter notebook with the code to train and test the classifier for Experiment 3.
- `run_grid_search.py` : Script to run a grid search on the classifier.

Contact Domenico Stefani for any issues with running the code to repeat the experiments.  

domenico[dot]stefani[at]unitn[dot]it  
[work.domenicostefani.com](http://work.domenicostefani.com/)
