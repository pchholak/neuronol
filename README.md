# neuronol

Use shell script in ./HelperFiles to extract EEG data files in a target folder.
Then use meta-analysis script to convert data to MNE-Epochs objects and save in
__aud-mne__ folder.

Discrepancies in UCI dataset

1. Data not bandpass filtered between 0.02 and 50 Hz as claimed in Zhang el al. 1995.
2. Unclear if baselines are included in the epoch data or not.
