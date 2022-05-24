#!/bin/bash
# Extract primary tar file to desired target folder
tar -xvf ~/data/eeg_full.tar -C ~/data/aud
# Change directory to the desired target folder
cd ~/data/aud
# Extract each subject's tar.gz file as directories with the same name in the
# same location
cat *.tar.gz | tar -xvf - --ignore-zeros
# Delete the compressed tar.gz files that have been extracted
find . -type f -name "*.tar.gz" -delete
# Extract and replace all the trial files for each subject inside their
# respective folders
find . -name "*.gz" -exec gunzip {} \;
