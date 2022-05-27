#!/bin/bash
# Set variable values
SRCFILE=~/data/eeg_full.tar
TARDIR=~/data/aud
# Copy the list of blacklisted files for later use
cp ./blacklist.txt $TARDIR
# Extract primary tar file to desired target folder
tar -xvf $SRCFILE -C $TARDIR
# Change directory to the desired target folder
cd $TARDIR
# Extract each subject's tar.gz file as directories with the same name in the
# same location
cat *.tar.gz | tar -xvf - --ignore-zeros
# Delete the compressed tar.gz files that have been extracted
find . -type f -name "*.tar.gz" -delete
# Extract and replace all the trial files for each subject inside their
# respective folders
find . -name "*.gz" -exec gunzip {} \;
# Delete the 17 'blacklisted' trials with empty files in co2c1000367
xargs rm < ./blacklist.txt
rm ./blacklist.txt
