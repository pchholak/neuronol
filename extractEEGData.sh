tar -xvf ~/data/eeg_full.tar -C ~/data/aud
cd ~/data/aud
cat *.tar.gz | tar -xvf - --ignore-zeros
