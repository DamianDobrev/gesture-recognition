# gesture-recognition
Gesture recognition software for controlling a drone

#### To train, edit and run model.py with the appropriate data:
```
python model
```

This will save in `./_results/timestamp-of-finish` the result

#### Syncing with hpc, run from `./`
```
rsync -a --exclude-from 'exclude-list.txt' -e ssh ./ k1631439@login.rosalind.kcl.ac.uk:~/sharedscratch/gest-src/
```