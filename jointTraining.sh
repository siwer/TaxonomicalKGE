#!/bin/bash

python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt

python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt

python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt

python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.01 |& tee -a jointNYC.txt

python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt

python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt

python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt

python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt
python3 jointRuns.py -data NYC -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.005 |& tee -a jointNYC.txt

python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt

python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt

python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt

python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.01 |& tee -a jointJRK.txt

python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 130 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt

python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 75 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt

python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 50 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt

python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.0 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.2 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.4 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.6 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 0.8 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt
python3 jointRuns.py -data JRK -nrEpochs 15 -ontWeight 1.0 -nrDimensions 130 -ontDimension 25 -batchSize 512 -lr 0.005 |& tee -a jointJRK.txt