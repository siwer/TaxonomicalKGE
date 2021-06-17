import torch
import argparse
import joinFullHype
from datetime import datetime

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu',default=0,type=int,help="which GPU to use")
    parser.add_argument('-data', default="NYC", type=str, help="which set (NYC or JRK)")
    parser.add_argument('-lr',default=0.01,type=float,help="learning rate")
    parser.add_argument('-nrDimensions', default=130, type=int,help="embedding dimensions")
    parser.add_argument('-ontDimension', default=50, type=int,help="ontology dimensions")
    parser.add_argument('-ontWeight',default=0.5,type=float,help="importance of ontology")
    parser.add_argument('-nrEpochs',default=15,type=int,help="nr Epochs")
    parser.add_argument('-batchSize',default=512,type=int,help="batchsize")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    now = datetime.now()
    print (now)
    args = get_parameter()
    if args.data == "NYC":
        train = 'Prepared/jointData/NYC/train7411_3626_293.data'
        valid = 'Prepared/jointData/NYC/valid7411_3626_293.data'
        test = 'Prepared/jointData/NYC/test7411_3626_293.data'
        nrLocations = 3626
        nrEntities = 7411
        nrTypes = 293
    elif args.data == "JRK":
        train = 'Prepared/jointData/JRK/train15020_8805_326.data'
        valid = 'Prepared/jointData/JRK/valid15020_8805_326.data'
        test = 'Prepared/jointData/JRK/test15020_8805_326.data'
        nrLocations = 8805
        nrEntities = 15020
        nrTypes = 326

    torch.manual_seed(12345)
    testLearner = joinFullHype.JointLearner(args.gpu,nrLocations,nrEntities,nrTypes,2,args.nrDimensions,args.ontDimension)
    testTrainer = joinFullHype.Trainer(train,valid,test,args.nrEpochs,args.lr,args.batchSize,args.ontWeight)
    testTrainer.trainModel(testLearner)
    testTrainer.testModel(testLearner)
    now = datetime.now()
    print (now)
    print ('~~~~~~~~~~~~~~~~~~')