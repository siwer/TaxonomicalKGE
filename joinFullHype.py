import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import math
'''
All HypE-related code is adapted or taken from:
https://github.com/ElementAI/HypE
TransE scoring adapted from:
https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/TransE.py
'''
class JointLearner(nn.Module):
    def __init__(self,gpuNr,nrLocations,nrEntities,nrOntology,nrRelations,embeddingDimension,ontologyDimension):
        super(JointLearner,self).__init__()
        self.gpuNr = gpuNr
        self.nrEntities = nrEntities
        self.nrOntology = nrOntology
        self.nrRelations = nrRelations
        self.embeddingDimension = embeddingDimension
        #ont Dim has to equal emb Dim because of shared relation dimensionality
        self.ontologyDimension = ontologyDimension
        self.nrLocations = nrLocations

        self.device = torch.device('cuda:'+str(self.gpuNr) if torch.cuda.is_available() else 'cpu')
        #Embedding layers
        self.entitySpace = torch.nn.Embedding(self.nrEntities,self.embeddingDimension).to(self.device)
        torch.nn.init.xavier_uniform_(self.entitySpace.weight)
        self.ontologySpace = torch.nn.Embedding(self.nrOntology, self.ontologyDimension).to(self.device)
        torch.nn.init.xavier_uniform_(self.ontologySpace.weight)
        self.relationSpace = torch.nn.Embedding(self.nrRelations, self.embeddingDimension).to(self.device)
        torch.nn.init.xavier_uniform_(self.relationSpace.weight)
        #Projection layer
        self.projection = torch.nn.Linear(self.ontologyDimension,self.embeddingDimension).to(self.device)
        self.relu = torch.nn.ReLU().to(self.device)

        self.projectRelation = torch.nn.Linear(self.embeddingDimension,self.ontologyDimension).to(self.device)
        ###HYPE STUFF
        self.in_channels = 1
        self.out_channels = 2
        self.filt_h = 1
        self.filt_w = 10
        self.stride = 5
        self.hidden_drop_rate = 0.4
        self.max_arity = 5

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels).to(self.device)
        self.inp_drop = torch.nn.Dropout(0.4).to(self.device)

        fc_length = (1-self.filt_h+1)*math.floor((embeddingDimension-self.filt_w)/self.stride + 1)*self.out_channels

        self.bn2 = torch.nn.BatchNorm1d(fc_length).to(self.device)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate).to(self.device)
        # Projection network
        self.fc = torch.nn.Linear(fc_length, embeddingDimension).to(self.device)

        # size of the convolution filters outputted by the hypernetwork
        fc1_length = self.in_channels*self.out_channels*self.filt_h*self.filt_w
        # Hypernetwork
        self.fc1 = torch.nn.Linear(embeddingDimension + self.max_arity + 1, fc1_length).to(self.device)
        self.fc2 = torch.nn.Linear(self.max_arity + 1, fc1_length).to(self.device)

    def forward(self,input,currentBatchSize):
        '''
        data is ordered like this: 0'relation',1'relation2',2'userId',3'hour',4'day',5'type','location','ontology'
        targets are location and ontology !!!
        returns scores for all potential targets at once
        forward function is adapted to specifically this use case, where the relations and therefore entity positions within the relations never change
        '''
        ###transe input
        typeSub = self.ontologySpace(input[5])
        #simple case with only subordinate relation as available in data
        subRelation = self.relationSpace(input[1])
        subRelation = self.projectRelation(subRelation)
        subRelation = self.relu(subRelation)

        ###calculate transe score for typeSub, subRelation and ontologySpace
        typeSub2 = torch.unsqueeze(typeSub,1).repeat(1,self.nrOntology,1)
        subRelation = torch.unsqueeze(subRelation,1).repeat(1,self.nrOntology,1)
        ontScores = torch.norm(typeSub2 + (subRelation - self.ontologySpace.weight),1,-1)
        
        ###hype input
        relation = self.relationSpace(input[0])
        user = self.convolve(input[0],input[2],1,False)
        day = self.convolve(input[0],input[4],2,False)
        time = self.convolve(input[0],input[3],3,False)
        ###translate type from ontology space to entityspace
        translatedType = self.projection(typeSub)
        translatedType = self.relu(translatedType)
        ###convolve translated type
        translatedType = self.convolve(input[0],translatedType,5,True)
        translatedType = torch.unsqueeze(translatedType,1).repeat(1,self.nrLocations,1)
        ###calulcate hype score for user, translated type, day, time and location space
        user = torch.unsqueeze(user,1).repeat(1,self.nrLocations,1)
        day = torch.unsqueeze(day,1).repeat(1,self.nrLocations,1)
        time = torch.unsqueeze(time,1).repeat(1,self.nrLocations,1)
        relation = torch.unsqueeze(relation,1).repeat(1,self.nrLocations,1)
        #reshaping locations
        e5_idx = torch.arange(0,self.nrLocations,dtype=torch.long).to(self.device)
        dummyRelation = torch.zeros(self.nrLocations,dtype=torch.long).to(self.device)
        e5 = self.convolve(dummyRelation, e5_idx, 4,False)
        allLocations = torch.unsqueeze(e5,0).repeat(currentBatchSize,1,1)
        
        hypeScores = relation * user * day * time * allLocations * translatedType
        hypeScores = torch.sum(hypeScores, dim=2)
        return ontScores, hypeScores
    

    def convolve(self, r_idx, e_idx, pos, ontSpace):
        if ontSpace:
            e = e_idx.view(-1, 1, 1, self.entitySpace.weight.size(1)).to(self.device)
        else:
            e = self.entitySpace(e_idx).view(-1, 1, 1, self.entitySpace.weight.size(1)).to(self.device)
        r = self.relationSpace(r_idx).to(self.device)
        x = e
        x = self.inp_drop(x)
        one_hot_target = (pos == torch.arange(self.max_arity + 1).reshape(self.max_arity + 1)).float().to(self.device)
        poses = one_hot_target.repeat(r.shape[0]).view(-1, self.max_arity + 1)
        one_hot_target.requires_grad = False
        poses.requires_grad = False
        k = self.fc2(poses)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(e.size(0)*self.in_channels*self.out_channels, 1, self.filt_h, self.filt_w)
        x = x.permute(1, 0, 2, 3)
        x = F.conv2d(x, k, stride=self.stride, groups=e.size(0))
        x = x.view(e.size(0), 1, self.out_channels, 1-self.filt_h+1, -1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(e.size(0), -1)
        x = self.fc(x)
        return x

class Trainer:
    def __init__(self,trainData,validationData,testData,nrEpochs,learningRate,batchSize,ontWeighting=1):
        self.trainData = trainData
        self.validationData = validationData
        self.testData = testData
        self.nrEpochs = nrEpochs
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.ontWeighting = ontWeighting

    def trainModel(self,model):
        print ('Parameters:\n' + 'Embedding Dimension: ' +str(model.embeddingDimension) + '\tOntology Dimension: ' + str(model.ontologyDimension))
        print ('Epochs: ' + str(self.nrEpochs) + '\tLearningRate: ' + str(self.learningRate) + '\tBatchSize: ' + str(self.batchSize) + '\tontWeight: ' + str(self.ontWeighting) +'\n' + 'Training Data: ' + self.trainData + '\tTest Data: ' + self.testData + '\tValidation Data: ' + self.validationData)
        lossFunction = nn.CrossEntropyLoss()
        data = torch.load(self.trainData)
        optimizer = optim.Adagrad(model.parameters(),lr = self.learningRate)
        #used for output later
        bestLoss = 100
        bestEpoch = 0
        model.train()
        for j in range(1,self.nrEpochs+1):
            model.zero_grad()
            lossAbs = 0
            nrBatches = 0
            for i in range(0,len(data),self.batchSize):
                tmp = torch.stack(data[i:i+self.batchSize]).transpose(0,1).to(model.device)
                targetsLoc = tmp[6]
                targetsOnt = tmp[7]
                ontScores, locScores = model(tmp,tmp.shape[1])
                lossLocs = lossFunction(locScores,targetsLoc)
                lossOnt = lossFunction(ontScores,targetsOnt)
                lossTotal = lossLocs + (self.ontWeighting * lossOnt)
                lossTotal.backward()
                optimizer.step()
                nrBatches +=1
                lossAbs += float(lossTotal)
            if j % (self.nrEpochs/10) == 0:
                print("Loss in Epoch #"+str(j)+": "+str(lossAbs/nrBatches))
                valLoss = self.validateModel(model)
                print('Validation Loss:\t\t'+str(valLoss))
                if bestLoss > valLoss:
                    bestLoss = valLoss
                    bestEpoch = j
        print ("Best loss: "+str(bestLoss)+" in Epoch: "+str(bestEpoch))

    def validateModel(self,model):
        lossFunction = nn.CrossEntropyLoss()
        data = torch.load(self.validationData)
        model.eval()
        nrBatches = 0
        lossAbs = 0
        with torch.no_grad():
            for i in range(0,len(data),self.batchSize):
                tmp = torch.stack(data[i:i+self.batchSize]).transpose(0,1).to(model.device)
                targetsLoc = tmp[6]
                targetsOnt = tmp[7]
                ontScores, locScores = model(tmp,tmp.shape[1])
                lossLocs = lossFunction(locScores,targetsLoc)
                lossOnt = lossFunction(ontScores,targetsOnt)
                lossTotal = lossLocs + (self.ontWeighting * lossOnt)
                nrBatches +=1
                lossAbs += float(lossTotal)
            return lossAbs/nrBatches

    def testModel(self,model):
        print("~~~EVALUATION~~~")
        data = torch.load(self.testData)
        model.eval()
        hits1 = 0
        hits3 = 0
        hits10 = 0
        checkinCounter = 0
        hits100 = 0
        ranks = 0
        rRanks = 0
        with torch.no_grad():
            for i in range(len(data)):
                checkinCounter += 1
                tmp = data[i].unsqueeze(0).transpose(0,1).to(model.device)
                target = tmp[6]
                ontScores, locScores = model(tmp,1)
                rank = int((torch.topk(locScores.squeeze(),len(locScores.squeeze())).indices == target).nonzero().flatten()) + 1
                ranks += rank
                rRanks += 1/rank
                if target in torch.topk(locScores.squeeze(),1).indices:
                    hits1 += 1
                if target in torch.topk(locScores.squeeze(),3).indices:
                    hits3 += 1
                if target in torch.topk(locScores.squeeze(),10).indices:
                    hits10 += 1
                if target in torch.topk(locScores.squeeze(),100).indices:
                    hits100 += 1
        print ("Hits@1: " + str(hits1/checkinCounter))
        print ("Hits@3: " + str(hits3/checkinCounter))
        print ("Hits@10: " + str(hits10/checkinCounter))
        print ("Hits@100: " + str(hits100/checkinCounter))
        print ("MR: " + str(ranks/checkinCounter))
        print ("MRR: " + str(rRanks/checkinCounter) + '\n')