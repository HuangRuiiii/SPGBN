import sys
import os
import numpy as np
sys.path.append('./utils/liblinear-2.1/python/')
from liblinear import *

def predict( Y, ThetaFreqAver, args):
    train_indices = args.train_indices
    test_indices = args.test_indices
    # Calculate Accuracy
    AddBiasTerm = False
    CCstart = -10
    CCend = 15
    CCstep = 1

    ThetaTmp = ThetaFreqAver[0] / np.maximum(np.sum(ThetaFreqAver[0], axis = 0), 2.2204e-16)
 
    sparse_TrainX_Transpose = ThetaTmp[:, train_indices].T
    TrainY = Y[train_indices]
    problem_TrainY = list(TrainY)
    problem_TrainX = []
    for doc in sparse_TrainX_Transpose:
        problem_TrainX.append(list(doc))
    assert len(problem_TrainY) == len(problem_TrainX), "The numbers of train label and data don't match."
    prob = problem(problem_TrainY, problem_TrainX)

    sparse_TestX_Transpose = ThetaTmp[:, test_indices].T
    TestY = Y[test_indices]
    problem_TestY = list(TestY)
    problem_TestX = []
    for doc in sparse_TestX_Transpose:
        problem_TestX.append(list(doc))
    assert len(problem_TestY) == len(problem_TestX), "The numbers of test label and data don't match."
    
    # Cross validation over 26 sets of parameters
    CC = 2 ** np.arange(CCstart, CCend + CCstep, CCstep, dtype=float)
    ModelOut = np.zeros(len(CC))
    for ij in range(len(CC)):
        if AddBiasTerm == False:
            option = '-s 0 -c ' + str(CC[ij]) + ' -v 5 -q '
        else:
            option = '-B 1 -s 0 -c ' + str(CC[ij]) + ' -v 5 q '   
        param = parameter(option)
        model = liblinear.train(prob, param) # return a ctype pointer
        model = toPyModel(model)
        all_train_result = []
        for pTrx in problem_TrainX:
            train_ret, _ = gen_feature_nodearray(pTrx)
            train_result = liblinear.predict(model, train_ret)
            all_train_result.append(train_result)
        ModelOut[ij] = calculate_predict_accuracy(all_train_result, problem_TrainY)
        print(f"Cross Validation Accuracy = {ModelOut[ij]:.5f}")

    # Select the best param and do prediction for test data
    maxdex = np.argmax(ModelOut)
    if AddBiasTerm == False:
        option = '-s 0 -c ' + str(CC[maxdex]) + ' -q '
    else:
        option = '-B 1 -s 0 -c ' + str(CC[maxdex]) + ' q '
    param = parameter(option)
    model = liblinear.train(prob, param)
    model = toPyModel(model)
    all_test_result = []
    for pTex in problem_TestX:
        test_ret, _ = gen_feature_nodearray(pTex)
        test_result = liblinear.predict(model, test_ret)
        all_test_result.append(test_result)
    accuracy = calculate_predict_accuracy(all_test_result, problem_TestY)
    print(f'The predict accuracy: {accuracy:.5f}')

def calculate_predict_accuracy(predict_result, label):
    '''
    predict_result: list
    label: list
    '''
    assert len(predict_result) == len(label), "The numbers of predict results and labels don't match."
    predict_result = np.array(predict_result, dtype = int)
    label = np.array(label, dtype = int)
    correct_num = (predict_result == label).sum()
    accuracy = correct_num / len(label)
    return accuracy
