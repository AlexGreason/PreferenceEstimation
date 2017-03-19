import copy
import numpy as np
import numpy.random as rand
import sklearn.linear_model as lin
import theano.tensor as T
import statsmodels.api as sm
from math import e
from time import time
#https://github.com/lucasmaystre/choix
#A python package for study of choice models
#has bayesian models
#might be useful

# logistic regression with n classes for n teams/items
# when A and B compete, A is given value 1, B is given -1, and all others are 0
# target output is 1 if A wins, and 0 if B wins
# the weights learned for each item is its "worth" in log-odds
# I'm sure I can find out how to find the standard errors for the parameters in logistic regression

# alternately, constant response (1), and the winner is assigned 1 and the loser is assigned -1? no threshold, to clarify
class BTModel:
    #I need some dummy comparisons for all elements to prevent nonconvergence
    #Maybe add a dummy element and give them all a tied game? that seems standard.
    #could the dummy element be the implied element? but if the error of all the real elements is bounded below by the error
    #on the implied element, and I never add data about the implied element, then the errors wouldn't go to 0

    #but their ordering might still be correct, and that's all I really care about for the actual implementation
    #actual values would be nice for comparing different question-asking schemes and the comparator, though
    def __init__(self, data, numvals = -1):
        self.data = data
        self.numvals = max(numvals, len(data[0]))
        self.worths = [0]*numvals
        self.errors = [1]*numvals
        self.questionOrder = np.random.permutation(np.arange(numvals+1))
        self.questionNum = -1

    def addData(self, data):
        self.data += data
        self.worths = [0] * self.numvals
        self.errors = [1] * self.numvals

    def fit(self):
        model = sm.Logit(np.array([1]*len(self.data)), np.array(self.data))
        values = model.fit()
        self.worths = values.params
        self.errors = values.bse

    def addDummyData(self):
        #add (fractional) tied games for making-the-linear-algebra-work-properly reasons
        #could add a dummy element for this, could also just use the first element.I'll use the first element for now
        for i in range(len(self.worths)):
            comp1 = [0]*len(self.worths)
            comp2 = [0]*len(self.worths)
            comp1[i] = .1
            comp2[i] = -.1
            self.addData([comp1, comp2])

    @staticmethod
    def genData(truevals, size):
        truevals = [0] + list(truevals)
        data = []
        numteams = len(truevals)
        data.append([0]*(numteams-1))
        for x in range(size):
            team1, team2 = rand.choice(np.arange(numteams), size=2, replace=False)
            prob = 1/(e**(truevals[team2] - truevals[team1]) + 1)
            row = [0]*numteams
            if rand.random() < prob:
                row[team1] = 1
                row[team2] = -1
            else:
                row[team1] = -1
                row[team2] = 1
            data.append(row[1:])
        return data

    def sortchars(self):
        self.fit()
        self.questionOrder = []

    def genQuestion(self):
        #can never ask about the implied element, and I don't have an error for the implied element because the
        #uncertainty in its value just gets spread out across the uncertainty in all the other values

        #this would be solved if I could represent the implied element explicitly, and just apply the constraint
        #that all params have to sum to 0.

        #could also be solved if I could find a way to determine the error of that element that would let it be the
        #most or least certain when appropriate

        #If you had infinite data for all comparisons not with the implied element, would the standard error of all the
        #other elements be the standard error of the implied element?

        #but if I just take the mean it'll never be the min or the max, and if I just simulate a bunch of data it'll never
        #be the max.

        #maybe I should just ask about the implied element on the first question, and then randomly with frequency 1/numelements

        #make it much more likely to ask about things that haven't come up yet
        #Maybe shuffle the list, then ask in order about a vs b then b vs c and so on until the end, then shuffle again?
        #or a vs b followed by c vd d, then shuffle the list at the end

        # plist = np.array([np.mean(self.errors)] + list(self.errors))
        # plist += .01
        # plist /= np.sum(plist)
        # print(plist)
        # return list(np.random.choice(len(self.worths)+1, size=2, replace=False, p=plist))
        if self.questionNum < len(self.questionOrder) - 2:
            self.questionNum += 2
            return (self.questionOrder[self.questionNum - 1], self.questionOrder[self.questionNum])
        else:
            self.questionNum = -1
            self.questionOrder = np.random.permutation(self.questionOrder)
            return self.genQuestion()

if __name__ == "__main__":
    model = BTModel([])
    starttime = time()
    truevals = rand.random(size=3)
    model.data = BTModel.genData(truevals, 10)
    print("generate data time: ", time()-starttime)
    newtime = time()
    model.fit()
    print("fit time: ", time()-newtime)
    print(truevals, (truevals.mean()-model.worths.mean()))
    print(model.worths, model.errors)