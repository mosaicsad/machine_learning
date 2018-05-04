import ID3, parse, random
import numpy as np
import matplotlib.pyplot as plt

def testID3AndEvaluate():
  data = [dict(a=1, b=0, Class=1), dict(a=1, b=1, Class=1)]
  tree = ID3.ID3(data, 0)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=1, b=0))
    if ans != 1:
      print "ID3 test failed."
    else:
      print "ID3 test succeeded."
  else:
    print "ID3 test failed -- no tree returned"

def testPruning():
  data = [dict(a=1, b=1, c=1, Class=0), dict(a=1, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1), dict(a=0, b=0, c=0, Class=1), dict(a=0, b=0, c=1, Class=0)]
  validationData = [dict(a=0, b=0, c=1, Class=1)]
  tree = ID3.ID3(data, 0)
  ID3.prune(tree, validationData)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=0, b=1, c=1))
    if ans != 1:
      print "pruning test failed."
    else:
      print "pruning test succeeded."
  else:
    print "pruning test failed -- no tree returned."


def testID3AndTest():
  trainData = [dict(a=1, b=0, c=0, Class=1), dict(a=1, b=1, c=0, Class=1), 
  dict(a=0, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1)]
  testData = [dict(a=1, b=0, c=1, Class=1), dict(a=1, b=1, c=1, Class=1), 
  dict(a=0, b=0, c=1, Class=0), dict(a=0, b=1, c=1, Class=0)]
  tree = ID3.ID3(trainData, 0)
  fails = 0
  if tree != None:
    acc = ID3.test(tree, trainData)
    if acc == 1.0:
      print "testing on train data succeeded."
    else:
      print "testing on train data failed."
      fails = fails + 1
    acc = ID3.test(tree, testData)
    if acc == 0.75:
      print "testing on test data succeeded."
    else:
      print "testing on test data failed."
      fails = fails + 1
    if fails > 0:
      print "Failures: ", fails
    else:
      print "testID3AndTest succeeded."
  else:
    print "testID3andTest failed -- no tree returned."

# inFile - string location of the house data file
def testPruningOnHouseData(inFile):
  withPruning = []
  withPruningAve = []
  withoutPruning = []
  withoutPruningAve = []
  data = parse.parse(inFile)
  for num in range(10,300):
      for i in range(100):
        random.shuffle(data)
        train = data[:int(0.7*num)]
        valid = data[int(0.7*num):num]
        test = data[num:]

        tree = ID3.ID3(train, 'democrat')
        acc = ID3.test(tree, train)
        #print "training accuracy: ",acc
        acc = ID3.test(tree, valid)
        #print "validation accuracy: ",acc
        acc = ID3.test(tree, test)
        #print "test accuracy: ",acc

        ID3.prune(tree, valid)
        acc = ID3.test(tree, train)
        #print "pruned tree train accuracy: ",acc
        acc = ID3.test(tree, valid)
        #print "pruned tree validation accuracy: ",acc
        acc = ID3.test(tree, test)
        #print "pruned tree test accuracy: ",acc
        withPruning.append(acc)
        tree = ID3.ID3(train+valid, 'democrat')
        acc = ID3.test(tree, test)
        #print "no pruning test accuracy: ",acc
        withoutPruning.append(acc)
      #print withPruning
      #print withoutPruning
      print "average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning)
      withPruningAve.append(sum(withPruning)/len(withPruning))
      withoutPruningAve.append(sum(withoutPruning)/len(withoutPruning))
  plt.title('Result')
  plt.plot(range(10,300), withPruningAve, label = "accuracy withPruning")
  plt.plot(range(10,300), withoutPruningAve, label="accuracy withoutPruning")
  plt.legend(loc = 'best')
  plt.xlabel('training set sizes')
  plt.ylabel('accuracy')
  plt.show()


#testPruning()
testPruningOnHouseData('house_votes_84.data')