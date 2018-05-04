from node import Node
import math
import copy

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  classList = [example['Class'] for example in examples]  # get classList

  mynode = Node()
  if(examples == None):# example is empty
    mynode.label = default
    mynode.MODE = default
    return mynode
  elif (classList.count(classList[0]) == len(classList) or len(examples[0]) == 1):# have same classification or no non-trivial splits are possible
    mynode.label = MODE(examples)
    mynode.MODE = MODE(examples)
    return mynode
  else:
    best = Choose_Attribute(examples)
    bestFeature = [example[str(best)] for example in examples]
    examples_2 = copy.deepcopy(examples)
    t = Node()
    t.label = best
    t.MODE = MODE(examples)
    for value_i in set(bestFeature):
        examples_i = []
        for example in examples:
            if (value_i == example[best]):
                examples_i.append(copy.deepcopy(example))
        for example in examples_i:
            del example[str(best)]
        subtree = ID3(examples_i, MODE(examples))
        t.children.update({value_i: subtree})
    return t


def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  # node stroed a tree
  originalAcc = test(node, examples)
  # traval all node, to know which node we can remove to increase the acc most.
  #if we remove a node, this node will become to a leaf node, change the value of label
  # the value of label equals to MODE(), so we have to know the MODE value of each node.
  # node.children = {}
  '''the first step: BFS to travase this tree'''
  queue = []
  queue.append(node)
  nodeCut = None
  while queue:
      nodeCurrent = queue.pop(0)
      if nodeCurrent.children != {}:
          for i in nodeCurrent.children:
              nodeChild = nodeCurrent.children[i]
              tempLabel = nodeChild.label
              tempChildren = nodeChild.children
              nodeChild.label = nodeChild.MODE
              nodeChild.children = {}
              acc = test(node, examples)
              if(acc > originalAcc):
                  originalAcc = acc
                  nodeCut = nodeChild

              else:
                  nodeChild.label = tempLabel
                  nodeChild.children = tempChildren
              queue.append(nodeChild)
  if(nodeCut != None):
      prune(node, examples)



def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  count = 0
  for example in examples:
      predictLabel = evaluate(node, example)
      if(predictLabel == example['Class']):
          count = count + 1
  return float(count) / len(examples)



def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  if(node.children == {}):
      return node.label
  else:
      if example[node.label] in node.children.keys():
          p2 = node.label
          p1 = example[p2]
          return evaluate(node.children[p1], example)
      else:
          return node.MODE
  #return classLabel

def Choose_Attribute(examples):
    classList = [example['Class'] for example in examples]  # get classList
    exampleList = example.keys()
    examples_noclass = copy.deepcopy(examples)
    for example in examples_noclass:
        del example['Class']
    min_entropy = 100.0
    for i in range(len(exampleList) - 1):
        attribute_i = [example.values()[i] for example in examples_noclass]
        countAttributeNum = []
        countClassNum = []
        for value in set(attribute_i):#calculate entropy
            countAttributeNum.append(countAttribute(examples_noclass, i, value))
            for classLabel in set(classList):
                countClassNum.append(countClass(examples, i, value, classLabel, examples_noclass))
        entro = Entropy(countAttributeNum, countClassNum, examples)
        if(entro < min_entropy):
            min_entropy = entro
            bestFeatureLabel = example.keys()[i]
            bestFeature = i
    return bestFeatureLabel

def Entropy(countAttributeNum, countClassNum, examples):
    classList = [example['Class'] for example in examples]  # get classList
    Entro_total = 0.0
    for i in range(len(countAttributeNum)):
        Entro = 0.0
        for j in range(len(set(classList))):
            p_ij = float(countClassNum[j]) / countAttributeNum[i]
            if(p_ij == 0):
                continue
            Entro -= p_ij * math.log(p_ij, 2)
        Entro_total = Entro_total + float(countAttributeNum[i]) / len(classList) * Entro
        countClassNum = countClassNum[len(set(classList)):]
    return float(Entro_total)

def countClass(examples, feature, value, classLabel, examples_noclass):
    num = 0
    i = 0
    classList = [example['Class'] for example in examples]  # get classList
    exampleList = [example.values() for example in examples_noclass]
    for example in exampleList:
        if(example[feature] == value and classList[i] == classLabel):
            num = num + 1
        i = i + 1
    return num


def countAttribute(examples, feature, value):
    num = 0
    exampleList = [example.values() for example in examples]
    for example in exampleList:
        if (example[feature] == value):
            num = num + 1
    return num

def MODE(examples):
    classList = [example['Class'] for example in examples]  # get classList
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return (sorted(classCount, key=lambda x: classCount[x])[-1])
