# Zoo Classification Project

**Project Objectives**
The goal of this project is to classify animals into one of seven classes using their features.

## Classifier Performance with MLP and k-N
**MLP Classifier**
Accuracy: Approximately 95%
Precision and Recall: Generally high, indicating good performance in both identifying positive examples and minimizing false negatives.
**k-NN Classifier**
Accuracy: Approximately 87%
Precision and Recall: Precision values vary depending on the task and chosen k value. Generally, precision is slightly lower compared to MLP.

**Predicting Class of a New Instance**
Both MLP and k-NN classifiers were used to predict the class of a new animal instance based on given features. Predictions from both classifiers are provided.

**10-Fold Cross Validation**
MLP Classifier: Evaluated using 10-fold cross-validation, showing metrics for accuracy, precision, and recall.
k-NN Classifier: Also evaluated using 10-fold cross-validation, with results for accuracy, precision, and recall.

**Summary of Classification Results**
**MLP Classifier**
Accuracy: Superior compared to k-NN, with nearly flawless performance.
Precision and Recall: High values indicate effectiveness in classifying positive examples and minimizing false negatives.
**k-NN Classifier**
Accuracy: Slightly lower than MLP, but still performs well.
Precision and Recall: Varies by k value, generally lower precision compared to MLP.
Overall, the MLP classifier demonstrates better performance for this classification problem, offering higher accuracy and precision. For cases where higher accuracy and precision are required, MLP is a better choice. For more flexibility and different performance balances, k-NN could be considered.

## Dataset
**Dataset Link:** [UCI Zoo Dataset](https://archive.ics.uci.edu/dataset/111/zoo)

**Dataset Information:**
- **Objective:** Identifying the class of an animal based on its characteristics by obtaining information.
- **Number of Instances:** 101
- **Number of Features:** 16

**Attributes:**

**Role (ID):**
- animal name (textual, unique)

**Role (Feature):**
- hair (boolean, yes/no)
- feathers (boolean, yes/no)
- eggs (boolean, yes/no)
- milk (boolean, yes/no)
- airborne (boolean, yes/no)
- aquatic (boolean, yes/no)
- predator (boolean, yes/no)
- toothed (boolean, yes/no)
- backbone (boolean, yes/no)
- breathes (boolean, yes/no)
- venomous (boolean, yes/no)
- fins (boolean, yes/no)
- number of legs (numerical, one of 0, 2, 4, 5, 6, or 8)
- tail (boolean, yes/no)
- domestic (boolean, yes/no)
- cat-sized (boolean, yes/no)

**Number of Classes:** 7

**Class Labels:**
1. **Mammal** (41 animals: aardvark, antelope, bear, boar, buffalo, calf, cavy, cheetah, deer, dolphin, elephant, fruitbat, giraffe, girl, goat, gorilla, hamster, hare, leopard, lion, lynx, mink, mole, mongoose, opossum, oryx, platypus, polecat, pony, porpoise, puma, pussycat, raccoon, reindeer, seal, sealion, squirrel, vampire, vole, wallaby, wolf)
2. **Bird** (20 animals: chicken, crow, dove, duck, flamingo, gull, hawk, kiwi, lark, ostrich, parakeet, penguin, pheasant, rhea, skimmer, skua, sparrow, swan, vulture, wren)
3. **Reptile** (5 animals: pitviper, seasnake, slowworm, tortoise, tuatara)
4. **Fish** (13 animals: bass, carp, catfish, chub, dogfish, haddock, herring, pike, piranha, seahorse, sole, stingray, tuna)
5. **Amphibian** (4 animals: frog, frog, newt, toad)
6. **Insect** (8 animals: flea, gnat, honeybee, housefly, ladybird, moth, termite, wasp)
7. **Invertebrate** (10 animals: clam, crab, crayfish, lobster, octopus, scorpion, seawasp, slug, starfish, worm)



