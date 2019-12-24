# B551 Assignment 3: Probability and Statistical Learning for NLP
##### Submission by Sri Harsha Manjunath - srmanj@iu.edu; Vijayalaxmi Bhimrao Maigur - vbmaigur@iu.edu; Disha Talreja - dtalreja@iu.edu
###### Fall 2019


## Part 1: Part-of-speech Tagging

Natural language processing (NLP) is an important research area in artificial intelligence, dating back to at least the 1950's. A basic problems in NLP is part-of-speech tagging, in which the goal is to mark every word in a sentence with its part of speech (noun, verb, adjective, etc.). This is a first step towards extracting semantics from natural language text. For example, consider the following sentence: \Her position covers a number of daily tasks common to any social director." Part-of-speech tagging here is not easy because many of these words can take on different parts of speech depending on context. For example, position can be a noun (as in the above sentence) or a verb (as in \They position themselves near the exit"). In fact, covers, number, and tasks can all be used as either nouns or verbs, while social and common can be nouns
or adjectives, and daily can be an adjective, noun, or adverb. The correct labeling for the above sentence is: Her position covers a number of daily tasks common to any social director. DET NOUN VERB DET NOUN ADP ADJ NOUN ADJ ADP DET ADJ NOUN where DET stands for a determiner, ADP is an adposition, ADJ is an adjective, and ADV is an adverb.1 Labeling parts of speech thus involves an understanding of the intended meaning of the words in the sentence, as well as the relationships between the words.

Fortunately, statistical models work amazingly well for NLP problems. Consider the Bayes net shown in Figure 1(a). This Bayes net has random variables 
S = {S1, . . . , SN} and W = {W1, . . . .WN}. The W's represent observed words in a sentence. The S's represent part of speech tags, so Si = {VERB,NOUN, . . .} The arrows between W and S nodes model the relationship between a given observed word and the possible parts of speech it can take on, P(Wi|Si). (For example, these distributions can model the fact that the word \dog" is a fairly common noun but a very rare verb.) The arrows between S nodes model the probability that a word of one part of speech follows a word of another part of speech, P(Si+1|Si). (For example, these arrows can model the fact that verbs are very likely to follow nouns, but are unlikely to follow adjectives.)

### Solution:

The problem is modelled as bayes net with
1. Hidden Variable : Parts of Speech Label for every word in a sentence
2. Observed Variable: Words in a sentence

#### Approach 1: Simple Bayes Net:

![alt text](https://github.iu.edu/cs-b551-fa2019/vbmaigur-srmanj-dtalreja-a3/blob/master/images/simple_bayes_net_1.png)

Training Part: Here, We have to find two argmax(P(S</sup><sub>i</sub>|W</sup><sub>i</sub>)) equals to P(W</sup><sub>i</sub>|S</sup><sub>i</sub>)*P(S</sup><sub>i</sub>)

##### P(S</sup><sub>i</sub>):
- DataType: dictionary
- Key: Part of Speech(POS)
- Value: Probability value of the POS
    - ###### Probability = number of occurance of POS / number of words
- Length: Number of predefined POS
- Name of Variable in code: self.initial_prob

```
{'adj': 0.06563706563706563,
 'adv': 0.023166023166023165,
 'adp': 0.11583011583011583,
 'conj': 0.03861003861003861,
 'det': 0.13513513513513514,
 'noun': 0.2702702702702703,
 'num': 0.003861003861003861,
 'pron': 0.023166023166023165,
 'prt': 0.011583011583011582,
 'verb': 0.1776061776061776,
 'x': 0.0,
 '.': 0.13513513513513514}
 ```

##### P(W</sup><sub>i</sub>|S</sup><sub>i</sub>):

- DataType: dictionary
- Key: Part of Speech(POS)
- Value: dictionary
    - Key: Word
    - Value: Probability of the word given the POS
        - ###### Probability = number of occurance of that word as POS / number of occurance of POS
- Length: Number of predefined POS
- Name of Variable in code: self.emiision_prob

```
{'adj': 
{'executive': 0.058823529411764705,
  'over-all': 0.058823529411764705,
  'superior': 0.058823529411764705,
  'possible': 0.058823529411764705,
  'hard-fought': 0.058823529411764705,
  'relative': 0.058823529411764705,
  'such': 0.058823529411764705,
  'widespread': 0.058823529411764705,
  'many': 0.058823529411764705,
  'outmoded': 0.058823529411764705,
  'inadequate': 0.058823529411764705,
  'ambiguous': 0.058823529411764705,
  'grand': 0.058823529411764705,
  'other': 0.058823529411764705,
  'best': 0.058823529411764705,
  'greater': 0.058823529411764705,
  'clerical': 0.058823529411764705}}
```

Prediction on Test Data: 
```
- for every word in a sentence
    - for every POS
        - find P(S</sup><sub>i</sub>|W</sup><sub>i</sub>)=P(W</sup><sub>i</sub>|S</sup><sub>i</sub>)*P(S</sup><sub>i</sub>)
    - the prediction for the word is the pos with max probability for the word
```

Finding the posterior, given a sentence and corresponding label

- sum of P(S</sup><sub>i</sub>|W</sup><sub>i</sub>) of all (word, pos) pair for a sentence

#### Approach 2: HMM Viterbi Algorithm:

![alt text](https://github.iu.edu/cs-b551-fa2019/vbmaigur-srmanj-dtalreja-a3/blob/master/images/hmm_viterbi.png)

Training Part: Here, We have to find two P(S</sup><sub>i</sub>|W</sup><sub>i</sub>) equals to P(W</sup><sub>i</sub>|S</sup><sub>i</sub>)*argmax(P(previous_state)*P(S</sup><sub>i+1</sub>|S</sup><sub>i</sub>))

##### P(W</sup><sub>i</sub>|S</sup><sub>i</sub>):

- explained above

##### P(S</sup><sub>i+1</sub>,S</sup><sub>i</sub>):

- DataType: dictionary
- Key: Part of Speech(POS) (state t-1)
- Value: dictionary
    - Key: POS (state t)
    - Value: Probability of the word given the POS
        - ###### Probability = number of occurance of POS at t-1 and t / number of joint occurance of POS
- Length: Number of predefined POS
- Name of Variable in code: self.transition_prob

```
{'adj': {'noun': 0.048, '.': 0.008, 'adp': 0.004, 'conj': 0.008},
 'adv': {'verb': 0.012, 'det': 0.004, 'adj': 0.004, '.': 0.004}}
 ```

 ##### P(previous_state):

 - Probability of the previous being a particular POS (no need to calculate using traning data handled in the implementation of algo)


 Prediction on Test Data:
```
 - for word in a sentence:
    - for every pos
        - prob of first word being pos is P(W</sup><sub>i</sub>|S</sup><sub>i</sub>) (store all values in a dictionary with key as pos and value is a tuple( prob, path_traversed_till_now))

        - prob of every other word other than first:
        - for every prev_pos
            - product of Transition probability and previous state prob
            - store in a different dict 
        - take the max of the above prob and multiply with emission probability
- take the max prob of the last word and the path attached to it
```

Finding the posterior, given a sentence and corresponding label

- sum of P(S</sup><sub>i</sub>|W</sup><sub>i</sub>)=P(W</sup><sub>i</sub>|S</sup><sub>i</sub>)* P(S</sup><sub>i+1</sub>|S</sup><sub>i</sub>) of all (word, pos) pair for a sentence


#### Simple illustration of the hmm-viterbi algorithm for POS
##### [ref]-http://www.phontron.com/slides/nlp-programming-en-04-hmm.pdf

![alt text](https://github.iu.edu/cs-b551-fa2019/vbmaigur-srmanj-dtalreja-a3/blob/master/images/re.png)

#### Approach 3: Gibbs Sampling:

![alt text](https://github.iu.edu/cs-b551-fa2019/vbmaigur-srmanj-dtalreja-a3/blob/master/images/gibbs.png)

Training Part: Here, We have to find two P(S</sup><sub>i</sub>|W</sup><sub>i</sub>) equals to P(W</sup><sub>i</sub>|S</sup><sub>i</sub>)*argmax(P(previous_state)*P(S</sup><sub>i+1</sub>|S</sup><sub>i</sub>))

##### P(W</sup><sub>i</sub>|S</sup><sub>i</sub>):

- explained above

##### P(S</sup><sub>i+1</sub>,S</sup><sub>i</sub>):

- explained above

##### P(S</sup><sub>n</sub>|S</sup><sub>0</sub>):

- DataType: dictionary
- Key: Part of Speech(POS) (state 0)
- Value: dictionary
    - Key: POS (state n)
    - Value: Probability of the word given the POS
        - ###### Probability = number of occurance of POS at t-1 and t / number of joint occurance of POS
- Length: Number of predefined POS
- Name of Variable in code: self.gibbs_transition_prob

```
{'adj': {},
 'adv': {'.': 0.1111111111111111},
 'adp': {},
 'conj': {},
 'det': {'.': 0.5555555555555556},
 'noun': {'verb': 0.1111111111111111},
 'num': {},
 'pron': {'.': 0.1111111111111111},
 'prt': {},
 'verb': {},
 'x': {},
 '.': {'.': 0.1111111111111111}}
```

Prediction on Test Data:

```
- first randomly initialized state is output of simple bayes net
- gibbs_algo, genearates the sample label based on the label that is passed to the function
- gibbs algo function
    - for every word 
        - for every pos
            - use the same implementation as hmm viterbi to calculate the prob of all words expect last word
            - for last word prob multiply extra gibbs transition prob according to the given bayes net
        
        random distribution is used to select a pos for a word
    - return the sample
- complex function
    - burn out first few iterations~ 200
    - collect the samples for around 500 iteration 
    - the predicted pos is the one with high frequency in 500 samples for that word.

```

Finding the posterior, given a sentence and corresponding label

- sum of P(S</sup><sub>i</sub>|W</sup><sub>i</sub>)=P(W</sup><sub>i</sub>|S</sup><sub>i</sub>)* P(S</sup><sub>i+1</sub>|S</sup><sub>i</sub>) of all (word, pos) pair for a sentence except last word
- for last word it is P(S</sup><sub>i</sub>|W</sup><sub>i</sub>)=P(W</sup><sub>i</sub>|S</sup><sub>i</sub>)* P(S</sup><sub>i+1</sub>|S</sup><sub>i</sub>) * P(S</sup><sub>n</sub>|S</sup><sub>0</sub>)

#### Assumption:

1. To numerical instability, we have considered log of the probabilities in our calculation
2. In the absence of any probability value, we have used very small prob to avoid log(0) error.
3. In finding transition probability during implementing the algo, we have used the below formula
    P(S</sup><sub>i+1</sub>|S</sup><sub>i</sub>)=P(S</sup><sub>i+1</sub>,S</sup><sub>i</sub>)/P(S</sup><sub>i</sub>)
    

#### Accuracy:

![alt text](https://github.iu.edu/cs-b551-fa2019/vbmaigur-srmanj-dtalreja-a3/blob/master/images/Accuracy.png)

**References** - </br>

[1] - https://www.freecodecamp.org/news/a-deep-dive-into-part-of-speech-tagging-using-viterbi-algorithm-17c8de32e8bc/
