###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids: Vijayalaxmi B Maigur, Sri Harsha Manjunath, Disha Talreja 
# user ids: vbmaigur, srmanj, dtalreja
#
# (Based on skeleton code by D. Crandall)
#

import random
import math
import collections
from copy import deepcopy

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!

    def __init__(self):
        self.initial_prob={}
        self.transition_prob={}
        self.emission_prob={}
        self.gibbs_transition_prob={}
        self.POS=[]

    def posterior(self, model, sentence, label):
        # calculating P(POS|sentence)=(P(w1|s1)*P(s1)) * (P(w2|s2)*P(s2)) * ...
        if model == "Simple":   
            posterior_val=0
            for word,pos_ in zip(sentence,label):
                initial_prob=math.log(self.initial_prob.get(pos_,1.0/100000000))
                emission_prob=math.log(self.emission_prob.get(pos_,{}).get(word,1.0/100000000))
                posterior_val+=(initial_prob+emission_prob)  
            return posterior_val

        elif model == "Complex":

            # P(S|W) = P(W1|S1)*P(S1) for 1st word
            # P(S|W) = P(Sn|Sn-1)*P(Sn|s0)*P(Wn|Sn) for last word
            # P(S|W) = P(Sn|Sn-1)*P(Wn|Sn) for rest of the words
            posterior_val=0
            for index,word in enumerate(sentence):
                pos_curr=label[index]
                pos_prev=label[index-1]
                if index==0:
                    initial_prob=math.log(self.initial_prob.get(pos_curr,1.0/100000000))
                    emission_prob=math.log(self.emission_prob.get(pos_curr,{}).get(word,1.0/100000000))
                    posterior_val+=(initial_prob+emission_prob)
                elif index==len(sentence)-1:
                    transition_prob=math.log(self.transition_prob.get(pos_prev,{}).get(pos_curr,1.0/100000000))-math.log(self.initial_prob.get(pos_prev,1.0/100000000))
                    emission_prob=math.log(self.emission_prob.get(pos_curr,{}).get(word,1.0/100000000))
                    gib_prob=math.log(self.gibbs_transition_prob.get(label[0]).get(pos_curr,1.0/100000000))-math.log(self.initial_prob.get(label[0],1.0/100000000))
                    posterior_val+=(transition_prob+emission_prob+gib_prob)
                else: 
                    emission_prob=math.log(self.emission_prob.get(pos_curr,{}).get(word,1.0/100000000))
                    transition_prob=math.log(self.transition_prob.get(pos_prev,{}).get(pos_curr,1.0/100000000))-math.log(self.initial_prob.get(pos_prev,1.0/100000000))
                    posterior_val+=(emission_prob+transition_prob)

            return posterior_val

        elif model == "HMM":
            
            #calculation P(POS'|sentence)=(P(w1|s)p(s)) * (P(w2|w1))*P(w2|s)) * ...
            posterior_val=0
            for index,word in enumerate(sentence):
                pos_curr=label[index]
                pos_prev=label[index-1]
                if index==0:
                    initial_prob=math.log(self.initial_prob.get(pos_curr,1.0/100000000))
                    emission_prob=math.log(self.emission_prob.get(pos_curr,{}).get(word,1.0/100000000))
                    posterior_val+=(initial_prob+emission_prob)
                else:
                    emission_prob=math.log(self.emission_prob.get(pos_curr,{}).get(word,1.0/100000000))
                    transition_prob=math.log(self.transition_prob.get(pos_prev,{}).get(pos_curr,1.0/100000000))-math.log(self.initial_prob.get(pos_prev,1.0/100000000))
                    posterior_val+=(emission_prob+transition_prob)
            return posterior_val
        
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        # Finding the probabilities

        #defining all POS
        self.POS=["adj","adv","adp","conj","det","noun","num","pron","prt","verb","x","."]

        
        #calculating counts for initial, transition and emission probabilty

        initial_prob={}
        emission_prob={}
        transition_prob={}
        gibbs_transition_prob={}
        for i in self.POS:
            initial_prob[i]=0
            emission_prob[i]=collections.defaultdict(lambda : 0)
            transition_prob[i]=collections.defaultdict(lambda : 0)
            gibbs_transition_prob[i]=collections.defaultdict(lambda : 0)

        number_of_words=0
        num_t_prob=0
        num_g_prob=0
        for words,pos in data:
            for j in range(len(words)):
                number_of_words+=1
                initial_prob[pos[j]]+=1
                emission_prob[pos[j]][words[j]]+=1
                if j!=0:
                    num_t_prob+=1
                    transition_prob[pos[j-1]][pos[j]]+=1
                if j==len(words)-1:
                    num_g_prob+=1
                    gibbs_transition_prob[pos[0]][pos[j]]+=1   

        # initial Probability
        self.initial_prob={key:val/number_of_words for key,val in initial_prob.items()}

        # Emission Probability
        for key,val in emission_prob.items():
            emission_prob[key]=dict(val)
            for key_in,val_in in val.items():
                emission_prob[key][key_in]=val_in/initial_prob[key]
        
        self.emission_prob=emission_prob

        # transition probability
        for (key_s,val_s) in transition_prob.items():
            transition_prob[key_s]=dict(val_s)
            for (key_s_in,val_s_in) in val_s.items():
                transition_prob[key_s][key_s_in]=val_s_in/num_t_prob
        self.transition_prob=transition_prob
        #print("TP:",transition_prob)

        # Gibbs probability
        for (key_g,val_g) in gibbs_transition_prob.items():
            gibbs_transition_prob[key_g]=dict(val_g)
            for (key_g_in,val_g_in) in val_g.items():
                gibbs_transition_prob[key_g][key_g_in]=val_g_in/num_g_prob
        self.gibbs_transition_prob=gibbs_transition_prob
        #print("GB:",gibbs_transition_prob)
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        listOfPOS=[]
        #iterating through every word in a sentence
        for word in sentence:
            listOfProb=[]
            # finding prob of word with every single POS
            #assumption: if there is no such word in the training set, returns very small prob

            for pos in self.POS:
                initial_prob=math.log(self.initial_prob.get(pos,1.0/100000000))
                emission_prob=math.log(self.emission_prob.get(pos,{}).get(word,1.0/100000000))
                listOfProb.append((initial_prob+emission_prob,pos))
            listOfPOS.append(self.POS[listOfProb.index(max(listOfProb))])
        return listOfPOS



    def complex_mcmc(self, sentence):
        sample_label=self.simplified(sentence)
        discard_iter=100
        num_of_iterations=400
        listofsamples=[]

        # Burning the first few iterations to reach the stable state
        for i in range(discard_iter):
            sample_label=self.gibbs_algo(sentence,sample_label)

        #collecting the samples
        for i in range(num_of_iterations):
            updated_sample=self.gibbs_algo(sentence,sample_label)
            listofsamples.append(updated_sample)
        
        #Calculating the freq of diff POS' for words in a sentence
        dict_freq=collections.defaultdict(lambda : 0)
        for j in range(len(sentence)):
            dict_freq[j]=collections.defaultdict(lambda : 0)
            for i in range(len(listofsamples)):
                dict_freq[j][listofsamples[i][j]]+=1

        #getting the max freq POS for every word
        listofPOS=[]
        for i in range(len(sentence)):
            key=list(dict_freq[i].keys())
            val=list(dict_freq[i].values())
            listofPOS.append(key[val.index(max(val))])
        
        return listofPOS



    def gibbs_algo(self,sentence,sample_label):
        sample_label_mod=deepcopy(sample_label)
        prob_POS=collections.defaultdict(lambda : 0)
        for word_num in range(len(sentence)):
            prob_POS[word_num]=collections.defaultdict(lambda : 0)
            for pos in self.POS:
                sample_label_mod[word_num]=pos
                word=sentence[word_num]
                pos_prev=sample_label_mod[word_num-1]
                if word_num==0:
                    initial_prob=math.log(self.initial_prob.get(pos,1.0/100000000))
                    emission_prob=math.log(self.emission_prob.get(pos,{}).get(word,1.0/100000000))
                    prob_POS[word_num][pos]=math.exp(initial_prob+emission_prob)
                elif word_num==len(sentence)-1:
                    transition_prob=math.log(self.transition_prob.get(pos_prev,{}).get(pos,1.0/100000000))-math.log(self.initial_prob.get(pos_prev,1.0/100000000))
                    emission_prob=math.log(self.emission_prob.get(pos,{}).get(word,1.0/100000000))
                    gib_prob=math.log(self.gibbs_transition_prob.get(sample_label_mod[0]).get(pos,1.0/100000000))-math.log(self.initial_prob.get(sample_label_mod[0],1.0/100000000))
                    prob_POS[word_num][pos]=math.exp(transition_prob+transition_prob+gib_prob)
                else:
                    transition_prob=math.log(self.transition_prob.get(pos_prev,{}).get(pos,1.0/100000000))-math.log(self.initial_prob.get(pos_prev,1.0/100000000))
                    emission_prob=math.log(self.emission_prob.get(pos,{}).get(word,1.0/100000000))
                    prob_POS[word_num][pos]=math.exp(transition_prob+emission_prob)
            
            #normalizing the probability values due to small values
            
            #initializing the sum to small number if its zero
            if sum(prob_POS[word_num].values())==0:
                sum_den=1.0/100000000
            else:
                sum_den=sum(prob_POS[word_num].values())
            
            prob_POS[word_num]={key: value/sum_den for key,value in prob_POS[word_num].items()}

            #random selection of the POS based on random value generation
            sum_prob=0
            for pos in self.POS:
                prob=prob_POS[word_num][pos]
                sum_prob+=prob
                if random.random()<sum_prob:
                    sample_label_mod[word_num]=pos
                    break
                    
        return sample_label_mod
        #return ["noun"]*len(sentence)


    def hmm_viterbi(self, sentence):

        viterbi_prob=[]

        for (index,word) in enumerate(sentence):
            word_prob={}
            for pos in self.POS:
                if index==0:
                    initial_prob=math.log(self.initial_prob.get(pos,1.0/100000000))
                    emission_prob=math.log(self.emission_prob.get(pos,{}).get(word,1.0/100000000))
                    word_prob[pos]=(initial_prob+emission_prob,pos)
                else:
                    listOfProb=[]
                    strOfPOS=[]
                    for pos_prev in self.POS:
                        transition_prob=math.log(self.transition_prob.get(pos_prev,{}).get(pos,1.0/100000000))-math.log(self.initial_prob.get(pos_prev,1.0/100000000))
                        prevState_Prob=viterbi_prob[index-1][pos_prev][0]
                        prob_val=transition_prob+prevState_Prob
                        listOfProb.append(prob_val)
                        strOfPOS.append(viterbi_prob[index-1][pos_prev][1]+" "+pos)
                    emission_prob=math.log(self.emission_prob.get(pos,{}).get(word,1.0/100000000))
                    word_prob[pos]=(max(listOfProb)+emission_prob,strOfPOS[listOfProb.index(max(listOfProb))])
            viterbi_prob.append(word_prob)
        prob_pos_val=list(viterbi_prob[len(sentence)-1].values())
        listOfPos = max(prob_pos_val, key = lambda i : i[0])[1]
        listOfPos=listOfPos.split(" ")
        return listOfPos


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
