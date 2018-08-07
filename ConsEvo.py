# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 08:43:48 2015

@author: tg422
"""
import random
import csv
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# consonant indices and notations
CON = {1: "$p$", 2: "$p^h$", 3: "$b$", 4: "$m$", 5: "$w$", 6: "$t$", 7: "$t^h$", 8: "$d$", 9: "$ts$", 10: "$ts^h$", 
       11: "$dz$", 12: "$n$", 13: "$s$", 14: "$z$", 15: "$ur$", 16: "$l$", 17: "$l_t$", 18: "$tsh$", 19: "$tsh^h$", 20: "$dsh$", 
       21: "$sh$", 22: "$dg$", 23: "$t6$", 24: "$t6^h$", 25: "$dz6$", 26: "$ln$", 27: "$c6$", 28: "$z6$", 29: "$j$", 30: "$k$", 
       31: "$k^h$", 32: "$g$", 33: "$lg$", 34: "$x$", 35: "$f$", 36: "$r/B$", 37: "$q$", 38: "$q^h$", 39: "$G$"}
# consonant frequencies in Lizu (Duoxu_NCVG): consonant index: number of occurrence          
LIZU = {1: 97, 2: 50, 3: 120, 4: 172, 5: 45, 6: 100, 7: 48, 8: 106, 9: 35, 10: 91, 
        11: 49, 12: 62, 13: 66, 14: 39, 15: 61, 16: 110, 17: 22, 18: 27, 19: 60, 20: 78, 
        21: 76, 22: 19, 23: 46, 24: 42, 25: 63, 26: 72, 27: 14, 28: 33, 29: 61, 30: 66, 
        31: 74, 32: 80, 33: 32, 34: 25, 35: 15, 36: 19, 37: 40, 38: 14, 39: 6}
# consonant frequencies in SWM: consonant index: number of occurrence             
SWM = {1: 140, 2: 55,         4: 116, 5: 55, 6: 149, 7: 111,       9: 148, 10: 30,
               12: 43, 13: 62,                  16: 100,         18: 82, 19: 70,        
       21: 140, 22: 24, 23: 127, 24: 84,        26: 48, 27: 103,         29: 152, 30: 114,
       31: 51,          33: 9,          35: 66}
# consonant frequencies in Duoxu: consonant index: number of occurrence
DUOXU = {1: 62, 2: 58, 3: 95, 4: 226, 5: 107, 6: 43, 7: 31, 8: 52, 9: 25, 10: 50, 
         11: 27, 12: 40, 13: 43, 14: 22,      16: 116,      18: 9, 19: 27, 20: 29, 
         21: 52, 22: 33, 23: 98, 24: 101, 25: 88, 26: 97, 27: 76, 28: 25, 29: 127, 30: 105,
         31: 96, 32: 91, 33: 17, 34: 37, 35: 12, 36: 6}

COMP_INC = [2,4,5,16,22,23,24,25,26,27,29,30,31,32,34]
COMP_DEC = [1,3,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,28,33,35,36,37,38,39]

# possible consonant replacement: from: to
CON_REP = {17:16, 15:5, 36:5, 39:32, 37:30, 38:31}

# Global variables
# population setting
NUM_POP = 200      # number of agents in the population
POP_biSWMLIZU = 0.0 # proportion of bilingual SWM/Lizu speakers in the population
POP_monoLIZU = 0.5  # proportion of monolingual Lizu speakers in the population
POP_monoSWM = 0.5   # proportion of monolingual SWM speakers in the population
# POP_monoSWM + POP_monoLIZU + POP_biSWMLIZU = 1.0
# communication setting
NUM_COMM = 10000 # number of pairwise communications
REC_FREQ = 200   # frequency for recording results
# in each communication
NUM_CON = 300   # number of consonants produced in one communication
SCEN = '0'    # type of scenario: 0, no consonant replacement; 
            # '1a', consonant replacement only in bi-bi communication and has threshold restrict;
            # '1b', consonant replacement only in bi-bi communication and has no threshold restrict;
            # '2a', consonant replacement occurs in all communications and has threshold restrict;
            # '2b', consonant replacement occurs in all communications and has no threshold restrict; 
FREQ_ADJ = 0.002    # freqency adjustment for consonant replacement to occur; 
FREQ_THRES = 0.01   # threshold of frequency for consonant replacement to occur;
# number of running under different seeds
NUM_RUN = 20    # number of runs: note this only works when running in a local machine



######
# functions for simulation
######
class Dictlist(dict):
    """
    dictionary class, allowing multiple values for one key
    """
    def __setitem__(self, key, value):
        try: self[key]
        except KeyError: super(Dictlist, self).__setitem__(key, [])
        self[key].append(value)
        
def saveDict(fn,dict_rap):
    f = open(fn,'wb')
    w = csv.writer(f)
    for key, val in dict_rap.items():
        w.writerow([key, val])
    f.close()
     
def readDict(fn):
    f = open(fn,'rb')
    dict_rap = Dictlist()
    for key, val in csv.reader(f):
        dict_rap[key] = eval(val)
    f.close()
    return(dict_rap)

def calfreq(lem, cls, lemitem):
    freq = 0.0    
    lenlem = len(lemitem)
    for i in np.arange(0,lenlem,2):
        if lemitem[i] == cls: freq += np.float64(lemitem[i+1])
    return(freq)        


######
# Agent setting
######
class Agent(object):
    """
    Agent represents a speaker of Lizu or SWM:
        _ind -- index
        _group -- string, "Lizu_SWM", "SWM" (for simulation), or "Lizu", "Duoxu" (for testing);
        _con -- dictionary, store consonants and their occurring frequencies;
    """
    def __init__(self, init_type, ind, group):
        """
        init_type -- 1, initialize a speaker; 0, new speaker with no language in later generation; 
        group: string, "Lizu_SWM", "Lizu", "SWM", or "Duoxu"
        """
        self._ind, self._group = ind, group
        # initialize consonant dictionary
        self._con = {}
        for key in CON.keys():
            self._con[key] = 0
        if init_type == 1:
            if group == 'Lizu_SWM': # Lizu/SWM bilingual       
                self.initCon(LIZU); self.normCon()
                newdict = dict(SWM); sum_occur = sum(newdict.values())
                if sum_occur != 0.0:
                    for key in newdict.keys():
                        newdict[key] /= np.float64(sum_occur)              
                self.initCon(newdict)
            elif group == 'SWM': self.initCon(SWM) # SWM monolingual
            elif group == 'Lizu': self.initCon(LIZU) # Lizu monolingual
            elif group == 'Duoxu': self.initCon(DUOXU) # Duoxu monolingual
            else: raise ValueError("group input wrong, only 'Lizu_SWM', 'SWM', 'Lizu' or 'Duoxu' allowed!")
        self.normCon()
    
    def getInd(self):
        return self._ind
    
    def setInd(self, ind):
        if ind < 0 or ind >= NUM_POP: raise ValueError("index out of range [0, " + str(NUM_POP) + ")")
        self._ind = ind
    
    def getGroup(self):
        return self._group
    
    def setGroup(self, group):
        if group != 'Lizu_SWM' or group != 'SWM' or group != 'Lizu' or group != 'Duoxu':
            raise ValueError("group input wrong, only 'Lizu_SWM', 'SWM', 'Lizu' or 'Duoxu' allowed!")
        self._group = group
        
    def getCon(self):
        return self._con
        
    def setCon(self, conDict):
        self._con = conDict
    
    def __str__(self):
        return "Agent[" + str(self._ind) + "]: Group: " + self._group + "\nConsonants: " + str(self._con) + "\n"
    
    def initCon(self, conDict):
        """
        initialize _con based on conDict
        """
        for key in conDict.keys():
            self._con[key] += conDict[key]
        
    def normCon(self):
        """
        create a normalized consonant dictionary based on conson
        """
        sum_occur = sum(self._con.values())
        if sum_occur != 0.0:
            for key in self._con.keys():
                self._con[key] /= np.float64(sum_occur)        
    
    def speak(self, comType):
        """ 
        create a list of consonants based on self_.consonants
        arguments:
            comType -- 1, monoLizu <--> monoLizu (using Lizu consonants);
                       2, biLizu <--> biLizu (using all consonants, possible replacement);
                       3, biLizu <--> monoLizu (using Lizu consonants);
                       4, SWM -> biLizu (using SWM consonants);
                       5, random communications (using all consonants, possible replacement);
        """
        if comType < 0 or comType > 5: raise ValueError("speak: comType out of range [1, 5]")
        
        roulette, commcon, accu = [], [], 0.0
        if comType == 1 or comType == 3:
            # monoLizu <--> monoLizu or # biLizu <--> monoLizu (using Lizu consonants)
            for key in LIZU.keys():
                accu += self.getCon()[key]; roulette.append((key, accu))
        elif comType == 2 or comType == 5:
            # biLizu <--> biLizu (using all consonants, possible replacement)
            for key in self.getCon().keys():
                accu += self.getCon()[key]; roulette.append((key, accu))
        elif comType == 4:
            # SWM -> biLizu (using SWM consonants);
            for key in SWM.keys():
                accu += self.getCon()[key]; roulette.append((key, accu))
                
        # select consonants        
        for i in range(NUM_CON):
            r = np.random.uniform()
            for (cons, accu) in roulette:
                if r <= accu:                    
                    if comType != 4:                    
                        if SCEN == '0': commcon.append(cons) # no consonant replacement
                        elif SCEN == '1a':
                            # consonant replacement only occurs in bi-bi communications and has threshold restriction
                            if comType == 2:
                                if cons in CON_REP.keys() and self._con[cons] <= FREQ_THRES:
                                    # replace
                                    commcon.append(CON_REP[cons]) # consonant replacement
                                    self._con[CON_REP[cons]] += FREQ_ADJ # adjust replaced one's frequency
                                    if self._con[CON_REP[cons]] >= 1.0: self._con[CON_REP[cons]] = 1.0
                                    self._con[cons] -= FREQ_ADJ # reduce original one's frequency
                                    if self._con[cons] <= 0.0: self._con[cons] = 0.0
                                else: commcon.append(cons)
                            else: commcon.append(cons)
                        elif SCEN == '1b':
                            # consonant replacement only occurs in bi-bi communications and has no threshold restriction
                            if comType == 2:
                                if cons in CON_REP.keys():
                                    # replace
                                    commcon.append(CON_REP[cons]) # consonant replacement
                                    self._con[CON_REP[cons]] += FREQ_ADJ # adjust replaced one's frequency
                                    if self._con[CON_REP[cons]] >= 1.0: self._con[CON_REP[cons]] = 1.0
                                    self._con[cons] -= FREQ_ADJ # reduce original one's frequency
                                    if self._con[cons] <= 0.0: self._con[cons] = 0.0
                                else: commcon.append(cons)
                            else: commcon.append(cons)
                        elif SCEN == '2a':
                            # consonant replacement occurs in all communications and has threshold restriction
                            if cons in CON_REP.keys() and self._con[cons] <= FREQ_THRES:
                                commcon.append(CON_REP[cons]) # consonant replacement
                                self._con[CON_REP[cons]] += FREQ_ADJ # adjust replaced one's frequency
                                if self._con[CON_REP[cons]] >= 1.0: self._con[CON_REP[cons]] = 1.0
                                self._con[cons] -= FREQ_ADJ # reduce original one's frequency
                                if self._con[cons] <= 0.0: self._con[cons] = 0.0
                            else: commcon.append(cons)
                        elif SCEN == '2b':
                            # consonant replacement occurs in all communications and has no threshold restriction
                            if cons in CON_REP.keys():
                                commcon.append(CON_REP[cons]) # consonant replacement
                                self._con[CON_REP[cons]] += FREQ_ADJ # adjust replaced one's frequency
                                if self._con[CON_REP[cons]] >= 1.0: self._con[CON_REP[cons]] = 1.0
                                self._con[cons] -= FREQ_ADJ # reduce original one's frequency
                                if self._con[cons] <= 0.0: self._con[cons] = 0.0
                            else: commcon.append(cons)
                        else: raise ValueError("speak: SCEN out of range ['0', '1a', '1b', '2a', '2b']")
                    else: commcon.append(cons)                        
                    break
        return commcon
        
    def listen(self, comType, commcon):
        """
        update self._consonants based on heard consonList
        arguments:
            comType -- 1, monoLizu <--> monoLizu (using Lizu consonants);
                       2, biLizu <--> biLizu (using all consonants, possible replacement);
                       3, biLizu <--> monoLizu (using Lizu consonants);
                       4, SWM -> biLizu (using SWM consonants);
                       5, random communications (using all consonants, possible replacement);
            commcon -- list of heard consonants
        """
        if comType < 0 or comType > 5: raise ValueError("listen: comType out of range [1, 5]")

        if comType == 1 or comType == 3: newdict = dict(LIZU) # monoLizu <--> monoLizu or # biLizu <--> monoLizu (using Lizu consonants)
        elif comType == 2 or comType == 5: newdict = dict(self.getCon()) # biLizu <--> biLizu (using all consonants, possible replacement)
        elif comType == 4: newdict = dict(SWM) # SWM -> biLizu (using SWM consonants)
        
        for key in newdict.keys():
            newdict[key] = 0.0
        for con in commcon:
            newdict[con] += 1.0
        # normalize newdict    
        sum_occur = sum(newdict.values())
        if sum_occur != 0.0:
            for key in newdict.keys():
                newdict[key] /= np.float64(sum_occur)
        # add newdict to original dict        
        self.initCon(newdict); self.normCon()        

      
######
# functions for running
######
def runSim(direct, sim_case):
    """
    run simulation
    sim_case = 1: all speakers are Lizu, to see if the learning framework can help preserve their own language
             = 2: all speakers are SWM, to see if the learning framework can help preserve their own language
             = 3: all speakers are Duoxu, to see if the learning framework can help preserve their own language
             = 4: some Lizu and some SWM speakers, to see if contact can induce Duoxu language
    """    
    seed = random.randint(0, sys.maxint)
    np.random.seed(seed)
    #np.random.seed(0)  # for testing
    f = open(os.path.join(direct,'seed.txt'), 'w'); f.write('seed=' + str(seed) + '\n'); f.close()
    # initialize agents
    if sim_case != 4:
        # initialize agents
        pop = []
        for ind in range(NUM_POP):
            if sim_case == 1: pop.append(Agent(1, ind, "Lizu"))
            elif sim_case == 2: pop.append(Agent(1, ind, "SWM"))
            elif sim_case == 3: pop.append(Agent(1, ind, "Duoxu"))                    
        # record initial results
        recRes(pop, os.path.join(direct, 'output_0.csv'))
        # communications
        for comm in range(NUM_COMM):
            if (comm+1) % REC_FREQ == 0:
                recRes(pop, os.path.join(direct, 'output_' + str(comm+1) + '.csv'))
                print "Comm = ", comm+1        
            # select speaker and listener
            sp, li = -1, -1
            while sp == li:
                sp = random.choice(range(NUM_POP)); li = random.choice(range(NUM_POP))   
            comType = 5
            commcon = pop[sp].speak(comType); pop[li].listen(comType, commcon)
            
    else:
        # initialize agents
        pop = []
        for ind in range(NUM_POP):
            if ind < int(NUM_POP*POP_biSWMLIZU): pop.append(Agent(1, ind, "Lizu_SWM"))
            elif ind < int(NUM_POP*(POP_biSWMLIZU + POP_monoLIZU)): pop.append(Agent(1, ind, "Lizu"))
            else: pop.append(Agent(1, ind, "SWM"))
        # record initial results
        recRes(pop, os.path.join(direct, 'output_0.csv'))
        
        biLizu_set = range(0, int(NUM_POP*POP_biSWMLIZU))
        monoLizu_set = range(int(NUM_POP*POP_biSWMLIZU), int(NUM_POP*(POP_biSWMLIZU + POP_monoLIZU)))
        Lizu_set = biLizu_set + monoLizu_set        
        SWM_set = range(int(NUM_POP*(POP_biSWMLIZU + POP_monoLIZU)), NUM_POP)
        
        # communications
        for comm in range(NUM_COMM):
            if (comm+1) % REC_FREQ == 0:
                recRes(pop, os.path.join(direct, 'output_' + str(comm+1) + '.csv')) 
                print " Comm = ", comm+1
            # select listener first
            sp, li = -1, -1
            while sp == li or (sp in SWM_set and li in monoLizu_set):
                sp = random.choice(Lizu_set + SWM_set); li = random.choice(Lizu_set)
            if sp in monoLizu_set and li in monoLizu_set: comType = 1 # monoLizu <--> monoLizu            
            elif sp in biLizu_set and li in biLizu_set: comType = 2 # biLizu <--> biLizu  
            elif (sp in biLizu_set and li in monoLizu_set) or (sp in monoLizu_set and li in biLizu_set): comType = 3 # biLizu <--> monoLizu             
            elif sp in SWM_set and li in biLizu_set: comType = 4 # speaker is SWM: SWM -> biLizu
        
            commcon = pop[sp].speak(comType); pop[li].listen(comType, commcon)


def MainFunc(direct, sim_case):
    """
    main function
    sim_case = 1: all speakers are Lizu, to see if the learning framework can help preserve their own language
             = 2: all speakers are SWM, to see if the learning framework can help preserve their own language
             = 3: all speakers are Duoxu, to see if the learning framework can help preserve their own language
             = 4: some Lizu and some SWM speakers, to see if contact can induce Duoxu language
    """    
    for run in range(NUM_RUN):
        print "Run = ", run+1
        dir = os.path.join(direct, str(run+1))     
        try: os.stat(dir)
        except: os.mkdir(dir)
        runSim(dir, sim_case)
        


######
# functions for recording and drawing results
######
def recRes(pop, filename):
    """
    record results to csv file
    """
    resDict = Dictlist()
    for person in pop:
        resDict["group"] = person.getGroup(); resDict["ind"] = person.getInd()
        # record dictionary
        for cons in CON.keys():
            if cons in person.getCon().keys(): resDict[str(cons)] = np.float64(person.getCon()[cons])
            else: resDict[str(cons)] = np.float64(0.0)
    
    resDF = pd.DataFrame(np.zeros((len(resDict), NUM_POP+1+2)))
    resDF.loc[0,0] = "group"; resDF.loc[1,0] = "ind"
    for ind in range(NUM_POP):
        resDF.loc[0,ind+1] = pop[ind].getGroup(); resDF.loc[1,ind+1] = pop[ind].getInd()        
    resDF.loc[0,NUM_POP+1] = 0.0; resDF.loc[0,NUM_POP+2] = 0.0
    resDF.loc[1,NUM_POP+1] = 0.0; resDF.loc[1,NUM_POP+2] = 0.0    
    cur = 2    
    for cons in CON.keys():
        resDF.loc[cur,0] = str(cons)
        val = resDict[str(cons)]
        for ind in range(NUM_POP):
            resDF.loc[cur,ind+1] = val[ind]
        val_float = [np.float64(x) for x in val]
        resDF.loc[cur,NUM_POP+1] = np.mean(val_float)
        resDF.loc[cur,NUM_POP+2] = np.std(val_float)/np.float64(np.sqrt(NUM_POP-1))
        cur += 1
    col = ['conson']
    for ind in range(NUM_POP):
        col.append("Ag_" + str(ind))
    col.append('mean'); col.append('stderr')    
    resDF.columns = col    
    resDF.to_csv(filename, index=False)      
    

def plotLang(resDF, title, fileName):
    """
    draw the consonant distribution of a language
    """
    ind = np.arange(len(CON.keys())); width = 0.6; linewidth = 1; fontsize = 20  
    fig = plt.figure(); fig.set_size_inches(15, 10); ax = fig.add_subplot(111)
    ax.bar(ind, list(resDF['Freq']), width=width, linewidth=linewidth, color='grey')
    ax.grid(True)    
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('Consonants', fontsize=fontsize); ax.set_ylabel('Occurring frequencies', fontsize=fontsize)
    ax.set_xticks(ind+0.5)
    ax.set_ylim([0.0,0.12])    
    #xtickNames = ax.set_xticklabels(CON.values())
    xtickNames = ax.set_xticklabels(np.arange(1,len(CON.keys())+1,1))    
    plt.setp(xtickNames, rotation=45, fontsize=fontsize)
    #plt.show()
    plt.savefig(fileName, orientation = 'landscape', dpi=300)
    plt.close(fig)        


def plotImpLang(impType):
    """
    draw superimposed figure for Lizu and Duoxu
    arguments:
        impType: type of superimposition: 0, Lizu (Duoxu_NCVG) + Duoxu; 1, Lizu (Duoxu_NCVG) + SWM; 2, Duoxu + SWM; 3, Lizu (Duoxu_NCVG) + Duoxu + SWM
    """
    resDF_Lizu = pd.read_csv('./Duoxu_NCVG.csv'); resDF_SWM = pd.read_csv('./SWM.csv'); resDF_Duoxu = pd.read_csv('./Duoxu.csv')
    ind = np.arange(2*len(CON.keys())-1)
    Lizu_FreqList, SWM_FreqList, Duoxu_FreqList = [0]*len(ind), [0]*len(ind), [0]*len(ind)
    for i in range(len(CON.keys())):
        Lizu_FreqList[i*2] = resDF_Lizu.Freq[i]; SWM_FreqList[i*2] = resDF_SWM.Freq[i]; Duoxu_FreqList[i*2] = resDF_Duoxu.Freq[i]
    
    fig = plt.figure(); fig.set_size_inches(20, 10); ax = fig.add_subplot(111)
    width = 0.6; linewidth = 1; fontsize = 20
    if impType == 0:
        # Lizu + Duoxu
        rects1 = ax.bar(ind, Lizu_FreqList, width=width, linewidth=linewidth, color='black')
        rects2 = ax.bar(ind+width, Duoxu_FreqList, width=width, linewidth=linewidth, color='white')
        ax.legend((rects1[0], rects2[0]), ('Duoxu_NCVG', 'Duoxu'), fontsize=fontsize)
        filename = './Duoxu_NCVG_Duoxu.png'
    elif impType == 1:
        # Lizu + SWM
        rects1 = ax.bar(ind, Lizu_FreqList, width=width, linewidth=linewidth, color='black')
        rects2 = ax.bar(ind+width, SWM_FreqList, width=width, linewidth=linewidth, color='white')
        ax.legend((rects1[0], rects2[0]), ('Duoxu_NCVG', 'SWM'), fontsize=fontsize)
        filename = './Duoxu_NCVG_SWM.png'
    elif impType == 2:
        # Duoxu + SWM
        rects1 = ax.bar(ind, Duoxu_FreqList, width=width, linewidth=linewidth, color='black')
        rects2 = ax.bar(ind+width, SWM_FreqList, width=width, linewidth=linewidth, color='white')
        ax.legend((rects1[0], rects2[0]), ('Duoxu', 'SWM'), fontsize=fontsize)
        filename = './Duoxu_SWM.png'        
    elif impType == 3:
        # Lizu + Duoxu + SWM
        rects1 = ax.bar(ind-width, Lizu_FreqList, width=width, linewidth=linewidth, color='black')
        rects2 = ax.bar(ind, Duoxu_FreqList, width=width, linewidth=linewidth, color='white')
        rects3 = ax.bar(ind+width, SWM_FreqList, width=width, linewidth=linewidth, color='grey')
        ax.legend((rects1[0], rects2[0], rects3[0]), ('Duoxu_NCVG', 'Duoxu', 'SWM'), fontsize=fontsize)
        filename = './Duoxu_NCVG_Duoxu_SWM.png'
    else:
        raise ValueError("Wrong impType %d! Only 0, 1, 2, 3 allowed" % impType)
    #ax.xaxis_date()
    #ax.autoscale(tight=True)
    ax.grid(True)    
    ax.set_title('Consonant Occurring Frequencies', fontsize=fontsize) 
    ax.set_xlabel('Consonants', fontsize=fontsize)
    ax.set_xlim([-1, 2*len(CON.keys())])
    ax.set_ylabel('Occurring frequencies', fontsize=fontsize)
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(np.arange(1,2*len(CON.keys())+1,1))
    plt.setp(xtickNames, rotation=45, fontsize=fontsize)
    #plt.show()
    plt.savefig(filename, orientation = 'landscape', dpi=300)
    plt.close(fig) 


def calDfFreqLang(lang1, lang2):
    """
    calculate differences in frequencies between languages
    arguments:
        lang1, lang2: two languages, 'Lizu', 'Duoxu' or 'SWM'
    """
    if lang1 == 'Lizu' or lang1 == 'Duoxu' or lang1 == 'SWM': DF1 = pd.read_csv('./' + lang1 + '.csv')        
    else: raise ValueError("Wrong lang1 %s! only 'Lizu', 'Duoxu', or 'SWM' allowed" % lang1)
    if lang2 == 'Lizu' or lang2 == 'Duoxu' or lang2 == 'SWM': DF2 = pd.read_csv('./' + lang2 + '.csv')        
    else: raise ValueError("Wrong lang2 %s! only 'Lizu', 'Duoxu', or 'SWM' allowed" % lang2)

    diffDF = pd.DataFrame(np.zeros((len(CON.keys()), 9)))
    diffDF.loc[:,0] = CON.keys()
    diffDF.loc[:,1] = DF1['Occur']; diffDF.loc[:,2] = DF1['Freq']
    diffDF.loc[:,3] = DF2['Occur']; diffDF.loc[:,4] = DF2['Freq']
    for ind in range(len(CON.keys())):
        diffDF.loc[ind,5] = DF2['Occur'][ind] - DF1['Occur'][ind] 
        if diffDF.loc[ind,1] != 0.0: diffDF.loc[ind,6] = diffDF.loc[ind,5]/diffDF.loc[ind,1]
        else: diffDF.loc[ind,6] = 0.0
        diffDF.loc[ind,7] = DF2['Freq'][ind] - DF1['Freq'][ind] 
        if diffDF.loc[ind,2] != 0.0: diffDF.loc[ind,8] = diffDF.loc[ind,7]/diffDF.loc[ind,2]
        else: diffDF.loc[ind,8] = 0.0
    
    diffDF.columns = ['Consonants', 'Occur_'+lang1, 'Freq_'+lang1, 'Occur_'+lang2, 'Freq_'+lang2, 'Df_Occur', 'Df_Occur_p', 'Df_Freq', 'Df_Freq_p']
    diffDF.to_csv('./Diff_'+lang1+'_'+lang2+'.csv', index=False)


def DrawLang():
    """
    draw a standard Lizu, SWM, and Duoxu consonant distributions
    """
    agent_Lizu, agent_SWM, agent_Duoxu = Agent(1, 1, "Lizu"), Agent(1, 1, "SWM"), Agent(1, 1, "Duoxu")
    occur_Lizu, occur_SWM, occur_Duoxu = [], [], []
    for cons in CON.keys():        
        if cons in agent_Lizu.getCon().keys(): occur_Lizu.append(agent_Lizu.getCon()[cons])
        else: occur_Lizu.append(0.0)
        if cons in agent_SWM.getCon().keys(): occur_SWM.append(agent_SWM.getCon()[cons])
        else: occur_SWM.append(0.0)
        if cons in agent_Duoxu.getCon().keys(): occur_Duoxu.append(agent_Duoxu.getCon()[cons])
        else: occur_Duoxu.append(0.0)
    
    resDF_Lizu = pd.DataFrame(np.zeros((len(CON.keys()), 3))); resDF_Lizu.loc[:,0] = CON.keys() 
    resDF_SWM = pd.DataFrame(np.zeros((len(CON.keys()), 3))); resDF_SWM.loc[:,0] = CON.keys()
    resDF_Duoxu = pd.DataFrame(np.zeros((len(CON.keys()), 3))); resDF_Duoxu.loc[:,0] = CON.keys()

    cur = 0
    for cons in CON.keys():
        if cons in LIZU.keys(): resDF_Lizu.loc[cur,1], resDF_Lizu.loc[cur,2] = LIZU[cons], agent_Lizu.getCon()[cons]
        else: resDF_Lizu.loc[cur,1], resDF_Lizu.loc[cur,2] = 0.0, 0.0
        if cons in SWM.keys(): resDF_SWM.loc[cur,1], resDF_SWM.loc[cur,2] = SWM[cons], agent_SWM.getCon()[cons]
        else: resDF_SWM.loc[cur,1], resDF_SWM.loc[cur,2] = 0.0, 0.0
        if cons in DUOXU.keys(): resDF_Duoxu.loc[cur,1], resDF_Duoxu.loc[cur,2] = DUOXU[cons], agent_Duoxu.getCon()[cons]
        else: resDF_Duoxu.loc[cur,1], resDF_Duoxu.loc[cur,2] = 0.0, 0.0    
        cur += 1
        
    resDF_Lizu.columns = ['Consonants', 'Occur', 'Freq']; resDF_Lizu.to_csv('./Duoxu_NCVG.csv', index=False)
    resDF_SWM.columns = ['Consonants', 'Occur', 'Freq']; resDF_SWM.to_csv('./SWM.csv', index=False)
    resDF_Duoxu.columns = ['Consonants', 'Occur', 'Freq']; resDF_Duoxu.to_csv('./Duoxu.csv', index=False)

    # draw results
    plotLang(resDF_Lizu, 'Duoxu_NCVG', './Duoxu_NCVG.png')
    plotLang(resDF_SWM, 'SW Mandarin', './SWM.png')
    plotLang(resDF_Duoxu, 'Duoxu', './Duoxu.png')
        
    
def DrawRes(filename, figname, gen, sim_case):
    """
    draw results
    sim_case = 1: all speakers are Lizu, to see if the learning framework can help preserve their own language
             = 2: all speakers are SWM, to see if the learning framework can help preserve their own language
             = 3: all speakers are Duoxu, to see if the learning framework can help preserve their own language
             = 4: some Lizu and some SWM speakers, to see if contact can induce Duoxu language
    """    
    resDF = pd.read_csv(filename)
    mean_occur, stderr_occur = [], []
    for con_ind in range(resDF.shape[0]-2):        
        val = [np.float64(x) for x in list(resDF.loc[con_ind+2,:])]
        mean_occur.append(val[-2]); stderr_occur.append(val[-1])
        
    fig = plt.figure(); fig.set_size_inches(15, 10); ax = fig.add_subplot(111); fontsize = 20
    ax.bar(CON.keys(), mean_occur, width=0.6, linewidth=2, color='green', yerr=stderr_occur, error_kw=dict(capsize=5, elinewidth=2, ecolor='black'))
    ax.grid(True)    
    if sim_case == 1: ax.set_title('Lizu (Comm = ' + str(gen) + ')', fontsize=fontsize)
    elif sim_case == 2: ax.set_title('SWM (Comm = ' + str(gen) + ')', fontsize=fontsize)
    elif sim_case == 3: ax.set_title('Duoxu (Comm = ' + str(gen) + ')', fontsize=fontsize)
    elif sim_case == 4: ax.set_title('Mixed (Comm = ' + str(gen) + ')', fontsize=fontsize)
    ax.set_xlabel('Consonants', fontsize=fontsize); ax.set_ylabel('Occurring frequencies', fontsize=fontsize)
    ax.set_xticks([x + 0.5 for x in CON.keys()])
    xtickNames = ax.set_xticklabels(np.arange(1,len(CON)+1,1))
    plt.setp(xtickNames, rotation=45, fontsize=fontsize)
    #plt.show()
    plt.savefig(figname, orientation = 'landscape', dpi=300)
    plt.close(fig)        


def RecRes_all(direct, sim_case):
    """
    record all results
    sim_case = 1: all speakers are Lizu, to see if the learning framework can help preserve their own language
             = 2: all speakers are SWM, to see if the learning framework can help preserve their own language
             = 3: all speakers are Duoxu, to see if the learning framework can help preserve their own language
             = 4: some Lizu and some SWM speakers, to see if contact can induce Duoxu language
    """
    # draw each result's each generation's figure
    # print "1. Draw each run's each generation figures"
    # for run in range(NUM_RUN):
    #    for gen in range(NUM_GEN+1):
    #        filename = os.path.join(direct, str(run+1), "output_" + str(gen) + ".csv")
    #        figname = os.path.join(direct, str(run+1), "output_" + str(gen) + ".png")
    #        DrawRes(filename, figname, gen, sim_case)
            
    # calculate average results over all runs in the same condition
    for comm in range(NUM_COMM+1):
        if comm == 0 or comm % REC_FREQ == 0:
            print "At Comm = ", str(comm)
            size = NUM_POP
            resDF_all = pd.DataFrame(np.zeros((len(CON.keys()), 1+size*NUM_RUN)))
            resDF_all.loc[:,0] = CON.keys()
            resDF_rec = pd.DataFrame(np.zeros((len(CON.keys()), 3)))
            resDF_rec.loc[:,0] = CON.keys()

            for run in range(NUM_RUN):
                filename = os.path.join(direct, str(run+1), 'output_' + str(comm) + '.csv')
                resDF = pd.read_csv(filename)
                for cons_ind in range(len(CON.keys())):
                    for ind in range(size):
                        resDF_all.loc[cons_ind, int(size*run)+ind+1] = resDF.loc[cons_ind+2][ind+1] 
        
            mean_occur = []; stderr_occur = []
            cur = 0
            for cons_ind in range(len(CON.keys())):
                val = [np.float64(x) for x in list(resDF_all.loc[cons_ind,:])]
                mean_occur.append(np.mean(val[1:])); resDF_rec.loc[cur, 1] = np.mean(val[1:])
                stderr_occur.append(np.std(val[1:])/np.float64(np.sqrt(size*NUM_RUN-1))); resDF_rec.loc[cur, 2] = np.std(val[1:])/np.float64(np.sqrt(size*NUM_RUN-1))
                cur += 1
        
            resDF_rec.columns = ['Consonants', 'Mean', 'Stderr']
            resDF_rec.to_csv(os.path.join(direct, 'AvgRes_' + 'comm' + str(comm) + '.csv'), index=False)


def DrawRes_all(direct, sim_case):
    """
    record all results
    sim_case = 1: all speakers are Lizu, to see if the learning framework can help preserve their own language
             = 2: all speakers are SWM, to see if the learning framework can help preserve their own language
             = 3: all speakers are Duoxu, to see if the learning framework can help preserve their own language
             = 4: some Lizu and some SWM speakers, to see if contact can induce Duoxu language
    """
    for comm in range(NUM_COMM+1):
        if comm == 0 or comm % REC_FREQ == 0:
            print "At Comm = ", str(comm)
            # read data frames
            DF = pd.read_csv(os.path.join(direct, 'AvgRes_' + 'comm' + str(comm) + '.csv'), sep = ',')
            # draw average result's figure
            fig = plt.figure(); fig.set_size_inches(15, 10); ax = fig.add_subplot(111); fontsize = 20
            ax.bar(CON.keys(), DF['Mean'], width=0.6, linewidth=1.5, color='red', yerr=DF['Stderr'], error_kw=dict(capsize=5, elinewidth=2, ecolor='black'))
            ax.grid(True)    
            if sim_case == 1: ax.set_title('Lizu (Comm = ' + str(comm) + ')', fontsize=fontsize)
            elif sim_case == 2: ax.set_title('SWM (Comm = ' + str(comm) + ')', fontsize=fontsize)
            elif sim_case == 3: ax.set_title('Duoxu (Comm = ' + str(comm) + ')', fontsize=fontsize)
            elif sim_case == 4: ax.set_title('Mixed (Comm = ' + str(comm) + ')', fontsize=fontsize)
            ax.set_xlabel('Consonants', fontsize=fontsize); ax.set_ylabel('Occurring frequencies', fontsize=fontsize)
            ax.set_xticks([x + 0.5 for x in CON.keys()])
            xtickNames = ax.set_xticklabels(np.arange(1,len(CON)+1,1))
            plt.setp(xtickNames, rotation=45, fontsize=fontsize)
            #plt.show()
            plt.savefig(os.path.join(direct, 'AvgRes_' + 'comm' + str(comm) + '.png'), orientation = 'landscape', dpi=300)
            plt.close(fig)
    
    
def RecRes_sep(direct, popType):
    """
    record separate results; popType: "Lizu", "SWM", or "Both"
    """        
    for comm in range(NUM_COMM+1):
        if comm == 0 or comm % REC_FREQ == 0:
            print "At Comm = ", str(comm)
            
            if popType == 'Lizu' or popType == 'Both':
                size1 = int(NUM_POP*(POP_biSWMLIZU + POP_monoLIZU)) 
                resDF_mix1 = pd.DataFrame(np.zeros((len(CON.keys()), 1+size1*NUM_RUN)))
                resDF_mix1.loc[:,0] = CON.keys()
                resDF_mixrec1 = pd.DataFrame(np.zeros((len(CON.keys()), 3)))
                resDF_mixrec1.loc[:,0] = CON.keys()
            if popType == 'SWM' or popType == 'Both':
                size2 = int(NUM_POP*(1.0 - POP_biSWMLIZU - POP_monoLIZU))   
                resDF_mix2 = pd.DataFrame(np.zeros((len(CON.keys()), 1+size2*NUM_RUN)))
                resDF_mix2.loc[:,0] = CON.keys()
                resDF_mixrec2 = pd.DataFrame(np.zeros((len(CON.keys()), 3)))
                resDF_mixrec2.loc[:,0] = CON.keys()
                
            for run in range(NUM_RUN):
                filename = direct + "/" + str(run+1) + "/output_" + str(comm) + ".csv"
                resDF = pd.read_csv(filename)
                for cons_ind in range(len(CON.keys())):
                    if popType == 'Lizu' or popType == 'Both':
                        for ind in range(size1):
                            resDF_mix1.loc[cons_ind, size1*run+ind+1] = resDF.loc[cons_ind+2][ind+1]
                    if popType == 'SWM' or popType == 'Both':
                        for ind in range(size1, size1+size2):
                            resDF_mix2.loc[cons_ind, size2*run+ind-size1+1] = resDF.loc[cons_ind+2][ind+1]
                
            cur = 0
            if popType == 'Lizu' or popType == 'Both':
                mean_occur1 = []; stderr_occur1 = []
            if popType == 'SWM' or popType == 'Both':
                mean_occur2 = []; stderr_occur2 = []
            for cons_ind in range(len(CON.keys())):
                if popType == 'Lizu' or popType == 'Both':
                    val1 = [np.float64(x) for x in list(resDF_mix1.loc[cons_ind,:])]
                    mean_occur1.append(np.mean(val1[1:])); resDF_mixrec1.loc[cur, 1] = np.mean(val1[1:])
                    stderr_occur1.append(np.std(val1[1:])/np.float64(np.sqrt(NUM_RUN*size1))); resDF_mixrec1.loc[cur, 2] = np.std(val1[1:])/np.float64(np.sqrt(NUM_RUN*size1))
                if popType == 'SWM' or popType == 'Both':
                    val2 = [np.float64(x) for x in list(resDF_mix2.loc[cons_ind,:])]
                    mean_occur2.append(np.mean(val2[1:])); resDF_mixrec2.loc[cur, 1] = np.mean(val2[1:])
                    stderr_occur2.append(np.std(val2[1:])/np.float64(np.sqrt(NUM_RUN*size2))); resDF_mixrec2.loc[cur, 2] = np.std(val2[1:])/np.float64(np.sqrt(NUM_RUN*size2))
                cur += 1
        
            if popType == 'Lizu' or popType == 'Both':
                resDF_mixrec1.columns = ['Consonants', 'Mean', 'Stderr']
                resDF_mixrec1.to_csv(os.path.join(direct, 'AvgRes_' + 'comm' + str(comm) + '_Lizu.csv'), index=False)
            if popType == 'SWM' or popType == 'Both':
                resDF_mixrec2.columns = ['Consonants', 'Mean', 'Stderr']
                resDF_mixrec2.to_csv(os.path.join(direct, 'AvgRes_' + 'comm' + str(comm) + '_SWM.csv'), index=False)


def DrawRes_sep(direct, popType):
    """
    draw separate results; popType: "Lizu", "SWM", or "Both"
    """        
    for comm in range(NUM_COMM+1):
        if comm == 0 or comm % REC_FREQ == 0:
            print "At Comm = ", str(comm)                
            if popType == 'Lizu' or popType == 'Both':
                # read data frame                
                DF1 = pd.read_csv(os.path.join(direct, 'AvgRes_' + 'comm' + str(comm) + '_Lizu.csv'), sep=',')
                # draw average result's figure
                fig1 = plt.figure(); fig1.set_size_inches(15, 10); ax = fig1.add_subplot(111); fontsize=20
                ax.bar(np.arange(len(CON.keys())), DF1['Mean'], width=0.6, linewidth=1, color='grey', yerr=DF1['Stderr']*np.sqrt(NUM_POP/2.0), error_kw=dict(capsize=5, elinewidth=2, ecolor='black'))
                #ax.bar(np.arange(len(CON.keys())), list(DF1['Mean']), width=0.6, linewidth=1, color='grey')          
                ax.grid(True)    
                ax.set_title('Lizu (Comm = ' + str(comm) + ')', fontsize=fontsize)
                ax.set_xlabel('Consonants', fontsize=fontsize); ax.set_ylabel('Occurring frequencies', fontsize=fontsize)
                ax.set_ylim([0.0,0.12])            
                ax.set_xticks(np.arange(len(CON.keys()))+0.5)
                xtickNames = ax.set_xticklabels(np.arange(1,len(CON)+1,1))
                plt.setp(xtickNames, rotation=45, fontsize=fontsize)
                #plt.show()
                plt.savefig(os.path.join(direct, 'AvgRes_' + 'comm' + str(comm) + '_Lizu.png'), orientation = 'landscape', dpi=300)
                plt.close(fig1)
            if popType == 'SWM' or popType == 'Both':
                # read data frame                
                DF2 = pd.read_csv(os.path.join(direct, 'AvgRes_' + 'comm' + str(comm) + '_SWM.csv'), sep=',')     
                # draw average result's figure
                fig2 = plt.figure(); fig2.set_size_inches(15, 10); ax = fig2.add_subplot(111); fontsize = 20
                ax.bar(np.arange(len(CON.keys())), DF2['Mean'], width=0.6, linewidth=1, color='grey', yerr=DF2['Stderr']*np.sqrt(NUM_POP/2.0), error_kw=dict(capsize=5, elinewidth=2, ecolor='black'))
                #ax.bar(np.arange(len(CON.keys())), list(DF2['Mean']), width=0.6, linewidth=1, color='grey')          
                ax.grid(True)    
                ax.set_title('SWM (Comm = ' + str(comm) + ')', fontsize=fontsize)
                ax.set_xlabel('Consonants', fontsize=fontsize); ax.set_ylabel('Occurring frequencies', fontsize=fontsize)
                ax.set_ylim([0.0,0.12])            
                ax.set_xticks(np.arange(len(CON.keys()))+0.5)
                xtickNames = ax.set_xticklabels(np.arange(1,len(CON)+1,1))
                plt.setp(xtickNames, rotation=45, fontsize=fontsize)
                #plt.show()
                plt.savefig(os.path.join(direct, 'AvgRes_' + 'comm' + str(comm) + '_SWM.png'), orientation = 'landscape', dpi=300)
                plt.close(fig2)


def plotImpRes(direct, impType, simType, ComRange):
    """
    draw imposed results between simulation results with Lizu/Duoxu/SWM
    arguments:
        direct: directory of simulation results; language's result is in the same directory of the py code
        impType: which language result to be imposed: 'Lizu', 'Duoxu', or 'SWM'
        simType: which language simulation result to be imposed: 'Lizu' or 'SWM'
        ComRange: range of communications of the simulation results at which number of communications
    """
    ind = np.arange(2*len(CON.keys())-1) 
    
    if impType == 'Lizu' or impType == 'Duoxu' or impType == 'SWM' or impType == 'Duoxu_PCVG': DF2 = pd.read_csv('./' + impType + '.csv')        
    else: raise ValueError("Wrong impType %s! only 'Lizu', 'Duoxu', or 'SWM' allowed" % impType)
    
    for numCom in ComRange:
        if simType == 'Lizu' or simType == 'SWM': DF1 = pd.read_csv(os.path.join(direct, 'AvgRes_comm' + str(numCom) + '_' + simType + '.csv'))
        else: raise ValueError("Wrong simType %s! only 'Lizu' or 'SWM' allowed" % simType)
        
        DF1_FreqList, DF2_FreqList = [0]*len(ind), [0]*len(ind)
        DF1_ErrList = [0]*len(ind)
        for i in range(len(CON.keys())):
            DF1_FreqList[i*2] = DF1.Mean[i]; DF2_FreqList[i*2] = DF2.Freq[i]
            DF1_ErrList[i*2] = DF1['Stderr'][i]
    
        fig, ax = plt.subplots(); fig.set_size_inches(20, 10)
        width = 0.6; linewidth = 1; fontsize = 20
        rects1 = ax.bar(ind, DF1_FreqList, width=width, linewidth=linewidth, color='red', yerr=DF1_ErrList, error_kw=dict(capsize=2, elinewidth=1, ecolor='black'))
        rects2 = ax.bar(ind+width, DF2_FreqList, width=width, linewidth=linewidth, color='blue')
        ax.legend((rects1[0], rects2[0]), ('sim_' + simType, impType), fontsize=fontsize)
        ax.grid(True)    
        ax.set_title('Comparing Consonant Occurring Frequencies (Comm= ' + str(numCom) + ')', fontsize=fontsize) 
        ax.set_xlabel('Consonants', fontsize=fontsize); ax.set_ylabel('Occurring frequencies', fontsize=fontsize)
        ax.set_xticks(ind+width)
        xtickNames = ax.set_xticklabels(np.arange(1,2*len(CON.keys())+1,1))     
        plt.setp(xtickNames, rotation=45, fontsize=fontsize)
        #plt.show()
        plt.savefig(os.path.join(direct, 'sim' + simType + '_' + impType + '_comm' + str(numCom) + '.png'), orientation = 'landscape', dpi=300)
        plt.close(fig)   
    

def calSSDRes(direct, simType, ComRange):
    """
    calculate differences in frequencies between simulation results
    arguments:
        direct: directory of simulation results
        simType: which language simulation result to be imposed: 'Lizu' or 'SWM'
        ComRange: range of communications of the simulation results at which number of communications
    """
    if simType == 'Lizu' or simType == 'SWM': DF0 = pd.read_csv(os.path.join(direct, 'AvgRes_comm' + str(ComRange[0]) + '_' + simType + '.csv')) 
    else: raise ValueError("Wrong simType %s! only 'Lizu' or 'SWM' allowed" % simType)
    
    for numCom in ComRange[1:]:
        DF1 = pd.read_csv(os.path.join(direct, 'AvgRes_comm' + str(numCom) + '_' + simType + '.csv')) 
           
        diffDF = pd.DataFrame(np.zeros((len(CON.keys()), 4)))
        diffDF.loc[:,0] = CON.keys()
        diffDF.loc[:,1] = DF0['Mean']; diffDF.loc[:,2] = DF1['Mean']
        diffDF.loc[:,3] = DF1['Mean'] - DF0['Mean']        
        
        diffDF.columns = ['Consonants', 'F0', 'F_' + str(numCom), 'DF']
        diffDF.to_csv(direct + './SSD_comm' + str(numCom) + '.csv', index=False)


def calSSD_avg(direct, SSDtype, simType, empType, cond_Pop, ScenType, numCom):
    """
    calculate global measure of accumulated squared frequency difference of all consonants
    between a simulation result and an empirical data at a particular time
    arguments:
        direct -- directory of simulation data; empirical data is in the same folder of the code
        SSDtype -- type of SSD; 0, all included; 1, removed two biggest        
        simType -- which type of simulation data for calculation
        empType -- which type of empirical data for calculation
        cond_Pop -- simulation condition: e.g., 'P_2_48_50': ratio of mono-/bi-lingual speakers
        ScenType -- type of scenario: 'Scen0', 'Scen1a', 'Scen1b'        
        numCom -- number of communication at which calculation is done
    outputs:
        totDif -- absolute summed squared differences
        incDif -- percentage summed squared differences bigger than 0 (increasing consonant instances)
        decDif -- percentage summed squared differences smaller than 0
        totDif_p -- percentage summed squared differences
        incDif_p -- percentage summed squared differences bigger than 0
        decDif_p -- percentage summed squared differences smaller than 0
    """
    empData = pd.read_csv(empType + '.csv')
    simData = pd.read_csv(os.path.join(direct, ScenType, cond_Pop, 'AvgRes_comm' + str(numCom) + '_' + simType + '.csv'))
    
    SSDDF = pd.DataFrame(np.zeros((len(CON.keys()), 3)))
    SSDDF.columns = ['Consonants', 'SSD', 'std_SSD']
    SSDDF.Consonants = CON.keys()
    SSDDF.SSD = [simData.loc[i,'Mean'] - empData.loc[i, 'Freq'] for i in range(len(empData))]
    SSDDF.std_SSD = simData.Stderr
    
    if SSDtype == 0:
        totSSD = sum(SSDDF.SSD**2)
    elif SSDtype == 1:
        tempSSD = list(SSDDF.SSD**2)
        remNo = int(len(CON.keys()) - len(CON.keys())*0.95)
        for i in range(remNo):
            tempSSD.remove(max(tempSSD))
        totSSD = sum(tempSSD)    
    std_totSSD = np.mean(SSDDF.std_SSD)
    incSSD = sum(SSDDF['SSD'][[i-1 for i in COMP_INC]]**2)
    std_incSSD = np.mean(SSDDF['std_SSD'][[i-1 for i in COMP_INC]])
    decSSD = sum(SSDDF['SSD'][[i-1 for i in COMP_DEC]]**2)
    std_decSSD = np.mean(SSDDF['std_SSD'][[i-1 for i in COMP_DEC]])
    
    return totSSD, std_totSSD, incSSD, std_incSSD, decSSD, std_decSSD


def calSSD_ind(direct, SSDtype, empType, cond_Pop, ScenType, run, numCom):
    """
    calculate global measure of accumulated squared frequency difference of all consonants
    between a simulation result and an empirical data at a particular time
    arguments:
        direct -- directory of simulation data; empirical data is in the same folder of the code
        SSDtype -- type of SSD; 0, all included; 1, removed two biggest        
        empType -- which type of empirical data for calculation
        cond_Pop -- simulation condition: e.g., 'P_2_48_50': ratio of mono-/bi-lingual speakers
        ScenType -- type of scenario: 'Scen0', 'Scen1a', 'Scen2a'
        run --- runID         
        numCom -- number of communication at which calculation is done
    outputs:
        totSSD-- absolute summed squared differences
        incSSD -- percentage summed squared differences bigger than 0 (increasing consonant instances)
        decSSD -- percentage summed squared differences smaller than 0        
    """
    empData = pd.read_csv(empType + '.csv')
    simData = pd.read_csv(os.path.join(direct, ScenType, cond_Pop, str(run), 'output_' + str(numCom) + '.csv'))
    
    SSDDF = pd.DataFrame(np.zeros((len(CON.keys()), 2)))
    SSDDF.columns = ['Consonants', 'SSD']
    SSDDF.Consonants = CON.keys()
    for con in range(len(CON.keys())):
        listFreq = list(simData.loc[2+con][1:(1+int(NUM_POP/2.0))])
        listFreq = [float(i) for i in listFreq]
        conFreq = np.mean(listFreq)
        SSDDF.loc[con,'SSD'] = conFreq - empData.loc[con, 'Freq']
    
    if SSDtype == 0:
        totSSD = sum(SSDDF.SSD**2)
    elif SSDtype == 1:
        tempSSD = list(SSDDF.SSD**2)
        remNo = int(len(CON.keys()) - len(CON.keys())*0.95)
        for i in range(remNo):
            tempSSD.remove(max(tempSSD))
        totSSD = sum(tempSSD)    
    incSSD = sum(SSDDF['SSD'][[i-1 for i in COMP_INC]]**2)
    decSSD = sum(SSDDF['SSD'][[i-1 for i in COMP_DEC]]**2)
    
    return totSSD, incSSD, decSSD


def colSSD_IndRun(direct, SSDtype, empType, cond_PopList, ScenTypeList, ComRange):
    """
    create a data frame to collect DfFreq values at different numCom in different conditions (individual run's results) 
    arguments:
        direct -- directory of simulation data; empirical data is in the same folder of the code
        SSDtype -- type of SSD; 0, all included; 1, removed two biggest        
        empType -- which type of empirical data for calculation
        cond_PopList -- list of simulation condition: e.g., 'P_2_48_50': ratio of mono-/bi-lingual speakers
        ScenTypeLits -- type of scenario: ['Scen0', 'Scen1a', 'Scen2a']
        ComRange -- range of communication at which calculation is done
    """
    for ScenType in ScenTypeList:
        for cond_Pop in cond_PopList:
            print "ScenType: ", ScenType, "; cond_Pop: ", cond_Pop
                
            resDF = pd.DataFrame(np.zeros((len(ComRange)*NUM_RUN, 7)))
            resDF.columns = ['runID', 'numCom', 'Pop', 'Scen', 'totSSD', 'incSSD', 'decSSD']
            resDF.Pop = cond_Pop; resDF.Scen = ScenType

            curCase = 0
            for run in range(1,NUM_RUN+1):
                # calculate DfFreq at different run
                for comm in ComRange:                                        
                    resDF.loc[curCase, 'runID'] = run; resDF.loc[curCase, 'numCom'] = comm                
                    resDF.loc[curCase, 'totSSD'], resDF.loc[curCase, 'incSSD'], resDF.loc[curCase, 'decSSD'] = calSSD_ind(direct, SSDtype, empType, cond_Pop, ScenType, run, comm)
                    curCase += 1   
            
            # save results            
            if SSDtype == 0: resDF.to_csv(os.path.join(direct, ScenType, 'IndRun_SSD' + '_' + cond_Pop + '_' + ScenType + '.csv'), index=False)
            elif SSDtype == 1: resDF.to_csv(os.path.join(direct, ScenType, 'IndRun_SSD' + '_' + cond_Pop + '_' + ScenType + '_1.csv'), index=False)

    
def colSSD_AvgRun(direct, SSDtype, simType, empType, cond_PopList, ScenTypeList, ComRange):
    """
    create a data frame to collect DfFreq values at different numCom in different conditions 
    arguments:
        direct -- directory of simulation data; empirical data is in the same folder of the code
        SSDtype -- type of SSD; 0, all included; 1, removed two biggest        
        simType -- which type of simulation data for calculation
        empType -- which type of empirical data for calculation
        cond_PopList -- list of simulation condition: e.g., 'P_2_48_50': ratio of mono-/bi-lingual speakers
        ScenTypeLits -- type of scenario: ['Scen0', 'Scen1', 'Scen2']
        ComRange -- range of communication at which calculation is done
    """
    for ScenType in ScenTypeList:
        for cond_Pop in cond_PopList:
            resDF = pd.DataFrame(np.zeros((len(ComRange), 9)))
            resDF.columns = ['numCom', 'Pop', 'Scen', 'totSSD', 'std_totSSD', 'incSSD', 'std_incSSD', 'decSSD', 'std_decSSD']
            resDF.numCom = ComRange; resDF.Pop = cond_Pop; resDF.Scen = ScenType
            # calculate DfFreq at different numCom
            curCase = 0            
            for com in ComRange:
                resDF.loc[curCase, 'totSSD'], resDF.loc[curCase, 'std_totSSD'], resDF.loc[curCase, 'incSSD'], resDF.loc[curCase, 'std_incSSD'], resDF.loc[curCase, 'decSSD'], resDF.loc[curCase, 'std_decSSD'] = calSSD_avg(direct, SSDtype, simType, empType, cond_Pop, ScenType, com)
                curCase += 1    
            # save results            
            if SSDtype == 0: resDF.to_csv(os.path.join(direct, ScenType, 'SSD' + '_' + cond_Pop + '_' + ScenType + '.csv'), index=False)
            elif SSDtype == 1: resDF.to_csv(os.path.join(direct, ScenType, 'SSD' + '_' + cond_Pop + '_' + ScenType + '_1.csv'), index=False)


def colSSD_AvgCond(direct, SSDtype, simType, empType, cond_PopList, ScenTypeList, numCom):
    """
    create a data frame to collect DfFreq values at same numCom in different conditions
    arguments:
        direct -- directory of simulation data; empirical data is in the same folder of the code
        SSDtype -- type of SSD; 0, all included; 1, removed two biggest        
        simType -- which type of simulation data for calculation
        empType -- which type of empirical data for calculation
        cond_PopList -- list of simulation condition: e.g., 'P_2_48_50': ratio of mono-/bi-lingual speakers
        ScenTypeList -- type of scenario: ['Scen0', 'Scen1', 'Scen2']
        numCom -- number of communication at which calculation is done    
    """
    for ScenType in ScenTypeList:
        totCase = len(cond_PopList)
        resDF = pd.DataFrame(np.zeros((totCase, 9)))
        resDF.columns = ['numCom', 'Pop', 'Scen', 'totSSD', 'std_totSSD', 'incSSD', 'std_incSSD', 'decSSD', 'std_decSSD']
        # calculate DfFreq in different conditions
        curCase = 0
        for cond_Pop in cond_PopList:
            resDF.loc[curCase, 'numCom'] = numCom; resDF.loc[curCase, 'Pop'] = cond_Pop; resDF.loc[curCase, 'Scen'] = ScenType
            resDF.loc[curCase, 'totSSD'], resDF.loc[curCase, 'std_totSSD'], resDF.loc[curCase, 'incSSD'], resDF.loc[curCase, 'std_incSSD'], resDF.loc[curCase, 'decSSD'], resDF.loc[curCase, 'std_decSSD'] = calSSD_avg(direct, SSDtype, simType, empType, cond_Pop, ScenType, numCom)
            curCase += 1    
        # save results            
        if SSDtype == 0: resDF.to_csv(os.path.join(direct, ScenType, 'SSD_' + str(numCom) + '.csv'), index=False)
        elif SSDtype == 1: resDF.to_csv(os.path.join(direct, ScenType, 'SSD_' + str(numCom) + '_1.csv'), index=False)


def colSSD_Ind_Avg(direct, SSDtype, cond_PopList, ScenTypeList, ComRange):
    """
    transform from IndSSD_*.csv to SSD_*.csv
    arguments:
        direct -- directory of simulation data; empirical data is in the same folder of the code
        SSDtype -- type of SSD; 0, all included; 1, removed two biggest        
        simType -- which type of simulation data for calculation
        empType -- which type of empirical data for calculation
        cond_PopList -- list of simulation condition: e.g., 'P_2_48_50': ratio of mono-/bi-lingual speakers
        ScenTypeList -- type of scenario: ['Scen0', 'Scen1', 'Scen2']
        numCom -- number of communication at which calculation is done    
    """
    for ScenType in ScenTypeList:
        for cond_Pop in cond_PopList:
            if SSDtype == 0: IndRunDF = pd.read_csv(os.path.join(direct, ScenType, 'IndRun_SSD' + '_' + cond_Pop + '_' + ScenType + '.csv'))
            elif SSDtype == 1: IndRunDF = pd.read_csv(os.path.join(direct, ScenType, 'IndRun_SSD' + '_' + cond_Pop + '_' + ScenType + '_1.csv'))
            numRun = len(pd.unique(IndRunDF.runID))        
            resDF_Avg = pd.DataFrame(np.zeros((numRun, 3)))
            resDF_Avg.columns = ['runID', 'numCom', 'totSSD_min']
            cur = 0
            for run in range(1,numRun+1):
                subDF = IndRunDF.loc[IndRunDF.runID==run,:].reindex()
                resDF_Avg.loc[cur,'runID'] = run
                resDF_Avg.loc[cur,'numCom'] = float(subDF.loc[subDF.totSSD==np.min(subDF.totSSD),'numCom'])
                resDF_Avg.loc[cur,'totSSD_min'] = np.min(subDF.totSSD)
                cur += 1
            
            # save results            
            if SSDtype == 0: resDF_Avg.to_csv(os.path.join(direct, ScenType, 'IndRunAvg_SSD' + '_' + cond_Pop + '_' + ScenType + '.csv'), index=False)
            elif SSDtype == 1: resDF_Avg.to_csv(os.path.join(direct, ScenType, 'IndRunAvg_SSD' + '_' + cond_Pop + '_' + ScenType + '_1.csv'), index=False)
    
            # calculate DfFreq at different numCom
            resDF = pd.DataFrame(np.zeros((len(ComRange), 9)))
            resDF.columns = ['numCom', 'Pop', 'Scen', 'totSSD', 'std_totSSD', 'incSSD', 'std_incSSD', 'decSSD', 'std_decSSD']
            resDF.numCom = ComRange; resDF.Pop = cond_Pop; resDF.Scen = ScenType
            curCase = 0            
            for com in ComRange:
                totSSD_value, incSSD_value, decSSD_value = [], [], []
                for run in range(1,numRun+1):
                    subDF = IndRunDF.loc[IndRunDF.runID==run,:].reindex()
                    totSSD_value.append(float(subDF.loc[subDF.numCom==com,'totSSD']))
                    incSSD_value.append(float(subDF.loc[subDF.numCom==com,'incSSD']))
                    decSSD_value.append(float(subDF.loc[subDF.numCom==com,'decSSD']))
                resDF.loc[curCase, 'totSSD'] = np.mean(totSSD_value)
                resDF.loc[curCase, 'std_totSSD'] = np.std(totSSD_value)
                resDF.loc[curCase, 'incSSD'] = np.mean(incSSD_value)
                resDF.loc[curCase, 'std_incSSD'] = np.std(incSSD_value)
                resDF.loc[curCase, 'decSSD'] = np.mean(decSSD_value)
                resDF.loc[curCase, 'std_decSSD'] = np.std(decSSD_value)
                curCase += 1
            # save results            
            if SSDtype == 0: resDF.to_csv(os.path.join(direct, ScenType, 'SSD' + '_' + cond_Pop + '_' + ScenType + '.csv'), index=False)
            elif SSDtype == 1: resDF.to_csv(os.path.join(direct, ScenType, 'SSD' + '_' + cond_Pop + '_' + ScenType + '_1.csv'), index=False)
    
    
def drawSSD_cond(direct, SSDtype, DrawType, cond_PopList, ScenTypeList, ComRange):
    """
    draw results cross conditions
    arguments:
        direct -- directory of result csv file
        SSDtype -- type of SSD; 0, all included; 1, removed two biggest        
        DrawType -- type of drawing: 1, draw fixed rep and Mix figure; 2, draw fixed rep and Pop figure; 3, draw bar plot comparing rep and norep (mix); 4, draw bar plot comparing rep and norep (pop)  
        cond_PopList -- list of simulation condition: e.g., 'P_10_90': ratio of mono-/bi-lingual speakers
        ScenTypeList -- type of scenario: ['Scen0', 'Scen1', 'Scen2']
        ComRange -- range of communication at which calculation is done
    """
    if DrawType == 1:
        for ScenType in ScenTypeList:
            # draw fixed rep figure
            totRow, totCol = len(ComRange), len(cond_PopList)
            # get data
            resDF = pd.DataFrame(np.zeros((totRow, 6*totCol)))
            resDF.loc[:,0] = ComRange
            colName = ['numCom']
            curCol = 1
            for Pop in cond_PopList:
                if SSDtype == 0: simData = pd.read_csv(os.path.join(direct, ScenType, 'SSD_' + Pop + '_' + ScenType + '.csv'))
                elif SSDtype == 1: simData = pd.read_csv(os.path.join(direct, ScenType, 'SSD_' + Pop + '_' + ScenType + '_1.csv'))
                        
                resDF.loc[:,curCol] = simData.totSSD; resDF.loc[:,curCol+1] = simData.std_totSSD
                resDF.loc[:,curCol+2] = simData.incSSD; resDF.loc[:,curCol+3] = simData.std_incSSD
                resDF.loc[:,curCol+4] = simData.decSSD; resDF.loc[:,curCol+5] = simData.std_decSSD
                curCol += 6
                colName += [Pop + '_totSSD', Pop + '_totSSD_std', Pop + '_incSSD', Pop + '_incSSD_std', Pop + '_decSSD', Pop + '_decSSD_std']
            resDF.columns = colName
            if SSDtype == 0: resDF.to_csv(os.path.join(direct, ScenType, 'resSSD_' + ScenType + '.csv'), index = False)
            elif SSDtype == 1: resDF.to_csv(os.path.join(direct, ScenType, 'resSSD_' + ScenType + '_1.csv'), index = False)
                
            # draw errorbar figure
            fig = plt.figure(); fig.set_size_inches(20, 10); ax = fig.add_subplot(111); fontsize = 20
            linestyleList = ['-','--','-.', ':']
#            dashstyleList = [[12,12], # line
#                             [3,3], # dot
#                             [6,6], # dash
#                             [12,6,12,6], # dash dot
#                             [3,6,3,6], # dash dot dot
#                             [12,6,12,6,3,6]] # dash dash dot
            curStyle = 0
            for Pop in cond_PopList:
                if curStyle < len(linestyleList): ax.errorbar(resDF.numCom, resDF.loc[:, Pop + '_totSSD'], color = 'black', linestyle = linestyleList[curStyle], yerr = resDF.loc[:, Pop + '_totSSD_std'], elinewidth = 2, capsize = 5)    
                else: ax.errorbar(resDF.numCom, resDF.loc[:, Pop + '_totSSD'], color = 'red', linestyle = linestyleList[curStyle%len(linestyleList)], yerr = resDF.loc[:, Pop + '_totSSD_std'], elinewidth = 2, capsize = 5)    
                #if curStyle < len(dashstyleList): ax.errorbar(resDF.numCom, resDF.loc[:, Pop + '_totSSD'], color = 'black', dashes = dashstyleList[curStyle], yerr = resDF.loc[:, Pop + '_totSSD_std'], elinewidth = 2, capsize = 5)    
                #else: ax.errorbar(resDF.numCom, resDF.loc[:, Pop + '_totSSD'], color = 'red', dashes = dashstyleList[curStyle%len(dashstyleList)], yerr = resDF.loc[:, Pop + '_totSSD_std'], elinewidth = 2, capsize = 5)    
                curStyle += 1
            ax.set_xlabel('Communications per Agent', fontsize=fontsize)
            ax.set_ylabel('SSD', fontsize=fontsize)
            ax.legend(cond_PopList, loc='lower right', fontsize=fontsize)
            plt.title('SSD in ' + ScenType, fontsize=fontsize)
            plt.grid()
            plt.ylim((0.00, 0.03))
            plt.xlim((0,ComRange[-1]))
            xticks = range(0,ComRange[-1]+1,1000)
            plt.xticks(xticks)
            xtickNames = ax.set_xticklabels([x/100 for x in xticks])     
            plt.setp(xtickNames, fontsize=fontsize)
            plt.show()
            if SSDtype == 0: plt.savefig(os.path.join(direct, ScenType, 'resSSD_' + ScenType + '.png'), orientation = 'landscape', dpi = 300)
            elif SSDtype == 1: plt.savefig(os.path.join(direct, ScenType, 'resSSD_' + ScenType + '_1.png'), orientation = 'landscape', dpi = 300)
            plt.close(fig)

    elif DrawType == 2:
        # draw bar plot comparing rep and norep (pop)
        totRow, totCol = len(cond_PopList), 3*len(ScenTypeList)
        resDF = pd.DataFrame(np.zeros((totRow, 1+totCol)))
        col = ['Pop']
        for ScenType in ScenTypeList:
            col.append('minSSD_' + ScenType); col.append('minSSD_std_' + ScenType); col.append('numCom_' + ScenType)
        resDF.columns = col    
        curLine = 0  
        for Pop in cond_PopList:
            # get data
            for ScenType in ScenTypeList:
                if SSDtype == 0: simData_Scen = pd.read_csv(os.path.join(direct, ScenType, 'SSD_' + Pop + '_' + ScenType + '.csv'))
                elif SSDtype == 1: simData_Scen = pd.read_csv(os.path.join(direct, ScenType, 'SSD_' + Pop + '_' + ScenType + '_1.csv'))
                resDF.loc[curLine, 'Pop'] = Pop    
                resDF.loc[curLine, 'minSSD_' + ScenType] = np.min(simData_Scen.totSSD)
                resDF.loc[curLine, 'minSSD_std_' + ScenType] = np.float(simData_Scen.loc[simData_Scen['totSSD'] == np.min(simData_Scen.totSSD), 'std_totSSD'])
                resDF.loc[curLine, 'numCom_' + ScenType] = np.int(simData_Scen.loc[simData_Scen['totSSD'] == np.min(simData_Scen.totSSD), 'numCom'])
            curLine += 1
        if SSDtype == 0: resDF.to_csv(os.path.join(direct, 'resSSD_AllScen.csv'), index = False)
        elif SSDtype == 1: resDF.to_csv(os.path.join(direct, 'resSSD_AllScen_1.csv'), index = False)
        
        # draw errorbar figure
        n_group = len(cond_PopList)
        fig, ax = plt.subplots(); fig.set_size_inches(20, 10)
        index = np.arange(n_group)
        bar_width = 0.15; opacity = 0.5; fontsize = 20
        error_config = {'ecolor': '0.3'}
        #colors = ['0.0', '0.16', '0.32', '0.48', '0.64', '0.8']
        colors = ['black','red','yellow','green','blue','purple']
        for sc in range(len(ScenTypeList)):
            means_Scen = resDF['minSSD_' + ScenTypeList[sc]]; std_Scen = resDF['minSSD_std_' + ScenTypeList[sc]]
            rect = plt.bar(index + (sc-1)*bar_width, means_Scen, bar_width, alpha = opacity, color = colors[sc], yerr = std_Scen, error_kw = error_config, capsize=10, label = 'Scen0')
        for i in range(n_group):
            for sc in range(len(ScenTypeList)):
                plt.text(index[i] + (sc-1)*bar_width, resDF.loc[i, 'minSSD_' + ScenTypeList[sc]], str(int(resDF.loc[i, 'numCom_' + ScenTypeList[sc]])/100), fontsize=fontsize, rotation=90)
        plt.xticks(index + bar_width)
        plt.ylabel('SSD', fontsize=fontsize)
        plt.title('Minimum SSD in various scenarios', fontsize=fontsize)
        plt.legend(ScenTypeList, loc='lower left', fontsize=fontsize)
        plt.tight_layout()
        plt.xlim((-1.5*bar_width, n_group-1 + 3.5*bar_width))
        plt.ylim((0.00,0.012))
        xtickNames = ax.set_xticklabels(cond_PopList)     
        plt.setp(xtickNames, fontsize=fontsize)
        plt.grid()
        plt.show()
        if SSDtype == 0: plt.savefig(os.path.join(direct, 'resSSD_AllScen.png'), orientation = 'landscape', dpi = 300)
        elif SSDtype == 1: plt.savefig(os.path.join(direct, 'resSSD_AllScen_1.png'), orientation = 'landscape', dpi = 300)
        plt.close(fig)



# to run simulations, use one of the following lines
#MainFunc('./Lizu', 1)
#MainFunc('./SWM', 2)
#MainFunc('./Duoxu', 3)
#MainFunc('.', 4)

# to record results, use one of the following lines
#RecRes_all('.', 4)
#RecRes_sep('.', 'Lizu')

# to draw average result of the population, use one of the following lines
#DrawRes_all('./Lizu', 1)
#DrawRes_all('./SWM', 2)
#DrawRes_all('./Duoxu', 3)
#DrawRes_all('./Mixed', 4)



# Overall functions
# 1) to run and record results, use the following lines
MainFunc('.', 4)
RecRes_sep('.', 'Lizu')
DrawRes_sep('.', 'Lizu')


# 2) after gathering all the results in all scenarios go to the top directory, copy language.csv files there
# to draw figures
direct = '.'
simType = 'Lizu'; empType = 'Duoxu'
cond_PopList = ['P_0_100', 'P_10_90', 'P_20_80', 'P_30_70', 'P_40_60', 'P_50_50', 'P_60_40', 'P_70_30', 'P_80_20', 'P_90_10']
ScenTypeList = ['Scen0', 'Scen1a', 'Scen1b', 'Scen2a', 'Scen2b']
ComRange = np.arange(0,10001,200)

# Based on average results of all runs in the same condition
#colSSD_AvgRun(direct, 0, simType, empType, cond_PopList, ScenTypeList, ComRange)
#drawSSD_cond(direct, 0, 1, cond_PopList, ScenTypeList, ComRange)
#drawSSD_cond(direct, 0, 2, cond_PopList, ScenTypeList, ComRange)

# Based on individual run results in the same condition
SSDtype = 0
colSSD_IndRun(direct, SSDtype, empType, cond_PopList, ScenTypeList, ComRange)
colSSD_Ind_Avg(direct, SSDtype, cond_PopList, ScenTypeList, ComRange)
cond_PopList = ['P_0_100', 'P_10_90', 'P_20_80', 'P_30_70', 'P_60_40', 'P_70_30', 'P_80_20', 'P_90_10']
drawSSD_cond(direct, 0, 1, cond_PopList, ScenTypeList, ComRange)
# part results
cond_PopList = ['P_0_100', 'P_10_90', 'P_20_80', 'P_30_70', 'P_40_60', 'P_50_50', 'P_60_40']
drawSSD_cond(direct, 0, 2, cond_PopList, ScenTypeList, ComRange)
# complete results
#cond_PopList = ['P_0_100', 'P_10_90', 'P_20_80', 'P_30_70', 'P_40_60', 'P_50_50', 'P_60_40', 'P_70_30', 'P_80_20', 'P_90_10']
#drawSSD_cond(direct, 0, 2, cond_PopList, ScenTypeList, ComRange)

# 3) for drawing imposed figure
# to further calculate and show imposed results
# plotImpRes('.', 'Duoxu', 'Lizu', np.arange(0,10001,200))
# calSSDRes('.', 'Lizu', np.arange(0,10001,200))
# for Scenario 2a, draw imposed figure at P_10_90 at 1000
sce = 'Scen2a'; fold = 'P_10_90'; comm = 1000
plotImpRes(os.path.join(direct, sce, fold), 'Duoxu', 'Lizu', [comm])
plotImpRes(os.path.join(direct, sce, fold), 'Duoxu_PCVG', 'Lizu', [comm])
# for Scenario 2b, draw imposed figure at P_20_80 at 200
sce = 'Scen2b'; fold = 'P_20_80'; comm = 200
plotImpRes(os.path.join(direct, sce, fold), 'Duoxu', 'Lizu', [comm])
plotImpRes(os.path.join(direct, sce, fold), 'Duoxu_PCVG', 'Lizu', [comm])

# compared to SVM
sce = 'Scen2a'; fold = 'P_60_40'; comm = 10000
plotImpRes(os.path.join(direct, sce, fold), 'SWM', 'Lizu', [comm])
sce = 'Scen2b'; fold = 'P_50_50'; comm = 10000
plotImpRes(os.path.join(direct, sce, fold), 'SWM', 'Lizu', [comm])

# create AllRes.csv
AllRes = pd.DataFrame()
for scen in ScenTypeList:
    for pop in cond_PopList:
        df = pd.read_csv(os.path.join(direct, scen, 'IndRun_SSD_' + pop + '_' + scen + '.csv'))
        poprate = float(pop.split('_')[1])/100.0
        for runID in range(1,21,1):
            subdf = df[df.runID==runID]
            numCom = int(subdf.loc[subdf.totSSD==min(subdf.totSSD),'numCom'])
            minSSD = float(subdf.loc[subdf.totSSD==min(subdf.totSSD),'totSSD'])
            if scen=='Scen0' or scen=='Scen1a' or scen=='Scen1b': socPres = 0
            else: socPres = 1
            if scen=='Scen0' or scen=='Scen2a' or scen=='Scen2b': markedness = 0
            else: markedness = 1
            if scen=='Scen0' or scen=='Scen1b' or scen=='Scen2b': comtype = 0
            else: comtype = 1 
            AllRes = AllRes.append({'Pop': poprate,
                                    'runID': runID,
                                    'numCom': numCom,
                                    'minSSD': minSSD,
                                    'scen': scen,
                                    'socPres': socPres,
                                    'markedness': markedness,
                                    'comtype': comtype}, ignore_index=True)

AllRes.to_csv(os.path.join(direct, 'AllRes.csv'), index=False)
# then, use AllRes.r to do statistical analyses

# for testing parameter effect, copy this python code into the top folder containing scenario and different parameter value folders
# also copy Duoxu.csv there
# for Fadj (Scen2b_P_10_90/0.001, 0.002, 0.005)
direct = '.'
simType = 'Lizu'; empType = 'Duoxu'
cond_PopList = ['0.001', '0.002', '0.005']
ScenTypeList = ['Scen2b_P_10_90']
ComRange = np.arange(0,10001,200)
# Based on individual run results in the same condition
SSDtype = 0
colSSD_IndRun(direct, SSDtype, empType, cond_PopList, ScenTypeList, ComRange)
colSSD_Ind_Avg(direct, SSDtype, cond_PopList, ScenTypeList, ComRange)
drawSSD_cond(direct, 0, 1, cond_PopList, ScenTypeList, ComRange)

# for Fmkd (Scenb_P_10_90/0.001, 0.002, 0.005)
direct = '.'
simType = 'Lizu'; empType = 'Duoxu'
cond_PopList = ['0.005', '0.01', '0.05']
ScenTypeList = ['Scen1a_P_10_90']
ComRange = np.arange(0,10001,200)
# Based on individual run results in the same condition
SSDtype = 0
colSSD_IndRun(direct, SSDtype, empType, cond_PopList, ScenTypeList, ComRange)
colSSD_Ind_Avg(direct, SSDtype, cond_PopList, ScenTypeList, ComRange)
drawSSD_cond(direct, 0, 1, cond_PopList, ScenTypeList, ComRange)