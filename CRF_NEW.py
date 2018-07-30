# ======================================================================#
# author            : Laveen Ekka 		                   	    			    #
# roll_number       : 17CS60R64   		                   	    			    #
# date              : 5th  June 2018      	                            #
# usage             : python3 Program.py  			   			                #
# python_version    : 3.6.5                                             #
# library used      : 1) NLTK for preprocessing	                        #	 
#		      2) Rouge for precision , recall and score calculation		      #
# ======================================================================#


import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from nltk import tokenize
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
#cue1   We agree with court
#cue2   Question for consideration is
import sys




def getLabelFromList(anList):
	label='-'
	for element in anList:
		if element =='-':
			continue
		else:
			label=element
	if label=='EFOTC':
		label='ESTABLISHING THE FACTS OF THE CASE'
	elif label=='ATC':
		label='ARGUING THE CASE'
	elif label=='HISTORY':
		label='HISTORY OF THE CASE'
	return label


def tokenizeLine(sentence):
    """
    :param sentence: Sentence is the english sentence of the file
    :return: List of tokens
    """
    return tokenize.word_tokenize(sentence)



def getListOfLabels(inputLine):
	mylist=[]
	tokens=tokenizeLine(inputLine)
	if len(tokens)==1:
		return []
	for token in tokens:
		spl_tkn=token.split('=')
		if len(spl_tkn)>1:
			if spl_tkn[0]=='EFOTC':
				mylist.append((spl_tkn[1],'EFOTC'))
			elif spl_tkn[0]=='RATIO':
				mylist.append((spl_tkn[1],'RATIO'))
			elif spl_tkn[0]=='ATC':
				mylist.append((spl_tkn[1],'ATC'))
			elif spl_tkn[0]=='ARGUMENTS':
				if spl_tkn[1]=='vs':
					spl_tkn[1]=spl_tkn[1]+'.'
				mylist.append((spl_tkn[1],'ARGUMENTS'))
			elif spl_tkn[0]=='HISTORY':
				mylist.append((spl_tkn[1],'HISTORY'))
			else:			
				mylist.append((token,'-'))
		else:			
				mylist.append((token,'-'))
		
	return mylist




def word2features(sent, i):
	word = sent[i][0]
	postag = sent[i][1]
	features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.cue1': False,
	'word.RATIO':False,
	'word.EFOTC':False,
	'word.ATC':False,
	'word.HISTORY':False,
	'word.ARGUMENTS':False
    }



	if i>=1:
		word2=sent[i][0]
		word1=sent[i-1][0]
		#[fact/fact's that] / [rent of] / [appellate court]
		value=((word1=='fact' or word2=='fact\'s') and (word2=='that' or word2=='of')) or ((word1.lower()=='rent') and word2=='of') or ((word1.lower()=='appellate') and word2=='court') or ((word1.lower()=='found') and word2=='that')

		if value==True:
			features.update({'word.EFOTC':True})		
		
		#[In View]
		value=(word1.lower()=='in' and word2.lower()=='view')

		if value==True:
			features.update({'word.RATIO':True})

	if word.lower()=='proved':
		features.update({'word.EFOTC':True})		
	
	if word=='holding' or word=='according':
		features.update({'word.RATIO':True})
	
	if word.lower()=='s.c.c':
		features.update({'word.ATC':True})

	if word.lower()=='dismissed':
		features.update({'word.HISTORY':True})

	if word.lower()=='vs' or word.lower()=='vs.':
		features.update({'word.ARGUMENTS':True})

	return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]







train_sents=[]



mylist=[]

if len(sys.argv)!=3:
	print 'Please enter fileName1 and fileName2 as command line arguments\n Where fileName1 is annotated Corpus and fileName2 is doc name that is to be annotated'
	exit(0)


myfile=open(sys.argv[1])

for line in myfile:
	listOf=getListOfLabels(line)

	if len(listOf)!=0:
		train_sents.append(listOf)

test_sents=[]

myfile=open(sys.argv[2])

for line in myfile:
	tokens=tokenizeLine(line)
	print tokens
	for token in tokens:
		mylist.append((token,'-'))
	
	test_sents.append(mylist)
	mylist=[]




#print train_sents

#print sent2features(train_sents[0])[3]

#print sent2labels(train_sents[0])[1]
	

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

print y_train

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', 
    c1=0.1, 
    c2=0.1, 
    max_iterations=100, 
    all_possible_transitions=True
)
crf.fit(X_train, y_train)


labels = list(crf.classes_)

X_test=[sent2features(s) for s in test_sents]
y_test= [sent2labels(s) for s in test_sents]

#print y_test

y_pred = crf.predict(X_test)
#print y_pred

print metrics.flat_f1_score(y_test, y_pred,average='weighted', labels=labels)



oFile = open('CRF-output.txt', 'w')

for i in range (0,len(y_pred)):
	label=None
	if i==0:
		label='IDENTIFYING THE CASE'
	elif i==len(y_pred)-1:
		label='FINAL DECISION'
	else:
		label=getLabelFromList(y_pred[i])
		if label=='-':
			label='HISTORY OF THE CASE'
	oFile.write(label+'\n')	

oFile.close()



