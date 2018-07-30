# ==================================================================================#
# author            : Laveen Ekka 		                   	    			    						#
# roll_number       : 17CS60R64   		                   	    			    						#
# date              : 5th  June 2018      	                            						#
# usage             : python3 Program.py  			   			                						#
# python_version    : 3.6.5                                             						#
# library used      : 1) NLTK for preprocessing	                        						#	 
#		    	            2) Rouge for precision , recall and score calculation		      #
#											3) Operator for sorting dictionary value by key and values		#
#											4) html2text																									#
# ==================================================================================#
import sys
import os
import math
import html2text
import operator
import itertools
import operator
from bs4 import BeautifulSoup
from nltk import tokenize
from nltk.corpus import stopwords, wordnet as wn
stopword_set = set(stopwords.words('english'))


stopword_set.add('[')


#===============================================#
#					Extractor Data												#
#===============================================#
#				Extract required data from html body		#
#===============================================#


def extractorData(html):
	# dummy list
	idx=0
	tempData= tokenizeSentence(html.decode('utf-8'))

	while 'for educational use only' not in tempData[idx]:
		idx=idx+1
	ans=[]
	for i in range(idx,len(tempData)):
		if '2015 thomson reuters south asia private limited' in tempData[i]:
			break
		ans.append(tempData[i])


	return ans




#Group By Key

def accumulate(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
       yield key, sum(item[1] for item in subiter) 


# 1st Tokenize and form sentence
def tokenizeSentence(fileContent):
    """
    :param fileContent: fileContent is the file content which needs to be summarized
    :return: Returns a list of string(sentences)
    """
    return tokenize.sent_tokenize(fileContent)


# 2nd Case Folding
def caseFolding(line):
    """
    :param line is the input on which case folding needs to be done
    :return: Line with all characters in lower case
    """
    return line.lower()


# 3rd Tokenize and form tokens from sentence
def tokenizeLine(sentence):
    """
    :param sentence: Sentence is the english sentence of the file
    :return: List of tokens
    """
    return tokenize.word_tokenize(sentence)


# 4th Stop Word Removal
def stopWordRemove(tokens):
    """
    :param tokens: List of Tokens
    :return: List of tokens after removing stop words
    """
    list_tokens = []
    for token in tokens:
        if (token not in stopword_set) and (token != '.') and (token != ','):
            list_tokens.append(token)
    return list_tokens


# 5th token Streammer
def tokenStemmer(tokens):
    """
    :param tokens: Sentence is the english sentence of the file
    :return: List of tokens after performing stemming
    """
    return [stemmer.stem(token) for token in tokens]



def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

# Don't process <img> tags, just strip them out. Use an indent of 4 spaces 
# and a page that's 80 characters wide.



ls=os.popen('ls ./train').read()

tr=ls.split('\n')
listOfData=[]
maxof=0
nameof=None
for line in tr:
	if line!='':
		print 'Reading...'+ line
		data=None
		with open('./train/'+line, 'r') as myfile:
			data = myfile.read()				
		#index=find_nth(data,'<p class="indent1">',0)
		#print index
				
		#for i in range(0,len(fp)):
		#	if 			
		#	<p class="indent1">		
		#for ln in fp:
		#	if '2015 Thomson Reuters South Asia Private Limited' in ln:
		#	print ln
		#soup = BeautifulSoup(fp)
		#print soup.get_text()
		#break
		#print(html2text.html2text(data))
		#text = html2text.html2text(fp)
		#print '134221452751'.replace('1','')
		data=(data.replace('</?(?!(?:p class=indent)\b)[a-z](?:[^>\"\']|\"[^\"]*\"|\'[^\']*\')*>','').decode('utf-8'))
		if(len(data)>maxof):
			maxof=len(data)
			nameof=line
		listOfData.append(caseFolding(html2text.html2text(data).encode('utf-8')))


print nameof
print maxof


if len(sys.argv)==2:		
	fileInput=open(sys.argv[1])
else:
	print 'Please Prove file Name for k mix model'
	exit()


mysentences=[]
mysentences_id=None
idx=1

for line in fileInput:
	mysentences.append(line)
	

for i in range(0,len(mysentences)):
	myline=mysentences[i]
	mysentences[i]=caseFolding((myline).encode('utf-8'))


index={}
listOfSentences=[]
docId=1
startReading=False








for File in listOfData:
	sentences=extractorData(File)
	for sentence in sentences:
		sentence=sentence.strip()
		if len(sentence)==0:
			continue
		else:
			tokens=tokenizeLine(sentence)
			final_tokens=stopWordRemove(tokens)
			for token in final_tokens:
				if token not in index:
					mylist=[]
					mylist.append((docId,1))
					index[token]=mylist
				else:
					val=index[token]
					val.append((docId,1))
					index[token]=val

	docId=docId+1
	




for key in index:
	getList=index[key]
	index[key]=list(accumulate(getList))






#print index
#print len(mysentences)

#for line in sentences:
#	print line
#print listOfData

N=360
cf=0
tf=0
df=0
num= math.log(N)/math.log(2)
kmixresult={}
sent_id=1
for sentence in mysentences:
	pk=0

	myline=tokenizeLine(sentence)
	final_tokens=stopWordRemove(myline)

	for token in final_tokens:
		if token not in index:
			continue
		mylist=index[token]
		cf=0		
		for x,y in mylist:
			if x==1:
				tf=y
				df=len(mylist)
				
			cf=cf+y
		#print 'token='+token
		t= cf*1.0/N
		#print 't='+str(t)
		if df!=0:
			idf=num*1.0/df
		else:
			idf=0
		#print 'idf='+str(idf)
		if df!=0:		
			s=(((cf-df)*1.0)/df)
		else:
			s=0
		#print 's='+str(s)
		r=1
		if s==0:
			r=0
		else:
			r=t/s
		#print 'r='+str(r)

		pnum=r/(s+1)
		pmult=s/(s+1)
		pmult=math.pow(pmult,tf)
		pk=pnum*pmult+pk

	#print sentence
	#print pk
	kmixresult[sent_id]=pk
	sent_id=sent_id+1


sorted_d = sorted(kmixresult.items(), key=operator.itemgetter(1),reverse=True)

oFile = open('k-mix-output.txt', 'w')
for key,value in sorted_d:
	oFile.write(str(key)+'\t'+str(value)+'\n')
oFile.close()
