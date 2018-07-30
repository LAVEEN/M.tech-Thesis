import sys
import operator

def populateData(inputFile):
	sentences=[]
	for line in inputFile:
		sentences.append(line)
	return sentences

def readAllFiles():
	CRF	  = open(sys.argv[1])
	K_Mix	  = open(sys.argv[2])
	inputFile = open(sys.argv[3])
	return CRF,K_Mix,inputFile



def postProcessing():
	if len(sys.argv)!=4:
		print 'Please Enter filename1 and filename2  and filename3 where filename1 is the output of CRF and filename2 is output of k-mixture model and filename3 is input file names'
		exit()

	CRF,K_Mix,inputFile=readAllFiles()

	sentences=populateData(inputFile)
	labels=populateData(CRF)
	sentenceAndWeight=populateData(K_Mix)
	visitedLabels={}
	ans={}
	for line in sentenceAndWeight:
		tokens=line.split('\t')
		sentId=int(tokens[0])
		label=labels[sentId-1]
		if label not in visitedLabels:
			ans[sentences[sentId-1]]=label
			visitedLabels[label]=1
		elif visitedLabels[label]==2:
			continue
		else:
			visitedLabels[label]=2
			ans[sentences[sentId-1]]=label
			

	
	sortedAns = sorted(ans.items(), key=operator.itemgetter(1),reverse=True)
	for key,value in sortedAns:
		print '--------------------------------------------------------------------'
		print value
		print '--------------------------------------------------------------------'
		print key

postProcessing()


