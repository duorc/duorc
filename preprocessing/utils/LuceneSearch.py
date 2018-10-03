import re
import nltk
from nltk.corpus import stopwords
import json
import string
import sys, os, lucene
from lucene import *
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader, IndexReader
from org.apache.lucene.index import Term
from org.apache.lucene.search import BooleanClause, BooleanQuery, PhraseQuery, TermQuery
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version
stop = set(stopwords.words('english'))
string.punctuation='!"#$&\'()*+,-./:;<=>?@[\]^_`{|}~ '
regex = re.compile('[%s]' % re.escape(string.punctuation))

class LuceneSearch():
	def __init__(self,lucene_index_dir='lucene_index/', num_docs_to_return=1000):
		lucene.initVM(vmargs=['-Djava.awt.headless=true'])
                directory = SimpleFSDirectory(File(lucene_index_dir))
                self.searcher = IndexSearcher(DirectoryReader.open(directory))
                self.num_docs_to_return =num_docs_to_return
                self.ireader = IndexReader.open(directory)
	
	def strict_search(self, value):
                query = TermQuery(Term("wiki_name",value))
                scoreDocs = self.searcher.search(query, self.num_docs_to_return).scoreDocs
                return scoreDocs
		
	def search(self, value, stopwords=[], min_length=0):
		words = [x for x in nltk.word_tokenize(value) if x not in stopwords and len(x)>min_length]
		query = BooleanQuery()
		query1 = PhraseQuery()
		query1.setSlop(2)
		query2 = PhraseQuery()
                query2.setSlop(2)
		query3 = PhraseQuery()
                query3.setSlop(2)
		for word in words:
			query1.add(Term("wiki_name_analyzed", word))
			query2.add(Term("wiki_name_analyzed_nopunct", word))
			query3.add(Term("wiki_name_analyzed_nopunct_nostop", word))
		query.add(query1, BooleanClause.Occur.SHOULD)
		query.add(query2, BooleanClause.Occur.SHOULD)
		query.add(query3, BooleanClause.Occur.SHOULD)
		scoreDocs = self.searcher.search(query, self.num_docs_to_return).scoreDocs
		if len(scoreDocs)>0:
			#self.printDocs(scoreDocs)
			return scoreDocs
		query = BooleanQuery()
		for word in words:
			query_word = BooleanQuery()
			query_word.add(TermQuery(Term("wiki_name_analyzed", word)), BooleanClause.Occur.SHOULD)
			query_word.add(TermQuery(Term("wiki_name_analyzed_nopunct", word)), BooleanClause.Occur.SHOULD)
			query_word.add(TermQuery(Term("wiki_name_analyzed_nopunct_nostop", word)), BooleanClause.Occur.SHOULD)
			query.add(query_word, BooleanClause.Occur.MUST)
		scoreDocs = self.searcher.search(query, self.num_docs_to_return).scoreDocs
		return scoreDocs
		
	def relaxed_search(self, value, text):
		#print '\nIn Lucene Search for "',value, '" ::: in :::',text, "\n\t",
		value = re.sub(' +', ' ', regex.sub(' ', value.lower())).strip()
		text = re.sub(' +', ' ', regex.sub(' ', text.lower())).strip()
		value_words = set(value.split(' '))
		scoreDocs = self.strict_search(value)
		if len(scoreDocs)>0:
			wiki_entities = self.get_wiki_entities(scoreDocs, value_words, text)
			if len(wiki_entities)>0:
                        	return wiki_entities
		scoreDocs = self.search(value, [])
		if len(scoreDocs)>0:
			wiki_entities = self.get_wiki_entities(scoreDocs, value_words, text)
                        if len(wiki_entities)>0:
                                return wiki_entities
		scoreDocs = self.search(value, stop)
		if len(scoreDocs)>0:
                        wiki_entities = self.get_wiki_entities(scoreDocs, value_words, text)
                        if len(wiki_entities)>0:
                                return wiki_entities	
		scoreDocs = self.search(value, stop, 1)
		if len(scoreDocs)>0:
                        wiki_entities = self.get_wiki_entities(scoreDocs, value_words, text)
                        if len(wiki_entities)>0:
                                return wiki_entities
                return []

	def get_wiki_entities(self, scoreDocs, value_words, text):
		if len(scoreDocs)>100:
                        return []
		entities = []
		for scoreDoc in scoreDocs:
			doc = self.searcher.doc(scoreDoc.doc)
			wiki_id = doc['wiki_id']
			wiki_name = doc['wiki_name']
			doc = doc['wiki_name_analyzed_nopunct']
			doc_words = set(doc.strip().split(' ')) #re.sub(' +', ' ', regex.sub(' ', doc.lower())).strip().split(' '))
			if doc.strip() in text:
				entities.append(wiki_name)
				#print doc+", ",
				continue	
			extra_words = doc_words - value_words
			extra_words = extra_words - stop
			#print 'doc_words ', doc, ' extra ', extra_words
			if len(extra_words)<1:
				entities.append(wiki_name)
				#try:
				#	print doc+", ",
				#except:
				#	continue
		entities = list(set(entities))
		#print entities
		return entities	

	def printDocs(self, scoreDocs):
		for scoreDoc in scoreDocs:
			doc = self.searcher.doc(scoreDoc.doc)
			#for f in doc.getFields():
			#	print f.name(),':', f.stringValue(),',  '
				
			#print ''
		#print '-------------------------------------\n'

if __name__=="__main__":
	ls = LuceneSearch('/dccstor/cssblr/amrita/dialog_qa/code/prepro_lucene/lucene_index_new/')
	for line in open('phrases_to_test_lucene.txt').readlines():
		line = line.strip()
		ls.relaxed_search(line)
