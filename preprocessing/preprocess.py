import copy
import unidecode
import unicodedata
import itertools
import re
import json
import gensim
import cPickle as pkl
import codecs
import string
import nltk
import sys
import csv
import spacy
from regex4dummies import regex4dummies
from regex4dummies import Toolkit
from annoy import AnnoyIndex
tool_tester = Toolkit()
nlp = spacy.load('en')
sys.path.append('utils')
from nltk_ner import *
from nltk.corpus import stopwords
from clean_utils import read_file_as_dict
stop = set(stopwords.words('english'))
#string.punctuation='!"#$&\'()*+,-./:;<=>?@[\]^_`{|}~ '
from LuceneSearch import *
stopwords = set(stopwords.words('english'))
regex = re.compile('[%s]' % re.escape(string.punctuation))
from question_parser_lucene2 import QuestionParser
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
dataset = json.load(open('data/dataset.json'))
stop_vocab = {}
with open('utils/stopwords.txt') as input_file:
        reader = csv.DictReader(input_file, delimiter='\t', fieldnames=['col1', 'col2'])
	for row in reader:
        	stop_vocab[row['col1']] = int(row['col2'])
stop_set = set([x.lower().strip() for x in stop_vocab.keys()])
stop_set.update([x.lower().strip() for x in stopwords])
with open('utils/all_template_words.txt') as fr:
        stop_set.update([x.strip().lower() for x in fr.readlines()])
stop_set.update(pkl.load(open('utils/all_parent_names.pkl')))
bad_qids = set(['Q184386','Q1541554','Q540955','Q2620241','Q742391'])  #adding Yes/No
bad_qids.update(pkl.load(open('utils/wikidata_entities_with_digitnames.pkl')))
ls = LuceneSearch('utils/lucene_index_9m')
question_parser = QuestionParser(None, stop_vocab, stop_set, bad_qids, ls, False)
wordnet_expansions = json.load(open('utils/wordnet_expansions.json'))
ann_index = AnnoyIndex(300, metric='euclidean')
ann_index.load('utils/annoy_index/glove_embedding_of_vocab.ann')
ann_index_desc = pkl.load(open('utils/annoy_index/index2word.pkl'))
ann_index_desc_inv = {v:k for k,v in ann_index_desc.items()}
skipthoughts_ann_index = AnnoyIndex(4800, metric='euclidean')
skipthoughts_ann_index.load('utils/annoy_index/skipthoughts_embedding_of_vocab.ann')
skipthoughts_ann_index_desc  = pkl.load(open('utils/annoy_index/skipthoughts_index2word.pkl'))
skipthoughts_ann_index_desc_inv = {v:k for k,v in skipthoughts_ann_index_desc.items()}
data = pkl.load(open('data/all_movie_data.pkl','rb'))
strip_unicode = re.compile("([^-_a-zA-Z0-9!@#%&=,/'\";:~`\$\^\*\(\)\+\[\]\.\{\}\|\?\<\>\\]+|[^\s]+)")
coref_resolved_data = json.load(open('data/coref_resolved_plots.json'))
def get_bigger(id):
        try:
                mdata = data[id]
                imdb_len = len(mdata['imdb_plot'])
                wiki_len = len(mdata['wiki_plot'])
                if imdb_len > wiki_len: return mdata['imdb_plot'].replace('\n',' ')
                else: return mdata['wiki_plot'].replace('\n',' ')
        except:
                #traceback.print_exc()
                return None


def clean_encoding(q):
        try:
                q = unidecode.unidecode(q)
        except:
                q = unicodedata.normalize('NFKD', q).encode('ascii','ignore')
        #q = re.sub(' +', ' ', regex.sub(' ', q)).strip()
        q = ' '.join(nltk.word_tokenize(q)).strip()
        if isinstance(q, str) or isinstance(q, basestring):
                q = unicode(q.encode('utf-8'))
        return q

def handle_unicode(x):
        try:
                x = unidecode.unidecode(x)
        except:
                if isinstance(x, unicode):
                        x = unicodedata.normalize('NFKD', x).encode('ascii','ignore')
                else:
                        try:
                                #print 'type 2', type(x)        
                                x = unicodedata.normalize('NFKD', unicode(x, encoding='utf-8')).encode('ascii','ignore')
                        except:
                                try:
                                        #print 'type 3', type(x)        
                                        if isinstance(x, str):
                                                x = x.encode("utf-8")
                                        else:
                                                x = unicode(x, "utf-8").encode("utf-8")
                                except:
                                        #print 'type 4', type(x)
                                        try:
                                                x = unicode(x).encode("ascii", "ignore")
                                                #x = x.encode('utf-8',errors='ignore')
                                        except:
                                                #print 'removing unicode chars', x
                                                x= strip_unicode.sub('', x)
                                                #print 'after removing ',x
        return x

def get_named_entities_noun_verbs(q_sents):
	entities = set([])
	noun_phrases = set([])
	verb_phrases = set([])
	sents_keywords = {}
	sentences  = nltk.sent_tokenize(q_sents)	
	for i,q in enumerate(sentences):
		sent_keywords = set([])
	        ne_q, kbe_q, q_replaced = question_parser.get_utterance_entities(q)
        	#print q ,':', 'KB entities', kbe_q #'REPLACED q', q_replaced
		if isinstance(q, basestring) or isinstance(q, str):
			doc = nlp(unicode(q.encode('utf-8')))
		else:
	        	doc = nlp(q)
	        for ent in doc.ents:
        	        #print '\t', ent.text,'(',ent.label_,')'
               		entities.add(ent.text)
			sent_keywords.add(ent.text)
	        nltk_noun_phrases = extract_entity_names(q)
		ne_q = set(ne_q).union(set(nltk_noun_phrases))
		ne_q = set(ne_q).union(set(kbe_q))	
	        noun_phrases.update(ne_q.difference(set(entities)))
		sent_keywords.update(ne_q)
	        try:
        	       	for x in tool_tester.extract_verb_phrases( text=q, parser="nltk")[0].replace(" and",",").split(","):
                	       	if x.strip().lower() not in stop and len(x.strip())>0:
                                	verb_phrases.add(x)
					sent_keywords.add(x)
		except:
        	  	print '',#raise Exception('error finding verb phrases from '+q)
		sents_keywords[i] = [re.sub(' +', ' ', regex.sub(' ', x.lower())).strip() for x in sent_keywords]
		sents_keywords[i] = [x for x in sents_keywords[i] if len(sents_keywords[i])>1]	
		sent_keywords = None
	sents_keywords_extended = {}
	for i in sents_keywords:
		before  = i-1
		after = i+1
		sents_keywords_extended[i] = set(sents_keywords[i])
		if before in sents_keywords:	
			sents_keywords_extended[i].update(sents_keywords[before])
		if after in sents_keywords:
			sents_keywords_extended[i].update(sents_keywords[after])
	sent_keywords = sents_keywords_extended	
	keywords = set([])
	keywords.update([re.sub(' +', ' ', regex.sub(' ', x.lower())).strip() for x in entities])
	keywords.update([re.sub(' +', ' ', regex.sub(' ', x.lower())).strip() for x in noun_phrases])
	keywords.update([re.sub(' +', ' ', regex.sub(' ', x.lower())).strip() for x in verb_phrases])
	keywords = [x for x in keywords if len(x)>1]
	keywords = set(sorted(keywords))
        return entities, noun_phrases, verb_phrases, keywords, sents_keywords, sentences

def get_bigger(id):
        try:
                mdata = data[id]
                imdb_len = len(mdata['imdb_plot'])
                wiki_len = len(mdata['wiki_plot'])
                if imdb_len > wiki_len: return mdata['imdb_plot'].replace('\n',' '), 'imdb'
                else: return mdata['wiki_plot'].replace('\n',' ')
        except: 
                #traceback.print_exc()
                return None

def get_nnbr_from_glove_index(word):
	return [ann_index_desc[i] for i in ann_index.get_nns_by_item(ann_index_desc_inv[word], 500)]

def get_nnbr_from_skipthoughts_index(word):
	return [skipthoughts_ann_index_desc[i] for i in skipthoughts_ann_index.get_nns_by_item(skipthoughts_ann_index_desc_inv[word], 50)]

def get_wordnet_expansion(words_to_match, keywordset, stemmed_keywordset, sentence, stemmed_sentence):
	matches_found = set([])	
	for ws in words_to_match:
		found_phrase = False
		for w in ws.split(' '):
			stemmed_w = porter_stemmer.stem(w)
			if w in keywordset or stemmed_w in stemmed_keywordset or ' '+stemmed_w+' ' in stemmed_sentence:
				found_phrase = True
				continue	
			if w not in wordnet_expansions:
				continue
			expansions = set(wordnet_expansions[w])
			if w in ann_index_desc_inv:
				glove_expansions = get_nnbr_from_glove_index(w)
				expansions.update(glove_expansions)
			if w in skipthoughts_ann_index_desc_inv:
				skipthoughts_expansions = get_nnbr_from_skipthoughts_index(w)
				expansions.update(skipthoughts_expansions)	
			for expansion in expansions:
				phrase = ws.replace(w, expansion).strip()
				stemmed_phrase = ' '.join([porter_stemmer.stem(x) for x in phrase.split(' ')])
				if phrase in keywordset or ' '+phrase.strip()+' ' in sentence or stemmed_phrase in stemmed_keywordset or ' '+stemmed_phrase.strip()+' ' in stemmed_sentence:
					found_phrase = True
					#print 'found wn expansion ', phrase, ' from ', ws,' (', keywordset, ' ::: ', sentence, ')' 
					break
		if found_phrase:
			matches_found.add(ws)
	return matches_found				


def get_top_relevant_sentences(query_words, query_keywords, sents_keywords, sentences):
	ints_dict = {}
	ints_words = {}
	sent_ints_dict = {}
	num_query_keywords = float(len(query_keywords))
        num_query_words = float(len(query_words))
	for s in sents_keywords:
		keywords = sents_keywords[s]
		stemmed_keywords = [' '.join([porter_stemmer.stem(x) for x in keyword]) for keyword in keywords]
		ints = set(keywords).intersection(set(query_keywords))
		remaining = set(query_keywords) - set(ints)
		chunk = ''
		if s-1 in sentences:
			chunk = chunk+' '+sentences[s-1]
		chunk=chunk+' '+sentences[s]
		if s in sentences:
			chunk = chunk+' '+sentences[s+1]
		chunk = ' '+re.sub(' +', ' ', regex.sub(' ', chunk.lower().strip()))+' '	
		stemmed_chunks = ' '.join([porter_stemmer.stem(x) for x in chunk.split(' ')]).strip()+' '
		matches_found = get_wordnet_expansion(remaining, keywords, stemmed_keywords, chunk, stemmed_chunks)
		ints.update(matches_found)
		words = set([])
		for x in ints:
			words.update([re.sub(' +', ' ', regex.sub(' ', xi.lower())).strip() for xi in x.split(' ')])
		#print 'intersection ', ints
		words = words - stopwords	
		words_len = float(len(words))/num_query_words
		if words_len not in ints_words:
			ints_words[words_len] = []
		ints_words[words_len].append(s)
		ints_len = float(len(ints))/num_query_keywords
		if ints_len not in ints_dict:
			ints_dict[ints_len]=[]
		ints_dict[ints_len].append(s)
		if ints_len>words_len:
			sent_ints_dict[s] = ints
		else:	
			sent_ints_dict[s] = words	
		
	thresh = 0.5
	relevant_sents = []
	for k in reversed(sorted(ints_dict.keys())):
		if k >= thresh:
			relevant_sents.extend(ints_dict[k])
	for k in reversed(sorted(ints_words.keys())):
		if k >= thresh:
			relevant_sents.extend(ints_words[k])	
	if len(relevant_sents)<5:
		thresh = 0.3
		for k in reversed(sorted(ints_dict.keys())):
	                if k >= thresh:
        	                relevant_sents.extend(ints_dict[k])
	        for k in reversed(sorted(ints_words.keys())):
        	        if k >= thresh:
                	        relevant_sents.extend(ints_words[k])
	if len(relevant_sents)<5:
                thresh = 0.1
                for k in reversed(sorted(ints_dict.keys())):
                        if k >= thresh:
                                relevant_sents.extend(ints_dict[k])
                for k in reversed(sorted(ints_words.keys())):
                        if k >= thresh:
                                relevant_sents.extend(ints_words[k])
	relevant_sents_as_paras = []
	para = ''
	last_i = None
	for i in sorted(set(relevant_sents)):
		sent_i = re.sub(' +', ' ', regex.sub(' ', sentences[i].lower())).strip()+'. '
		if last_i is not None and last_i==i-1:
			para = para+sent_i#+'(('+str(sent_ints_dict[i])+'))'
		else:
			if len(para)>0:
				relevant_sents_as_paras.append(para)
			para = sent_i#+'(('+str(sent_ints_dict[i])+'))'
		last_i = i
	return relevant_sents_as_paras

num_q = 0
avg_compression = 0
num_q_with_subplot_extracted = 0
num_q_with_ans_spotted = 0
old_dataset = {}#json.load(open('old_preprocessed.json'))
new_dataset = {}
movies = dataset.keys()
batch_size = int(sys.argv[2])
batch_number = int(sys.argv[1])
min_index = batch_size*batch_number
max_index = min(len(movies),batch_size*(batch_number+1))
for movie in movies[min_index:max_index]:
	#bigger_plot = handle_unicode(get_bigger(movie))
	bigger_plot = handle_unicode(coref_resolved_data[movie])
	if movie in old_dataset:
		old_qs = old_dataset[movie].keys()
		all_qs = [x['q'] for x in dataset[movie]]
		new_qs = set(all_qs) - set(old_qs)
		if len(new_qs)==0:
			print 'continued because len(new_qs)==0'
			continue
	b_plot_entities, b_plot_noun_phrases, b_plot_verb_phrases, b_plot_keywords, b_sents_keywords, b_sentences = get_named_entities_noun_verbs(bigger_plot)
	b_sentences_words = 0.
	for s in b_sentences:
		b_sentences_words += float(len(s.split(' ')))		
	if movie not in new_dataset:
		new_dataset[movie] = {}
	for d in dataset[movie]:
		q = d['q']
		q = handle_unicode(q)
		a_s = d['a']	
		'''if q.lower() in old_dataset[movie]:
			print 'q in old_dataset'
			continue'''
		if len(a_s)==0:
			new_dataset[movie][q] = {"relevant_plot":None, "answers":[]}
			print 'continued because len(a_s)==0'
                        continue
		q = handle_unicode(q)
		original_a_s = copy.deepcopy(a_s)
		a_s = [handle_unicode(a).lower() for a in a_s]
		if 'no answer' in a_s:
			a_s.remove('no answer')
		if None in a_s:
			a_s.remove(None)
		if 'not mentioned' in a_s:
			a_s.remove('not mentioned')
		q_entities, q_noun_phrases, q_verb_phrases, q_keywords, _, _ = get_named_entities_noun_verbs(q)
		q_words = set(re.sub(' +', ' ', regex.sub(' ', x.lower())).strip() for x in q.split(' ')) - set(stopwords)
		if len(q_words)==0:
			 q_words = set(re.sub(' +', ' ', regex.sub(' ', x.lower())).strip() for x in q.split(' '))
		if len(q_keywords)==0:
			q_keywords = q_words
		relevant_sents = get_top_relevant_sentences(q_words, q_keywords, b_sents_keywords, b_sentences)
		if len(relevant_sents)==0:
			new_dataset[movie][q] = {"relevant_plot":relevant_sents, "answers":original_a_s}
			num_q += 1
			print 'continued because len(relevant_sents)==0'
			continue
	
		q = q.lower()
		for w in b_plot_keywords:
			w=w.lower().strip()
			if w in q:
				q_keywords.add(w)
		#print 'question: ',q
		best_overlap_score_words = 0.0
		best_overlap_score_keywords = 0.0
		best_overlap_with_ans_words = None
		best_overlap_with_ans_keywords = None
		a_s_entities = []
		a_s_noun_phrases = []
		a_s_verb_phrases = []
		a_s_keywords = []
		for a in a_s:	
			a_entities, a_noun_phrases, a_verb_phrases, a_keywords, _, _ = get_named_entities_noun_verbs(a)
			a_s_entities.append(a_entities)
			a_s_noun_phrases.append(a_noun_phrases)
			a_s_verb_phrases.append(a_verb_phrases)
			a_s_keywords.append(a_keywords)
			a_words = set(re.sub(' +', ' ', regex.sub(' ', x.lower())).strip() for x in a.split(' ')) - set(stopwords)
			if len(a_words)==0:
				a_words = set(re.sub(' +', ' ', regex.sub(' ', x.lower())).strip() for x in a.split(' '))
			if len(a_keywords)==0:
				a_keywords = a_words
			overlap_with_ans_words = set([w for w in a_words if any([w in x for x in relevant_sents])])
			overlap_with_ans_keywords = set([w for w in a_keywords if any([w in x for x in relevant_sents])])
			overlap_score_words  = (float(len(overlap_with_ans_words))/float(len(a_words)))	
			overlap_score_keywords = (float(len(overlap_with_ans_keywords))/float(len(a_keywords)))
			if best_overlap_score_words <= overlap_score_words:
				best_overlap_score_words = overlap_score_words
				best_overlap_score_keywords = overlap_score_keywords
				best_overlap_with_ans_words = overlap_with_ans_words
				best_overlap_with_ans_keywords = overlap_with_ans_keywords
		words_in_relevant_sents = 0.
		for s in relevant_sents:
			words_in_relevant_sents += float(len(s.split(' ')))
		avg_compression += words_in_relevant_sents/b_sentences_words
		if best_overlap_score_words>0 or best_overlap_score_keywords>0:
			num_q_with_ans_spotted += 1
		num_q += 1
		num_q_with_subplot_extracted += 1
		if num_q%1==0:
                        print 'overlap with ans keywords ',best_overlap_with_ans_keywords, '(',best_overlap_score_keywords,')'
                        print 'fractional length of relevant snippet ', words_in_relevant_sents/b_sentences_words, '(',words_in_relevant_sents,'/',b_sentences_words,')'
			print 'Spotted Answer in ',num_q_with_ans_spotted, ' out of ',num_q, ' with coverage ', max(best_overlap_score_words, best_overlap_score_keywords)	
			print 'Overall Avg Compression as Fraction ', avg_compression/float(num_q_with_subplot_extracted), 'averaging out of ', num_q_with_subplot_extracted
			sys.stdout.flush()
			#print '-------------------------------------------------------------------\n'
		new_dataset[movie][q] = {"relevant_plot":relevant_sents, "answers":original_a_s, "answer_entities":a_s_entities, "answer_nps":a_s_noun_phrases, "answer_vps":a_s_verb_phrases, "answer_keywords":a_s_keywords}
json.dump(new_dataset, open('output/new_preprocessed_'+sys.argv[1]+'.json','w'))		
