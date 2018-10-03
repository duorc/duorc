#!/usr/bin/python

import argparse
import csv
import cPickle as pkl
import nltk, re, copy
from nltk.corpus import stopwords
import fnmatch
import codecs, json
import unidecode
import unicodedata
import timeit
from words2number import *
from LuceneSearch import *
stopwords = set(stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
#Taken from Su Nam Kim Paper...
grammar = r"""
      NBAR:
          {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
          
      NP:
          {<NBAR>}
          {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
  """
chunker = nltk.RegexpParser(grammar)

def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

def search(haystack, needle):
    """
    Search list `haystack` for sublist `needle`.
    """
    if len(needle) == 0:
        return 0
    char_table = make_char_table(needle)
    offset_table = make_offset_table(needle)
    i = len(needle) - 1
    while i < len(haystack):
        j = len(needle) - 1
        while needle[j] == haystack[i]:
            if j == 0:
                return i
            i -= 1
            j -= 1
        i += max(offset_table[len(needle) - 1 - j], char_table.get(haystack[i]));
    return -1


def make_char_table(needle):
    """
    Makes the jump table based on the mismatched character information.
    """
    table = {}
    for i in range(len(needle) - 1):
        table[needle[i]] = len(needle) - 1 - i
    return table

def make_offset_table(needle):
    """
    Makes the jump table based on the scan offset in which mismatch occurs.
    """
    table = []
    last_prefix_position = len(needle)
    for i in reversed(range(len(needle))):
        if is_prefix(needle, i + 1):
            last_prefix_position = i + 1
        table.append(last_prefix_position - i + len(needle) - 1)
    for i in range(len(needle) - 1):
        slen = suffix_length(needle, i)
        table[slen] = len(needle) - 1 - i + slen
    return table

def is_prefix(needle, p):
    """
    Is needle[p:end] a prefix of needle?
    """
    j = 0
    for i in range(p, len(needle)):
        if needle[i] != needle[j]:
            return 0
        j += 1    
    return 1

def suffix_length(needle, p):
    """
    Returns the maximum length of the substring ending at p that is a suffix.
    """
    length = 0;
    j = len(needle) - 1
    for i in reversed(range(p + 1)):
        if needle[i] == needle[j]:
            length += 1
        else:
            break
        j -= 1
    return length


def isint(x):
	try:
		int(x)
		return True
	except:
		try:
			text2int(x.strip())
			return True
		except:		
			return False	
	
def get_continuous_chunks(text, all_possible_ngrams, stop, split_into_commas=False):
  text = text.lower()
  text = unicodedata.normalize('NFKD', unicode(text, "utf-8")).encode('ascii','ignore')
  toks = nltk.word_tokenize(text)
  return_chunks = []
  return_chunks_tokenized = []
  if split_into_commas and "," in text:
        for chunk in text.split(","):
                return_chunks.append(chunk)
                chunk_toks = nltk.word_tokenize(chunk)
                return_chunks_tokenized.append(chunk_toks)   
  if all_possible_ngrams:
	ngrams = set([])
   	ngrams.update([x for x in toks if x not in stop and not isint(x)])
	ngrams.update([' '.join(list(x)) for x in nltk.bigrams(toks) if len(set(x)-stop)>0])
	for ngram_counter in range(3,6):	
	   	ngrams.update([' '.join(list(x)) for x in nltk.ngrams(toks,ngram_counter) if len(set(x)-stop)>0])
        return_chunks=ngrams
	return_chunks_tokenized=[x.split(" ") for x in ngrams]
  else: 	
  	postoks = nltk.pos_tag(toks)
  	tree = chunker.parse(postoks)
  	# print tree
	super_list = [w for w,t in tree.leaves()]

  	for subtree in tree.subtrees():
	    # print subtree
	    if subtree==tree:
	      continue
	    chunk_list = [x[0].strip() for x in subtree.leaves()]
	    chunk = ' '.join(chunk_list).strip()
	    if len(chunk)<=1:
	      continue
	    if chunk not in return_chunks:
	      if chunk not in return_chunks:
		      return_chunks.append(chunk)
		      return_chunks_tokenized.append(chunk_list)
    # values.add(chunk)

  return return_chunks, return_chunks_tokenized, toks


class QuestionParser(object):
  def __init__(self, valid_entities_set, stop_vocab, stop_set, bad_qids, ls, all_possible_ngrams):
    # self.valid_entities_set = valid_entities_set
    self.stop_vocab = stop_vocab
    self.stop_set = stop_set
    self.bad_qids = bad_qids	
    '''	
    self.stop_set = set([x.lower() for x in self.stop_vocab.keys()])		
    self.stop_set.update(stopwords)
    with open('/dccstor/cssblr/vardaan/dialog-qa/dict_val/all_template_words.txt') as fr:
	self.stop_set.update([x.strip().lower() for x in fr.readlines()])		
    self.stop_set.update(pkl.load(open('/dccstor/cssblr/vardaan/dialog-qa/all_parent_names.pkl')))	
    '''
    self.ls = ls
    self.all_possible_ngrams = all_possible_ngrams
    #with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/wikidata_fanout_dict.json','r','utf-8') as data_file:
    #  self.wikidata_fanout_dict = json.load(data_file)
    print 'Successfully loaded wikidata_fanout_dict json'

  def remove_all_stopwords_except_one(self, qn_entities):
    qn_entities_clean = set([])
    # Remove stop entities
    for entity in qn_entities:
      if entity not in self.stop_vocab:
        qn_entities_clean.add(entity)
    #If all entities were stopwords, keep the one with least freq.
    if len(qn_entities_clean) == 0:
      least_freq_stop_entity = qn_entities[0]
      for entity in qn_entities:
        if self.stop_vocab[entity] < self.stop_vocab[least_freq_stop_entity]:
          least_freq_stop_entity = entity
      qn_entities_clean.add(least_freq_stop_entity)
    return qn_entities_clean

  def remove_substrings(self, qn_entities):
    if len(qn_entities) > 1:
      qn_entities_clean = set([])
      #Remove entities contained in other entities
      for entity1 in qn_entities:
        is_substring_of = False
        for entity2 in qn_entities:
          if entity1 == entity2:
            continue
          if entity1 in entity2:
            is_substring_of = True
            break
        if not is_substring_of:
          qn_entities_clean.add(entity1)
      qn_entities = qn_entities_clean
    return list(qn_entities)

  def get_sets_after_difference(self, s1, s2):
    """ Returns difference s1-s2 and s2-s1 where minus sign indicates set difference"""
    intersection = s1.intersection(s2)
    for e in intersection:
      s1.remove(e)
      s2.remove(e)
    return s1, s2

  def get_sets_after_removing_stopwords(self, s1, s2):
    score1, score2 = 0, 0
    for word in list(s1):
      if word in self.stop_vocab:
        score1 = score1 + self.stop_vocab[word]
        s1.remove(word)
    for word in list(s2):
      if word in self.stop_vocab:
        score2 = score2 + self.stop_vocab[word]
        s2.remove(word)
    return s1, score1, s2, score2

  def remove_spurious_entities(self, qn_entities, utterance):
    #Remove spurious entities
    if len(qn_entities) > 1:
      qn_entities_clean = set([])
      for entity1 in qn_entities:
        for entity2 in qn_entities:
          if entity1 == entity2:
            continue
          s1, s2 = set(entity1.split(" ")), set(entity2.split(" "))
          pos1, pos2 = utterance.find(entity1), utterance.find(entity2)
          intersection = s1.intersection(s2)
          #If there is no intersection, none of them can be spurious and cannot be removed
          if len(intersection) == 0:
            qn_entities_clean.add(entity1)
          if pos1 < pos2 and pos1 + len(entity1) > pos2:
            #e1 lies to the left of e2 and e1 and e2 were picked from spatially intersecting windows
            s1, s2 = self.get_sets_after_difference(s1, s2)
            s1, score1, s2, score2 = self.get_sets_after_removing_stopwords(s1, s2)

            #Case 1: e1 is not spurious
            if len(s1) > 0:
              qn_entities_clean.add(entity1)
            #Case 2: e2 is not spurious
            if len(s2) > 0:
              qn_entities_clean.add(entity2)
            #Case 3: Both turn out to be spurious, pick the one with lower score, (less freq. stopwords)
            if len(qn_entities_clean) == 0:
              if score1 < score2:
                qn_entities_clean.add(entity1)
              else:
                qn_entities_clean.add(entity2)
      qn_entities = qn_entities_clean
    return list(qn_entities)

  def substringSieve(self, string_list, words_dict):
    string_list.sort(key = lambda s: len(words_dict[s]),reverse=True)
    #out = []
    #for s in string_list:
    #    s = s.strip()
    #    if not any([s in o for o in out]):
    #        out.append(s)
    return string_list
  
  def get_NER(self,text):
        text = unicodedata.normalize('NFKD', unicode(text, "utf-8")).encode('ascii','ignore').lower()
	toks = nltk.word_tokenize(text)
        text_toks = ' '+' '.join(toks)+' '
        values = set([])
        print 'in Target ', text, ':: ',
        if "," in text:
                text_split = text.split(",")
                for phrase in text_split:
                        phrase = phrase.strip().lower()
                        if phrase not in self.stop_set and not isint(phrase) and len(set(phrase.split(" "))-self.stop_set)>0:
                                values.add(phrase)
        qn_entities = [] #QIDs
        search_dict  = {}
        ent_name_list = []
        for ent in values:
          ent_list = self.ls.relaxed_search(ent, text_toks)
	  #print ' In lucene ', ent ,'--->',ent_list
          for e in ent_list:
                if e in self.bad_qids:
                        continue
                search_dict[e] = ent
                if e not in ent_name_list:
                        ent_name_list.append(e)
        ent_name_list.sort(key = lambda s: len(search_dict[s].split(' ')),reverse=True)
        e_words_found = []
        text_replaced = text_toks
        for e in ent_name_list:
                e_words = ' '+search_dict[e].strip()+' '
                if e_words in text_replaced or e_words in e_words_found:
                        text_replaced = text_replaced.replace(e_words, ' '+e.strip()+' ')
                        e_words_found.append(e_words)
                        qn_entities.append(e)
	if len(e_words_found)>0:
	        print 'final entities found  ',set(e_words_found),
        if len(qn_entities)>0:
	    print ''	
            return list(set(qn_entities))
        if self.all_possible_ngrams:
                values.update([x for x in toks if x not in self.stop_set and not isint(x)])
                values.update([' '.join(list(x)) for x in nltk.bigrams(toks) if len(set(x)-self.stop_set)>0])
                for ngram_counter in range(3,6):
                        values.update([' '.join(list(x)) for x in nltk.ngrams(toks,ngram_counter) if len(set(x)-self.stop_set)>0])
                values

        else:
                values = set([])
                toks = nltk.word_tokenize(text)
                postoks = nltk.pos_tag(toks)
                tree = chunker.parse(postoks)
                # print tree
                for subtree in tree.subtrees():
                    if subtree == tree:
                        continue
                    chunk = ' '.join([x[0].strip() for x in subtree.leaves()]).strip()
                    if len(chunk) <= 1:
                        continue
                    values.add(chunk)
        for ent in values:
	  ent_list = self.ls.relaxed_search(ent, text_toks)
          for e in ent_list:
                search_dict[e] = ent
                if e not in ent_name_list:
                        ent_name_list.append(e)
        ent_name_list.sort(key = lambda s: len(search_dict[s].split(' ')),reverse=True)
        e_words_found = []
        text_replaced = text_toks
        for e in ent_name_list:
                e_words = ' '+search_dict[e].strip()+' '
                if e_words in text_replaced or e_words in e_words_found:
                        text_replaced = text_replaced.replace(e_words, ' '+e.strip()+' ')
                        e_words_found.append(e_words)
                        qn_entities.append(e)
	if len(e_words_found)>0:
	        print 'final entities found   ',set(e_words_found),','
	print ''
        return list(set(qn_entities))

  def get_utterance_entities(self, utterance, split_into_commas=False):
    # qn_entities = []
    # q_words = utterance.split(" ")
    # max_gram = ""
    # for n in range(1, len(q_words) + 1):
    #   i = 0
    #   while i + n <= len(q_words):
    #     gram = q_words[i:i + n]
    #     gram = " ".join(gram)
    #     if gram in self.valid_entities_set:
    #       qn_entities.append(gram)
    #     i = i + 1

    #remove stop entities, substrings, spurious entities
    # print utterance
    all_entities = set([])
    qn_entities = [] # QIDs global list
    try:
	utterance = unicodedata.normalize('NFKD', unicode(utterance, "utf-8")).encode('ascii','ignore')  
    except:
	try:
		utterance = unicodedata.normalize('NFKD', utterance).encode('ascii','ignore')  
	except:
		try:	
			if isinstance(utterance, basestring):
				utterance = utterance.encode("utf-8")	
			else:
				utterance = unicode(utterance, "utf-8").encode("utf-8")
		except:
			utterance = utterance.encode('ascii', errors='ignore')	
			
    utter_token_list = [] # contains tree-tokenized version of each of the utterances
    utter_dict_list = [] # for each utter, it contains a dict, each key is a NE and corr. value is the tokenization of that entityW

    utterance_chunks = utterance.split('|')

    out_utter_list = []

    for utter_chunk in utterance_chunks:
      chunks, chunks_tokenized, list1 = get_continuous_chunks(utter_chunk, self.all_possible_ngrams, self.stop_set, split_into_commas)
      entity_list = chunks
      high_rank_qn_entities = [] # contains rank-1 entity of lucene search QIDs
      search_dict  = {}
      
      for i,ent in enumerate(entity_list):
        ent_name_list = self.ls.relaxed_search(ent, utterance)
        #print ent_name_list
	all_entities.add(ent.strip())
        if len(ent_name_list) > 0:
	  qn_entities.extend(ent_name_list)
	  #all_entities.add(ent.strip())		
	  for e in ent_name_list:
		if e in self.bad_qids:
			continue
		high_rank_qn_entities.append(e)
		search_dict[e] = chunks_tokenized[i]
	  '''
          #ent_name_list_fanout = [self.wikidata_fanout_dict[x] for x in ent_name_list]
	  #ent_name_list_fanout_max = sorted(ent_name_list_fanout, reverse=True)[:10]
	  for max_ent in ent_name_list:_fanout_max:
	          ent_max_fn = ent_name_list[ent_name_list_fanout.index(max_ent)]
	          # qn_entities.extend(ent_name_list)
        	  qn_entities.extend([ent_max_fn])
	          if ent_max_fn not in high_rank_qn_entities:
        	    	high_rank_qn_entities.append(ent_max_fn)
	            	search_dict[ent_max_fn] = chunks_tokenized[i]
	  '''
      word_list = ' '+' '.join(list1).strip()+ ' '
      high_rank_qn_entities = self.substringSieve(high_rank_qn_entities, search_dict)	
      #print 'high_rank_qn_entities ', high_rank_qn_entities
      e_words_found = []
      for e in high_rank_qn_entities:	
	   word_e = ' '+' '.join(search_dict[e]).strip()+' '
	   #print 'searching \"'+word_e+'\" in \"'+word_list+'\"'
	   if word_e in word_list or word_e in e_words_found:
		   word_list = word_list.replace(word_e, ' '+e.strip()+' ')
		   if e not in qn_entities:
			   qn_entities.append(word_e.strip())
		   e_words_found.append(word_e)	
      #print word_list	
      out_utter_list.append(word_list.strip())	
      '''
      #print high_rank_qn_entities	
      valid_high_rank_qn_entities = []
      valid_high_rank_qn_entities_idx = []
      valid_high_rank_qn_entities_len = []

      last_idx = 0
      prev_token_len = 0

      for ent in high_rank_qn_entities:
        ent_tokens = search_dict[ent]
        utter_tokens = list1

        idx = search(utter_tokens[last_idx:], ent_tokens)

        valid_high_rank_qn_entities.append(ent)
        valid_high_rank_qn_entities_idx.append(idx + last_idx)
        valid_high_rank_qn_entities_len.append(len(ent_tokens))

        prev_token_len = len(ent_tokens)
        last_idx = idx + prev_token_len

      print valid_high_rank_qn_entities
      print valid_high_rank_qn_entities_idx
      print valid_high_rank_qn_entities_len	
      # print 'ent_tokens = %s' % ' '.join(ent_tokens)
      # print 'utter tokens = %s' % ' '.join(nltk.word_tokenize(utterance))
      # sys.exit(0)
      # print [item_data[q] for q in valid_high_rank_qn_entities]

      perm = sorted(range(len(valid_high_rank_qn_entities_idx)), key=lambda k: valid_high_rank_qn_entities_idx[k], reverse=True)

      valid_high_rank_qn_entities = [valid_high_rank_qn_entities[i] for i in perm]
      valid_high_rank_qn_entities_idx = [valid_high_rank_qn_entities_idx[i] for i in perm]
      valid_high_rank_qn_entities_len = [valid_high_rank_qn_entities_len[i] for i in perm]

      ques_ptr = 0
      l3 = []
      utterance_tokens = list1
      i = 0

      print valid_high_rank_qn_entities
      print valid_high_rank_qn_entities_idx
      print valid_high_rank_qn_entities_len

      while i < len(valid_high_rank_qn_entities):
        # print 'i = %d, utter = %s' % (i,utter_chunk)
        if ques_ptr < valid_high_rank_qn_entities_idx[i]:
          # print utterance_tokens[ques_ptr : valid_high_rank_qn_entities_idx[i]]
          l3.extend(utterance_tokens[ques_ptr : valid_high_rank_qn_entities_idx[i]])
          ques_ptr = valid_high_rank_qn_entities_idx[i]
          # print 'ques_ptr = %d' % ques_ptr
        elif ques_ptr == valid_high_rank_qn_entities_idx[i]:
          l3.append(valid_high_rank_qn_entities[i])
          ques_ptr += valid_high_rank_qn_entities_len[i]
          i += 1
        else:
          break
        # else:
          print 'ERROR!!!!'
          print 'ques = %s ent = %s' % (utterance, valid_high_rank_qn_entities[i])

      l3.extend(utterance_tokens[ques_ptr:])

      out_utter_list.append(l3)
      '''
    #:print 'for :' ,utterance, ' found entities ', set(e_words_found)
    return all_entities, qn_entities, '|'.join(out_utter_list)



if __name__=="__main__":
	stop_vocab = {}
	with open('/dccstor/cssblr/vardaan/neural_kbqa_wikidata/data/movieqa/stopwords.txt') as input_file:
	    reader = csv.DictReader(input_file, delimiter='\t', fieldnames=['col1', 'col2'])
	    for row in reader:
	      stop_vocab[row['col1']] = int(row['col2'])
	stop_set = set([x.lower().strip() for x in stop_vocab.keys()])
        stop_set.update([x.lower().strip() for x in stopwords])
        with open('/dccstor/cssblr/vardaan/dialog-qa/dict_val/all_template_words.txt') as fr:
                stop_set.update([x.strip().lower() for x in fr.readlines()])
        stop_set.update(pkl.load(open('/dccstor/cssblr/vardaan/dialog-qa/all_parent_names.pkl')))
	bad_qids = set(['Q184386','Q1541554','Q540955','Q2620241','Q742391'])  #adding Yes/No
        bad_qids.update(pkl.load(open('wikidata_entities_with_digitnames.pkl')))
	ls = LuceneSearch('/dccstor/cssblr/amrita/dialog_qa/code/prepro_lucene/lucene_index_new')
	question_parser = QuestionParser(None, stop_vocab, stop_set, bad_qids, ls, True)
	#ques_entities, context_list = question_parser.get_utterance_entities(context, True)
	for line in open('train_old2/QA_144/QA_144_orig_response.txt').readlines():
		line = line.strip()
		question_parser.get_NER(line)
		
