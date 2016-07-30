from collections import defaultdict
from collections import Counter
import unicodedata
from unidecode import unidecode


stopwords_list = ["a", "about", "above", "above", "does", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
#stopwords_list = ["where", "what", "name", "a", "is", "of", "who", "why", "when", "was", "which", "what's"]
verb_list = ["the", "does"]
def remove_stopwords(sentence):
	sentence = sentence.replace("?", "")
	sentence = sentence.replace("'", ' ')
	items = sentence.split(' ')
	cleaned_string = map(lambda item:clean_token(item, stopwords_list), items)
	string_w_spaces = ' '.join(cleaned_string)
	raw_tokens = string_w_spaces.split()
	single_space_string = ' '.join(raw_tokens)

	if len(raw_tokens) < 1:
		return sentence
	else:
		return single_space_string

def clean_token(token, stopwords_list):
	if token.lower() in stopwords_list or token == "s" or token in verb_list:
		return ""
	else:
		return token

def replace_accents(token):
	""" Replaces accents with special value """
	utf8_str = token.decode('utf-8')
	normalized_str = unidecode(utf8_str)
	return normalized_str

def clean_name_arr(name_arr, query):
	""" Only returns values from name_dict whose keys are a substring of query 
		name_dict: maps names to ids, keys
	"""
	correct_names = []

	query = query + " "
	lowercase_query = query.lower()
	quote_removed_query = lowercase_query.replace('\\"', '')
	question_removed_query = lowercase_query.replace('?', '')
	quote_removed_question_query = lowercase_query.replace('"', '').replace('?', '')

	for k in name_arr:
		spaced_k = k.lower() + " "
		if spaced_k in lowercase_query or \
		spaced_k in quote_removed_query or \
		spaced_k in question_removed_query or \
		spaced_k in quote_removed_question_query:
			correct_names.append(k)

	return correct_names

def remove_substrings_arr(substring_arr):
	""" Remove any string in array that is a substring in another string 
	"""
	substring_set = set(substring_arr)
	filtered_items = filter(lambda item: not is_substring(item, substring_set), substring_arr)
	return filtered_items

def is_substring(string, string_set):
	"""
	Returns true if string is a substring of any string in 
	string_set that is not equal to string
	"""
	substrings = filter(lambda cur_string: (string in cur_string) and string != cur_string, string_set)
	is_substring = len(substrings) > 0
	return is_substring

def clean_name_dict(name_dict, query):
	""" Only returns values from name_dict whose keys are a substring of query 
		name_dict: maps names to ids, keys
	"""
	correct_names = dict()

	lowercase_query = query.lower()
	for k, v in name_dict.iteritems():
		if k.lower() in lowercase_query:
			correct_names[k] = v

	return correct_names