#https://stackoverflow.com/questions/40966014/how-to-use-gensim-bm25-ranking-in-python
#https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/summarization/bm25.py
import os
import csv
import argparse
import re
import json
import random


BOS = EOS = '<|endoftext|>'
BOF = '<bof>'
BOR = '<sep>'

def read_file_lines(path, filename):
	file_add = os.path.join(path, filename)
	with open(file_add) as f:
		content = f.readlines()
	content = [x.strip() for x in content] 
	
	return content

def read_csv_file_inp_ret(filename):
	inp_data = []
	ret_data = []
	with open(filename,encoding="utf-8") as f:
		for cnt, line in enumerate(f):
		# reader = csv.reader(f,delimiter = "_eos")
			sentences = line.split('\t')
			if sentences[-1] == '\n':
				sentences = sentences[:-1]
			sentences = [s.strip() for s in sentences]
			# data.append(sentences)
			inp_data.append([sentences[-2]])
			ret_data.append([sentences[-1]])
	print(len(inp_data), ' lines in ', filename)

	return inp_data, ret_data

hybrid_mapping = {
	'yes': 'yes',
	'no': 'no',
	'not':'no',
	"n't":'no',
	"why":"why",
	"when":"when",
	"who":"who",
	"whom":"who",
	"whose":"who",
	"what": "what",
	"where" : 'where',
	"which": "which",
	'?': "?",
	"how":"how",
	"i" : 'pronoun',
	"you": "pronoun",
	"he" : 'pronoun',
	"she": "pronoun",
	"it" : 'pronoun',
	"they": "pronoun",
}


def clean_frames_old(frame_row_data):
	frame_row_data = frame_row_data.split()   
	set_frames = set(frame_row_data) - set('_')

	return ' '.join(list(set_frames))


def clean_frames(frame_line):
    frame_words = frame_line.split(' ')
    exclude_frames = ['_']
    set_frame_words = []
    for frame in frame_words:
        if frame not in exclude_frames:
            set_frame_words.append(frame)
    line_frame_words = ' '.join(set_frame_words)

    return line_frame_words

def clean_frames_hybrid(frame_line, text):
	frame_words = frame_line.split(' ')
	words = text.split(' ')
	# print(frame_words) 
	# print(words)
	# print(len(frame_words), len(words))
	# import pdb;pdb.set_trace()
	exclude_frames = ['_']
	set_frame_words = []
	for f, frame in enumerate(frame_words):
		if f<len(words):
			word = words[f].lower()
			if word in hybrid_mapping:
				# print(word)
				frame = hybrid_mapping.get(word, word)
				set_frame_words.append(frame)
				continue
		if frame not in exclude_frames:
			set_frame_words.append(frame)
	line_frame_words = ' '.join(set_frame_words)
	# print(line_frame_words)
	return line_frame_words


def get_overlapped_frames(frames1, frames2):
	frame_set1 = set(frames1.split())
	frame_set2 = set(frames2.split())

	return frame_set1.intersection(frame_set2)    


def prepare_lm_data(opt, type_split = 'test', limit=None, num_false_cands=1, final_json = {}):
	is_train = True if type_split=='train' else False
	inp_file = opt.test_inp_file

	csv_file = open(inp_file, encoding="utf-8")
	csv_reader_data = []
	csv_reader = csv.DictReader(csv_file, delimiter=',')
	for i, row in enumerate(csv_reader):
		csv_reader_data.append(row)

	num_data_in_csv = sum(1 for line in csv_reader_data)
	print(num_data_in_csv, ' data points found')

	data_list = []
	for i, row in enumerate(csv_reader_data):
		if limit and i==limit:
			break
		if i%1000==0:
			print("line count now: ", i)
		
		ground_truth = row['response']
		input_context = row['context']
		context_history = input_context.split('_eos')[:-1]
		context_history = [c.strip() for c in context_history]
		retrieved_response = row['retrieved response']
		retrieved_context= row['retrieved context']
		inp_response_row_frames = row['response frames']
		ret_response_row_frames = row['retrieved response frames']
		ret_response_frames = clean_frames_hybrid(ret_response_row_frames.strip(), retrieved_response)
		# ret_response_frames = clean_frames(ret_response_row_frames.strip())
		response_frames = clean_frames_hybrid(inp_response_row_frames.strip(), ground_truth)
		# response_frames = clean_frames(inp_response_row_frames.strip())

		cand_list, cand_frames_list = [], []
		for j in range(num_false_cands):
			rand_ind = random.randint(0, num_data_in_csv-1)
			random_candidate = csv_reader_data[rand_ind]['response'].strip()
			cand_list.append(random_candidate)
			cand_frame = clean_frames_hybrid(csv_reader_data[rand_ind]['response frames'].strip(), random_candidate)
			cand_frames_list.append(cand_frame)

		data_point = {'utterances': [{'candidates': cand_list + [ground_truth], 'history': context_history, 'response': ground_truth, 'response_frames': cand_frames_list + [response_frames]}]}
		if not is_train:
			data_point['retrieved_response'] = retrieved_response
			data_point['retrieved_response_frames'] = ret_response_frames
		data_list.append(data_point)

	final_json[type_split] = data_list
	print('data_count', len(data_list))
	# print(final_json)
	csv_file.close()


	return final_json



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--test_inp_file", help="input file with text and frames", default = "sample_frame_data.csv")
	parser.add_argument("--train_inp_file", help="input file with text and frames")
	parser.add_argument("--valid_inp_file", help="input file with text and frames", default = "")

	parser.add_argument("--out_file", help="output file including retrieved responses and frames", default = "lm_data.json")	
	
	opt = parser.parse_args()

	output_file = open(opt.out_file, 'w') 

	final_json = prepare_lm_data(opt, type_split='test')
	if opt.train_inp_file:
		prepare_lm_data(opt, type_split='train', final_json=final_json)
	if opt.valid_inp_file:
		prepare_lm_data(opt, type_split='valid', final_json=final_json)

	json.dump(final_json, output_file)
	output_file.close()

if __name__ == '__main__':
	main()

# sample invocation python convert_convai_lm_data.py --train_inp_file sample_frame_data.csv --valid_inp_file sample_frame_data.csv
