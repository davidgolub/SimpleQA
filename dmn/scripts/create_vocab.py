"""
Preprocessing script for MT data.

"""
import re
import json
import os
import glob
import time

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def build_vocab(dataset_paths, dst_path, word_count_threshold = 5):
   
  # count up all word counts so that we can threshold
  # this shouldnt be too expensive of an operation
  print ('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, ))
  t0 = time.time()
  word_counts = {}
  nsents = 0

  for dataset_path in dataset_paths:
    dataset = open(dataset_path, 'r')
    line = 'asdf'
    while True:
      line = dataset.readline()
      if line == '': break
      # Remove newline characters
      trimmedLine = line.replace('\n', '').replace('\r', '').replace('\r\n', '').replace('(', '( ')
      # add space between question marks and periods
      paddedLine = trimmedLine.replace('?', ' ?').replace('.', ' .').replace(')', ' )').replace('-', ' - ')
      paddedLine = paddedLine.replace(',', ' , ').replace('"', ' " ')
      
      tokens = paddedLine.split(' ')
      for w in tokens:
        word_counts[w] = word_counts.get(w, 0) + 1

  vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
  print ('filtered words from %d to %d in %.2fs' % (len(word_counts), len(vocab), time.time() - t0))

  # with K distinct words:
  # - there are K+1 possible inputs (START token and all the words)
  # - there are K+1 possible outputs (END token and all the words)
  # we use ixtoword to take predicted indeces and map them to words for output visualization
  # we use wordtoix to take raw words and get their index in word vector matrix
  ixtoword = {}
  ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
  wordtoix = {}
  wordtoix['#START#'] = 0 # make first vector be the start token
  ix = 1
  for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

  with open(dst_path, 'w') as f:
    for i in ixtoword:
      w = ixtoword[i]
      f.write(w + '\n')
      
  print('saved vocabulary to %s' % dst_path)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing QA dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    QA_dir = os.path.join(data_dir, 'Translation/train')

    input_paths = [os.path.join(QA_dir, 'inputs.txt')]
    question_paths = [os.path.join(QA_dir, 'questions.txt')]
    output_paths = [os.path.join(QA_dir, 'outputs.txt')]

    input_save_path = os.path.join(QA_dir, 'input_vocab.txt')
    question_save_path = os.path.join(QA_dir, 'question_vocab.txt')
    output_save_path = os.path.join(QA_dir, 'output_vocab.txt')

    # get vocabulary
    token_paths = [input_paths, question_paths, output_paths]
    save_paths = [input_save_path, question_save_path, output_save_path]

    for i in range(0, len(token_paths)):   
      build_vocab(
          token_paths[i],
          save_paths[i],
          1
          )
