"""
Implementation from Texygen: https://github.com/geek-ai/Texygen
"""

import os
from multiprocessing import Pool

import nltk
import sys
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

from metrics.Metrics import Metrics


class SelfBleu(Metrics):
    def __init__(self, test_text='', gram=3):
        super().__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 500
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.test_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt



# Function to calculate Self-BLEU for a DataFrame using nltk's sentence_bleu
def calculate_self_bleu_for_df_nltk(df, text_column):
    # Tokenize sentences in the dataset
    df['tokenized_text'] = df[text_column].apply(word_tokenize)

    # Function to calculate BLEU for one sentence against all others
    def calculate_self_bleu(tokenized_sentences, index):
        candidate = tokenized_sentences[index]
        references = [s for i, s in enumerate(tokenized_sentences) if i != index]
        score = sentence_bleu(references, candidate, weights=(1/3, 1/3, 1/3), smoothing_function=SmoothingFunction().method1)
        return score

    # Calculate Self-BLEU for each sentence
    self_bleu_scores = [calculate_self_bleu(df['tokenized_text'].tolist(), i) for i in range(len(df))]

    # Compute the average Self-BLEU score
    return sum(self_bleu_scores) / len(self_bleu_scores)



# Function to calculate Self-BLEU for a DataFrame using the SelfBleu class implemented by Texygen
def calculate_self_bleu_for_df_texygen(df, text_column, gram=3, is_fast=True):
    # Tokenize sentences in the dataset
    df['tokenized_text'] = df[text_column].apply(word_tokenize)

    # Prepare the text file from the DataFrame for SelfBleu
    temp_filename = 'temp_bleu.txt'
    with open(temp_filename, 'w') as f:
        for text in df['tokenized_text']:
            f.write(' '.join(text) + '\n')

    # Create an instance of the SelfBleu class
    test_bleu = SelfBleu(test_text=temp_filename, gram=gram)
    
    # Calculate the BLEU score using the custom class
    average_self_bleu = test_bleu.get_score(is_fast=is_fast)

    # Clean up temporary file
    os.remove(temp_filename)

    return average_self_bleu





# import pandas as pd
# from nltk.tokenize import word_tokenize

# # Assuming SelfBleu class is defined as provided
# from metrics.SelfBleu import SelfBleu 

# def calculate_self_bleu_for_df(df, text_column, gram=3, is_fast=True):
#     # Tokenize sentences in the dataset
#     df['tokenized_text'] = df[text_column].apply(word_tokenize)

#     # Prepare the text file from the DataFrame for SelfBleu
#     temp_filename = 'temp_bleu.txt'
#     with open(temp_filename, 'w') as f:
#         for text in df['tokenized_text']:
#             f.write(' '.join(text) + '\n')

#     # Create an instance of the SelfBleu class
#     test_bleu = SelfBleu(test_text=temp_filename, gram=gram)
    
#     # Calculate the BLEU score using the custom class
#     average_self_bleu = test_bleu.get_score(is_fast=is_fast)

#     # Clean up temporary file
#     os.remove(temp_filename)

#     return average_self_bleu

# # Example usage with a dataset
# generated_df = pd.read_csv('path_to_generated_data.csv')  # Replace with your file path
# average_self_bleu = calculate_self_bleu_for_df(generated_df, 'Text')
# print("Average Self-BLEU Score for Generated Data:", average_self_bleu)







# from google.colab import drive
# import sys
# import nltk
# import pandas as pd
# from nltk.tokenize import word_tokenize
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# # Mount Google Drive and set up the path
# drive.mount('/content/drive')
# sys.path.append('/content/drive/MyDrive/FM_Final_Proj_Code_Repo')
# repopath = '/content/drive/MyDrive/FM_Final_Proj_Code_Repo/'

# # Download necessary NLTK resources
# nltk.download('punkt')

# # Import custom SelfBleu metric
# from metrics.SelfBleu import SelfBleu

# # Function to calculate Self-BLEU for a DataFrame
# def calculate_self_bleu_for_df(df, text_column):
#     # Tokenize sentences in the dataset
#     df['tokenized_text'] = df[text_column].apply(word_tokenize)

#     # Function to calculate BLEU for one sentence against all others
#     def calculate_self_bleu(tokenized_sentences, index):
#         candidate = tokenized_sentences[index]
#         references = [s for i, s in enumerate(tokenized_sentences) if i != index]
#         score = sentence_bleu(references, candidate, weights=(1/3, 1/3, 1/3), smoothing_function=SmoothingFunction().method1)
#         return score

#     # Calculate Self-BLEU for each sentence
#     self_bleu_scores = [calculate_self_bleu(df['tokenized_text'].tolist(), i) for i in range(len(df))]

#     # Compute the average Self-BLEU score
#     return sum(self_bleu_scores) / len(self_bleu_scores)

# # Example usage with two datasets
# test = SelfBleu(repopath + 'temp/test_coco.txt')
# print("Custom Self-BLEU Score:", test.get_score())

# generated_df = pd.read_csv('path_to_generated_data.csv')  # Replace with your file path
# average_self_bleu = calculate_self_bleu_for_df(generated_df, 'Text')
# print("Average Self-BLEU Score for Generated Data:", average_self_bleu)

# oracle_train_data = pd.read_csv('path_to_oracle_train_data.csv')  # Replace with your file path
# average_self_bleu = calculate_self_bleu_for_df(oracle_train_data, 'sentence')
# print("Average Self-BLEU Score for Oracle Train Data:", average_self_bleu)
