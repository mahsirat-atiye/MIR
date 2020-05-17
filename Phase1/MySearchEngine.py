from hazm import Normalizer, word_tokenize, Stemmer, WordTokenizer, stopwords_list
import re

# text preparation
from Phase1 import wiki_dump_parser_

stemmer = Stemmer()
normalizer = Normalizer()
tokenizer = WordTokenizer(separate_emoji=True, replace_links=True, replace_IDs=True, replace_emails=True,
                          replace_hashtags=True, replace_numbers=True)
tokenizer.number_int_repl = '.'
tokenizer.number_float_repl = '.'
tokenizer.email_repl = '.'
tokenizer.hashtag_repl = '.'
tokenizer.id_repl = '.'
tokenizer.emoji_repl = '.'
tokenizer.link_repl = '.'
punctuations = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`؟،{|}~"""


# 10 points
def prepare_text(text):
    text = text.lower()
    text = re.sub('\d+', '', text)
    text = text.translate(str.maketrans(punctuations, ' ' * len(punctuations)))
    text = ' '.join(re.sub(r'[^ضصثقفغعهخحجچشسیبلاتنمکگظطزرذدپوئژآؤ \n]', ' ', text).split())
    text = text.strip()
    normalized_text = normalizer.normalize(text)
    words = word_tokenize(normalized_text)
    words = [w for w in words if w != '.']
    words = [w for w in words if w not in stopwords_list()]
    words = [stemmer.stem(w) for w in words]
    return words


print('Enter text:')
raw_text = input()
print(prepare_text(raw_text))

# -------------------------------------------------------------------------------------------------
# positional index construction
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

persian_wiki_xml_data_dir = "Persian.xml"
wiki_dump_parser_.xml_to_csv(persian_wiki_xml_data_dir, ',')
PERSIAN_DIR = "Persian.csv"

postings = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

ids = list()
is_valid = list()
num_of_documents = 0
documents_words = []
titles_words = []
term_frequency_in_doc = defaultdict(lambda: 1)
term_frequency_in_title = defaultdict(lambda: 1)

TEXT = 'text'
TITLE = 'title'


def construct_positional_indexes(docs_path):
    global postings, ids, is_valid, num_of_documents, documents_words, titles_words, term_frequency_in_doc
    dataframe = pd.read_csv(docs_path)
    titles = dataframe['page_title']
    documents = dataframe['text']
    ids = dataframe['page_id'].tolist()
    num_of_documents = len(documents)
    is_valid = [True] * num_of_documents

    for title, document in zip(titles, tqdm(documents)):
        document_words = prepare_text(document)
        title_words = prepare_text(title)

        num_of_document_words = len(document_words)
        if num_of_document_words == 0:
            continue
        f = 1 / num_of_document_words
        for word in document_words:
            term_frequency_in_doc[word] += f

        num_of_title_words = len(title_words)
        if num_of_title_words == 0:
            continue
        f = 1 / num_of_title_words
        for word in title_words:
            term_frequency_in_title[word] += f

        #     check in next steps
        # todo
        documents_words.append(document_words)
        titles_words.append(title_words)

    for i, document_words_title_words in enumerate(list(zip(tqdm(documents_words), titles_words))):
        # doc_id = int(ids[i])
        doc_id = i
        for j, document_word in enumerate(document_words_title_words[0]):
            postings[document_word][doc_id][TEXT].append(j)

        for k, title_word in enumerate(document_words_title_words[1]):
            postings[title_word][doc_id][TITLE].append(k)


construct_positional_indexes(PERSIAN_DIR)

# -------------------------------------------------------------------------------------------------------
import json
import pprint


def get_posting_list(word):
    global postings
    prepared_word = prepare_text(word)
    if len(prepared_word) < 1:
        print("bad input")
        return
    word = prepared_word[0]
    if not word in postings.keys():
        print("word: ", word, " is not in DB!")
        return
    posting_list = postings[word]
    pprint.pprint(json.loads(json.dumps(posting_list)))
    return posting_list


get_posting_list('سلام')
get_posting_list('اسلام')
get_posting_list('سلامت')


# ----------------------------------------------------------------------------------------------------------
def add_document_to_indexes(docs_path, doc_num):
    wiki_dump_parser_.xml_to_csv(docs_path)
    dataframe = pd.read_csv(docs_path)
    title_text = dataframe['page_title'][0]
    doc_text = dataframe['text'][0]

    global ids, num_of_documents, documents_words, titles_words, is_valid
    if doc_num in ids and is_valid[ids.index(doc_num)]:
        print("already exist")
        return

    num_of_documents += 1
    doc_words = prepare_text(doc_text)
    documents_words.append(doc_words)
    title_words = prepare_text(title_text)
    titles_words.append(title_words)

    for j, document_word in enumerate(doc_words):
        postings[document_word][doc_num][TEXT].append(j)

    for k, title_word in enumerate(title_words):
        postings[title_word][doc_num][TITLE].append(k)

    ids.append(doc_num)
    is_valid += [True]
    print("added")


# add_document_to_indexes('new_wiki.xml', 3016)


# -------------------------------------------------------------------------------------------------------
def delete_document_from_indexes(docs_path, doc_num):
    global is_valid
    if doc_num in ids:
        is_valid[ids.index(doc_num)] = False


delete_document_from_indexes('data/wiki', 10)

# ---------------------------------------------------------------------------------------------------------
import pickle
import os
import dill


def save_index(destination):
    try:

        os.mkdir(destination.split("/")[0])
    except FileExistsError:
        pass
    global postings
    with open(destination, "wb") as f:
        print("Saving postings into a file")
        dill.dump(dict(postings), f)


save_index('storage/index_backup')


# ---------------------------------------------------------------------------------------------------------
def load_index(source):
    global postings
    with open(source, "rb") as f:
        print("Loading postings from file")
        postings = dill.load(f)


load_index('storage/index_backup')
get_posting_list('اسلام')
# --------------------------------------------------------------------------------------------------------
import numpy as np

all_terms = []


def build_tf():
    global all_terms
    print("Building tf table...")
    all_terms = list(postings.keys())
    word_to_id = {t: i for i, t in enumerate(all_terms)}
    tf = np.zeros(shape=(num_of_documents, len(all_terms), 2), dtype=np.int)
    # all terms
    # d
    # o
    # c
    # u
    # m
    # e
    # n
    # t
    # s
    for word in postings.keys():
        for document_including_word in postings[word].keys():
            tf[document_including_word][word_to_id[word]][0] = len(postings[word][document_including_word][TEXT])
            tf[document_including_word][word_to_id[word]][1] = len(postings[word][document_including_word][TITLE])
    return tf


tf = build_tf()


def my_log(x):
    if x == 0:
        return 0
    return np.log(x)


def combine_text_title_score(texts, titles, w):
    titles = sorted(titles, key=lambda x: x[0])
    texts = sorted(texts, key=lambda x: x[0])
    title_indx = 0
    text_indx = 0
    result = []
    while title_indx < len(titles) and text_indx < len(texts):

        if titles[title_indx][0] == texts[text_indx][0]:
            result.append((titles[title_indx][0], w * titles[title_indx][1] + texts[text_indx][1]))
            text_indx += 1
            title_indx += 1
        elif titles[title_indx][0] < texts[text_indx][0]:
            result.append((titles[title_indx][0], w * titles[title_indx][1]))
            title_indx += 1
        else:
            result.append(texts[text_indx])
            text_indx += 1
    while title_indx < len(titles):
        result.append((titles[title_indx][0], w * titles[title_indx][1]))
        title_indx += 1
    while text_indx < len(texts):
        result.append(texts[text_indx])
        text_indx += 1
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return result


def query_ltc_lnc_text(query, index):
    # index 0: text, index 1: title
    global tf
    query_terms_ = query
    query_terms_ = prepare_text(query_terms_)
    query_terms_ = [q for q in query_terms_ if q in postings]
    score = np.zeros(shape=(num_of_documents,))

    # scoring
    query_sum_square_weights = 0
    query_terms = list(set(query_terms_))
    query_weights = [0] * len(query_terms)
    for i, query_term in enumerate(query_terms):
        # l in lnc
        query_tf = 1 + my_log(query_terms_.count(query_term))
        # n in ltc
        query_idf = 1
        # saving results
        query_weight = query_tf * query_idf
        query_weights[i] = query_weight
        query_sum_square_weights += query_weight ** 2

    doc_ids_including_query_set = []
    for query_term in query_terms:
        doc_ids_including_query_set.extend(list(postings[query_term].keys()))

    doc_ids_including_query_set = set(doc_ids_including_query_set)
    docs_weights = [[0] * len(query_terms) for i in range(len(doc_ids_including_query_set))]
    doc_ids_including_query_list = list(doc_ids_including_query_set)

    doc_id_to_index = {t: i for i, t in enumerate(doc_ids_including_query_list)}
    query_term_to_index = {t: i for i, t in enumerate(query_terms)}
    for i, query_term in enumerate(query_terms):
        for doc_id in postings[query_term]:
            # l in ltc
            doc_tf = 1 + my_log(tf[doc_id][all_terms.index(query_term)][index])
            # t in ltc
            doc_idf = my_log(num_of_documents / len(postings[query_term]))
            doc_weight = doc_tf * doc_idf
            docs_weights[doc_id_to_index[doc_id]][query_term_to_index[query_term]] = doc_weight

    doc_sum_square_weights = [0] * len(doc_ids_including_query_list)
    for i, doc_id in enumerate(doc_ids_including_query_list):
        for j, query_term in enumerate(query_terms):
            doc_sum_square_weights[i] += docs_weights[i][j] ** 2

    final_score = [[0] * len(query_terms) for i in range(len(doc_ids_including_query_list))]

    for i, doc_id in enumerate(doc_ids_including_query_list):
        for j, query_term in enumerate(query_terms):
            final_score[i][j] = docs_weights[i][j] * query_weights[j] * (
                    doc_sum_square_weights[i] * query_sum_square_weights) ** (-0.5)

    doc_scores = [0] * len(doc_ids_including_query_list)
    for i, doc_id in enumerate(doc_ids_including_query_list):
        for j, query_term in enumerate(query_terms):
            doc_scores[i] += final_score[i][j]
    s = []
    for i, x in enumerate(doc_scores):
        if is_valid[doc_ids_including_query_list[i]]:
            s.append((doc_ids_including_query_list[i], x))
    # s = sorted(s, key=lambda x: x[1], reverse=True)
    return s


def query_ltn_lnn_text(query, index):
    # index 0: text, index 1: title
    global tf
    query_terms_ = query
    query_terms_ = prepare_text(query_terms_)
    query_terms_ = [q for q in query_terms_ if q in postings]

    # scoring
    query_terms = list(set(query_terms_))
    query_weights = [0] * len(query_terms)
    for i, query_term in enumerate(query_terms):
        # l in lnc
        query_tf = 1 + my_log(query_terms_.count(query_term))
        # n in ltc
        query_idf = 1
        # saving results
        query_weight = query_tf * query_idf
        query_weights[i] = query_weight

    doc_ids_including_query_set = []
    for query_term in query_terms:
        doc_ids_including_query_set.extend(list(postings[query_term].keys()))

    doc_ids_including_query_set = set(doc_ids_including_query_set)
    docs_weights = [[0] * len(query_terms) for i in range(len(doc_ids_including_query_set))]
    doc_ids_including_query_list = list(doc_ids_including_query_set)

    doc_id_to_index = {t: i for i, t in enumerate(doc_ids_including_query_list)}
    query_term_to_index = {t: i for i, t in enumerate(query_terms)}
    for i, query_term in enumerate(query_terms):
        for doc_id in postings[query_term]:
            # l in ltc
            doc_tf = 1 + my_log(tf[doc_id][all_terms.index(query_term)][index])
            # t in ltc
            doc_idf = my_log(num_of_documents / len(postings[query_term]))
            doc_weight = doc_tf * doc_idf
            docs_weights[doc_id_to_index[doc_id]][query_term_to_index[query_term]] = doc_weight

    final_score = [[0] * len(query_terms) for i in range(len(doc_ids_including_query_list))]

    for i, doc_id in enumerate(doc_ids_including_query_list):
        for j, query_term in enumerate(query_terms):
            final_score[i][j] = docs_weights[i][j] * query_weights[j]

    doc_scores = [0] * len(doc_ids_including_query_list)
    for i, doc_id in enumerate(doc_ids_including_query_list):
        for j, query_term in enumerate(query_terms):
            doc_scores[i] += final_score[i][j]
    s = []
    for i, x in enumerate(doc_scores):
        if is_valid[doc_ids_including_query_list[i]]:
            s.append((doc_ids_including_query_list[i], x))
    return s


def search(query, method="ltn-lnn", weight=2):
    if method == "ltc-lnc":
        relevant_docs_text = query_ltc_lnc_text(query, 0)
        relevant_docs_title = query_ltc_lnc_text(query, 1)
        relevant_docs = combine_text_title_score(relevant_docs_text, relevant_docs_title, weight)
        real_doc_ids = [ids[x[0]] for x in relevant_docs]
        return real_doc_ids
    else:
        # method="ltn-lnn"
        relevant_docs_text = query_ltn_lnn_text(query, 0)
        relevant_docs_title = query_ltn_lnn_text(query, 1)
        relevant_docs = combine_text_title_score(relevant_docs_text, relevant_docs_title, weight)
        real_doc_ids = [ids[x[0]] for x in relevant_docs]
        return real_doc_ids


print(search('"نظرخواهی انجام شده توسط دانشگاه "شهر نیویورک', "ltc-lnc", 3))

# -----------------------------------------------------------------------------------------


