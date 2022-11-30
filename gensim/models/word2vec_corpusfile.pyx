#!/usr/bin/env cython
# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8
#
# Copyright (C) 2018 Dmitry Persiyanov <dmitry.persiyanov@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Optimized cython functions for file-based training :class:`~gensim.models.word2vec.Word2Vec` model."""

import cython
from os import getpid
import numpy as np

from gensim.utils import any2utf8

cimport numpy as np

from libc.stdlib cimport free, malloc
from libc.stdio cimport fprintf, printf, FILE, fopen, fclose
from libc.math cimport sqrt
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as bool_t

#include <stdio.h>
#include <unistd.h>

from gensim.models.word2vec_inner cimport (
    w2v_fast_sentence_sg_hs,
    w2v_fast_sentence_sg_neg,
    w2v_fast_sentence_cbow_hs,
    w2v_fast_sentence_cbow_neg,
    random_int32,
    init_w2v_config,
    Word2VecConfig,
    our_dot
)


cdef double cy_cosine(REAL_t *x, const int xi, REAL_t *y, const int yi, const int size) nogil:
    cdef double xx = 0.0
    cdef double yy = 0.0
    cdef double xy = 0.0
    cdef int i
    for i in range(size):
        xx+=x[xi + i] * x[xi + i]
        yy+=y[yi + i] * y[yi + i]
        xy+=x[xi + i] * y[yi + i]
    return xy / sqrt(xx * yy)


cdef double our_cos(int *N, REAL_t *X, int *incX, REAL_t *Y, int *incY) nogil:
    cdef double xx, yy, xy
    xx = our_dot(N, X, incX, X, incX)
    yy = our_dot(N, Y, incY, Y, incY)
    xy = our_dot(N, X, incX, Y, incY)
    return xy / sqrt(xx * yy)


cdef void subtract(REAL_t *x, REAL_t *y, REAL_t *z, int size) nogil:
    for i in range(size):
        z[i] = y[i] - x[i]


DEF MAX_SENTENCE_LEN = 10000


@cython.final
cdef class CythonVocab:
    def __init__(self, wv, hs=0, fasttext=0):
        cdef VocabItem word

        vocab_sample_ints = wv.expandos['sample_int']
        if hs:
            vocab_codes = wv.expandos['code']
            vocab_points = wv.expandos['point']
        for py_token in wv.key_to_index.keys():
            token = any2utf8(py_token)
            word.index = wv.get_index(py_token)
            word.sample_int = vocab_sample_ints[word.index]

            if hs:
                word.code = <np.uint8_t *>np.PyArray_DATA(vocab_codes[word.index])
                word.code_len = <int>len(vocab_codes[word.index])
                word.point = <np.uint32_t *>np.PyArray_DATA(vocab_points[word.index])

            # subwords information, used only in FastText model
            if fasttext:
                word.subword_idx_len = <int>(len(wv.buckets_word[word.index]))
                word.subword_idx = <np.uint32_t *>np.PyArray_DATA(wv.buckets_word[word.index])

            self.vocab[token] = word

    cdef cvocab_t* get_vocab_ptr(self) nogil except *:
        return &self.vocab


def rebuild_cython_line_sentence(source, max_sentence_length):
    return CythonLineSentence(source, max_sentence_length=max_sentence_length)


cdef bytes to_bytes(key):
    if isinstance(key, bytes):
        return <bytes>key
    else:
        return key.encode('utf8')


@cython.final
cdef class CythonLineSentence:
    def __cinit__(self, source, offset=0, max_sentence_length=MAX_SENTENCE_LEN):
        self._thisptr = new FastLineSentence(to_bytes(source), offset)

    def __init__(self, source, offset=0, max_sentence_length=MAX_SENTENCE_LEN):
        self.source = to_bytes(source)
        self.offset = offset
        self.max_sentence_length = max_sentence_length
        self.max_words_in_batch = max_sentence_length

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    cpdef bool_t is_eof(self) nogil:
        return self._thisptr.IsEof()

    cpdef vector[string] read_sentence(self) nogil except *:
        return self._thisptr.ReadSentence()

    cpdef vector[vector[string]] _read_chunked_sentence(self) nogil except *:
        cdef vector[string] sent = self.read_sentence()
        return self._chunk_sentence(sent)

    cpdef vector[vector[string]] _chunk_sentence(self, vector[string] sent) nogil:
        cdef vector[vector[string]] res
        cdef vector[string] chunk
        cdef size_t cur_idx = 0

        if sent.size() > self.max_sentence_length:
            while cur_idx < sent.size():
                chunk.clear()
                for i in range(cur_idx, min(cur_idx + self.max_sentence_length, sent.size())):
                    chunk.push_back(sent[i])

                res.push_back(chunk)
                cur_idx += chunk.size()
        else:
            res.push_back(sent)

        return res

    cpdef void reset(self) nogil:
        self._thisptr.Reset()

    def __iter__(self):
        self.reset()
        while not self.is_eof():
            chunked_sentence = self._read_chunked_sentence()
            for chunk in chunked_sentence:
                if not chunk.empty():
                    yield chunk

    def __reduce__(self):
        # This function helps pickle to correctly serialize objects of this class.
        return rebuild_cython_line_sentence, (self.source, self.max_sentence_length)

    cpdef vector[vector[string]] next_batch(self) nogil except *:
        cdef:
            vector[vector[string]] job_batch
            vector[vector[string]] chunked_sentence
            vector[string] data
            size_t batch_size = 0
            size_t last_idx = 0
            size_t tmp = 0
            int idx

        # Try to read data from previous calls which was not returned
        if not self.buf_data.empty():
            job_batch = self.buf_data
            self.buf_data.clear()

            for sent in job_batch:
                batch_size += sent.size()

        while not self.is_eof() and batch_size <= self.max_words_in_batch:
            data = self.read_sentence()

            chunked_sentence = self._chunk_sentence(data)
            for chunk in chunked_sentence:
                job_batch.push_back(chunk)
                batch_size += chunk.size()

        if batch_size > self.max_words_in_batch:
            # Save data which doesn't fit in batch in order to return it later.
            self.buf_data.clear()

            tmp = batch_size
            idx = job_batch.size() - 1
            while idx >= 0:
                if tmp - job_batch[idx].size() <= self.max_words_in_batch:
                    last_idx = idx + 1
                    break
                else:
                    tmp -= job_batch[idx].size()

                idx -= 1

            for i in range(last_idx, job_batch.size()):
                self.buf_data.push_back(job_batch[i])
            job_batch.resize(last_idx)

        return job_batch


cdef void prepare_c_structures_for_batch(
        vector[vector[string]] &sentences, int sample, int hs, int window, long long *total_words,
        int *effective_words, int *effective_sentences, unsigned long long *next_random,
        cvocab_t *vocab, int *sentence_idx, np.uint32_t *indexes, int *codelens,
        np.uint8_t **codes, np.uint32_t **points, np.uint32_t *reduced_windows) nogil:
    cdef VocabItem word
    cdef string token
    cdef vector[string] sent

    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if sent.empty():
            continue # ignore empty sentences; leave effective_sentences unchanged
        total_words[0] += sent.size()

        for token in sent:
            # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if vocab[0].find(token) == vocab[0].end():
                continue

            word = vocab[0][token]
            if sample and word.sample_int < random_int32(next_random):
                continue
            indexes[effective_words[0]] = word.index
            if hs:
                codelens[effective_words[0]] = word.code_len
                codes[effective_words[0]] = word.code
                points[effective_words[0]] = word.point
            effective_words[0] += 1
            if effective_words[0] == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences[0] += 1
        sentence_idx[effective_sentences[0]] = effective_words[0]

        if effective_words[0] == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    for i in range(effective_words[0]):
        reduced_windows[i] = random_int32(next_random) % window


cdef REAL_t get_alpha(REAL_t alpha, REAL_t end_alpha, int cur_epoch, int num_epochs) nogil:
    return alpha - ((alpha - end_alpha) * (<REAL_t> cur_epoch) / num_epochs)


cdef REAL_t get_next_alpha(
        REAL_t start_alpha, REAL_t end_alpha, long long total_examples, long long total_words,
        long long expected_examples, long long expected_words, int cur_epoch, int num_epochs) nogil:
    cdef REAL_t epoch_progress

    if expected_examples != -1:
        # examples-based decay
        epoch_progress = (<REAL_t> total_examples) / expected_examples
    else:
        # word-based decay
        epoch_progress = (<REAL_t> total_words) / expected_words

    cdef REAL_t progress = (cur_epoch + epoch_progress) / num_epochs
    cdef REAL_t next_alpha = start_alpha - (start_alpha - end_alpha) * progress
    return max(end_alpha, next_alpha)


def train_epoch_sg(model, corpus_file, offset, _cython_vocab, _cur_epoch, _expected_examples, _expected_words, _work,
                   _neu1, compute_loss):
    """Train Skipgram model for one epoch by training on an input stream. This function is used only in multistream mode.

    Called internally from :meth:`~gensim.models.word2vec.Word2Vec.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.word2vec.Word2Vec`
        The Word2Vec model instance to train.
    corpus_file : str
        Path to corpus file.
    _cur_epoch : int
        Current epoch number. Used for calculating and decaying learning rate.
    _work : np.ndarray
        Private working memory for each worker.
    _neu1 : np.ndarray
        Private working memory for each worker.
    compute_loss : bool
        Whether or not the training loss should be computed in this batch.

    Returns
    -------
    int
        Number of words in the vocabulary actually used for training (They already existed in the vocabulary
        and were not discarded by negative sampling).
    """
    cdef Word2VecConfig c

    # For learning rate updates
    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef long long expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef long long expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)
    cdef CythonVocab vocab = _cython_vocab

    cdef int i, j, k, z
    cdef int effective_words = 0, effective_sentences = 0
    cdef long long total_sentences = 0
    cdef long long total_effective_words = 0, total_words = 0
    cdef long long iter_idx = 0
    cdef long long sent_idx = 0
    cdef int idx_start, idx_end
    cdef int one = 1

    #print('BEGIN TRAINING LOOP')

    cdef REAL_t *diff
    cdef np.uint32_t *targets
    cdef int n_targets, target_int
    #print('\n\n\n')
    #print(model.wv.vector_size)
    #print('HERE!!!\n\n\n')
    #diff = <REAL_t *>(np.PyArray_DATA(np.zeros(model.wv.vector_size)))
    diff = <REAL_t *>malloc(model.wv.vector_size * sizeof(REAL_t))
    targets = <np.uint32_t *>(np.PyArray_DATA(model.targets))
    n_targets = int(len(model.targets)) - 2  # because the first two targets are the dimension anchors (TODO: fix this later?)
    target_int = model.target_int

    init_w2v_config(&c, model, _alpha, compute_loss, _work)

    cdef vector[vector[string]] sentences

    # probe output file
    cdef FILE *ptr_fout
    ptr_fout = fopen(model.probe_fname, 'ab')

    with nogil:
        input_stream.reset()
        while not (input_stream.is_eof() or total_words > expected_words / c.workers):
            effective_sentences = 0
            effective_words = 0

            sentences = input_stream.next_batch()

            prepare_c_structures_for_batch(
                sentences, c.sample, c.hs, c.window, &total_words, &effective_words, &effective_sentences,
                &c.next_random, vocab.get_vocab_ptr(), c.sentence_idx, c.indexes,
                c.codelens, c.codes, c.points, c.reduced_windows)

            #printf('%d + %d\n', effective_sentences, total_sentences)

            for sent_idx in range(effective_sentences):
                idx_start = c.sentence_idx[sent_idx]
                idx_end = c.sentence_idx[sent_idx + 1]
                for i in range(idx_start, idx_end):
                    j = i - c.window + c.reduced_windows[i]
                    if j < idx_start:
                        j = idx_start
                    k = i + c.window + 1 - c.reduced_windows[i]
                    if k > idx_end:
                        k = idx_end
                    for j in range(j, k):
                        if j == i:
                            continue
                        if c.hs:
                            w2v_fast_sentence_sg_hs(
                                c.points[i], c.codes[i], c.codelens[i], c.syn0, c.syn1, c.size, c.indexes[j],
                                c.alpha, c.work, c.words_lockf, c.words_lockf_len, c.compute_loss,
                                &c.running_training_loss)
                        if c.negative:
                            c.next_random = w2v_fast_sentence_sg_neg(
                                c.negative, c.cum_table, c.cum_table_len, c.syn0, c.syn1neg, c.size,
                                c.indexes[i], c.indexes[j], c.alpha, c.work, c.next_random,
                                c.words_lockf, c.words_lockf_len,
                                c.compute_loss, &c.running_training_loss)
                iter_idx = (cur_epoch * expected_examples) + sent_idx + total_sentences
                #printf('%s\n', 'MID TRAINING LOOP')
                if iter_idx % target_int == 0:
                    #printf('train another skipgram sentence\n')
                    #printf('%d-%d: %f\n', c.indexes[i], c.indexes[j], cy_cosine(c.syn0, c.indexes[i], c.syn0, c.indexes[j], c.size))
                    #printf('pairwise cosines:\n')
                    #printf('%s\n', 'MID TRAINING LOOP PT 2')
                    fprintf(ptr_fout, '%d', cur_epoch)
                    #printf('%d * %d = %d\n', cur_epoch, expected_examples, cur_epoch * expected_examples)
                    #printf('%d\n', sent_idx)
                    #printf('%d\n', total_sentences)
                    fprintf(ptr_fout, '\t%d', sent_idx + total_sentences)
                    fprintf(ptr_fout, '\t%d', iter_idx)
                    fprintf(ptr_fout, '\t%f', c.alpha)
                    
                    subtract(&c.syn0[targets[0] * c.size], &c.syn0[targets[1] * c.size], &diff[0], c.size)
                    #printf('\nembs: %f', c.syn0[targets[0] * c.size])
                    #printf('\nembs: %f', c.syn0[targets[1] * c.size])
                    #printf('\ndiff: %f', diff[0])

                    #subtract(&c.syn0[targets[0] * c.size], &c.syn0[targets[1] * c.size], &diff[0], c.size)
                    
                    #printf('%f\n', diff[0])
                    #printf('%f\n', diff[1])
                    #printf('%f\n', diff[2])
                    #printf('%f\n', diff[3])
                    for z in range(n_targets):
                        #fprintf(ptr_fout, '\t%f', our_cos(&c.size, &c.syn0[targets[2 * z] * c.size], &one, &c.syn0[targets[(2 * z) + 1] * c.size], &one))
                        #fprintf(ptr_fout, '\t%f', our_cos(&c.size, &c.syn0[targets[0] * c.size], &one, &c.syn0[targets[z + 2] * c.size], &one))

                        fprintf(ptr_fout, '\t%f', our_cos(&c.size, &diff[0], &one, &c.syn0[targets[z + 2] * c.size], &one))
                        pass

                    fprintf(ptr_fout, '\n')

            total_sentences += sentences.size()
            total_effective_words += effective_words

            #printf('diff %p\n', &diff)
            #printf('ptr_fout %p\n', &ptr_fout)
            #printf('iter_idx %p\n', &iter_idx)
            #printf('alpha %p\n', &c.alpha)

            #printf('%s\n', 'END TRAINING LOOP PT 1')

            c.alpha = get_next_alpha(
                start_alpha, end_alpha, total_sentences, total_words,
                expected_examples, expected_words, cur_epoch, num_epochs)

        #printf('%s\n', 'END TRAINING LOOP PT 2')
    #print('AFTER TRAINING LOOP')
    model.running_training_loss = c.running_training_loss
    fclose(ptr_fout)
    return total_sentences, total_effective_words, total_words


def train_epoch_cbow(model, corpus_file, offset, _cython_vocab, _cur_epoch, _expected_examples, _expected_words, _work,
                     _neu1, compute_loss):
    """Train CBOW model for one epoch by training on an input stream. This function is used only in multistream mode.

    Called internally from :meth:`~gensim.models.word2vec.Word2Vec.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.word2vec.Word2Vec`
        The Word2Vec model instance to train.
    corpus_file : str
        Path to corpus file.
    _cur_epoch : int
        Current epoch number. Used for calculating and decaying learning rate.
    _work : np.ndarray
        Private working memory for each worker.
    _neu1 : np.ndarray
        Private working memory for each worker.
    compute_loss : bool
        Whether or not the training loss should be computed in this batch.

    Returns
    -------
    int
        Number of words in the vocabulary actually used for training (They already existed in the vocabulary
        and were not discarded by negative sampling).
    """
    cdef Word2VecConfig c

    # For learning rate updates
    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef long long expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef long long expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)
    cdef CythonVocab vocab = _cython_vocab

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef long long total_sentences = 0
    cdef long long total_effective_words = 0, total_words = 0
    cdef int sent_idx, idx_start, idx_end

    init_w2v_config(&c, model, _alpha, compute_loss, _work, _neu1)

    cdef vector[vector[string]] sentences

    with nogil:
        input_stream.reset()
        while not (input_stream.is_eof() or total_words > expected_words / c.workers):
            effective_sentences = 0
            effective_words = 0

            sentences = input_stream.next_batch()

            prepare_c_structures_for_batch(
                sentences, c.sample, c.hs, c.window, &total_words, &effective_words,
                &effective_sentences, &c.next_random, vocab.get_vocab_ptr(), c.sentence_idx,
                c.indexes, c.codelens, c.codes, c.points, c.reduced_windows)

            for sent_idx in range(effective_sentences):
                idx_start = c.sentence_idx[sent_idx]
                idx_end = c.sentence_idx[sent_idx + 1]
                for i in range(idx_start, idx_end):
                    j = i - c.window + c.reduced_windows[i]
                    if j < idx_start:
                        j = idx_start
                    k = i + c.window + 1 - c.reduced_windows[i]
                    if k > idx_end:
                        k = idx_end
                    if c.hs:
                        w2v_fast_sentence_cbow_hs(
                            c.points[i], c.codes[i], c.codelens, c.neu1, c.syn0, c.syn1, c.size, c.indexes, c.alpha,
                            c.work, i, j, k, c.cbow_mean, c.words_lockf, c.words_lockf_len, c.compute_loss,
                            &c.running_training_loss)

                    if c.negative:
                        c.next_random = w2v_fast_sentence_cbow_neg(
                            c.negative, c.cum_table, c.cum_table_len, c.codelens, c.neu1, c.syn0,
                            c.syn1neg, c.size, c.indexes, c.alpha, c.work, i, j, k, c.cbow_mean,
                            c.next_random, c.words_lockf, c.words_lockf_len, c.compute_loss,
                            &c.running_training_loss)

            total_sentences += sentences.size()
            total_effective_words += effective_words

            c.alpha = get_next_alpha(
                start_alpha, end_alpha, total_sentences, total_words,
                expected_examples, expected_words, cur_epoch, num_epochs)

    model.running_training_loss = c.running_training_loss
    return total_sentences, total_effective_words, total_words


CORPUSFILE_VERSION = 1
