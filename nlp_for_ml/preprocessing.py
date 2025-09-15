import emoji
import spacy
from gensim import corpora
from scipy.sparse import csr_matrix

from typing import List, Optional, Iterable, Tuple

import nltk
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora, matutils
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

def identity_tokenizer(x):
    return x

class SklearnGensimBowVectorizer(BaseEstimator, TransformerMixin):
    """
    A Gensim-based Bag-of-Words vectorizer fully compatible with scikit-learn pipelines.

    This class adheres to the scikit-learn API by inheriting from BaseEstimator
    and TransformerMixin. It uses gensim's Dictionary for vocabulary creation
    and transforms tokenized documents into a scipy CSR sparse matrix.

    Parameters
    ----------
    min_df : int, default=1
        An alias for no_below. The minimum number of documents a word must
        appear in to be included in the vocabulary.
    
    max_df : float, default=0.95
        An alias for no_above. The maximum proportion of documents a word can
        appear in to be included in the vocabulary.
        
    no_below : int, optional
        Gensim's native parameter for min_df. If None, defaults to `min_df`.
        
    no_above : float, optional
        Gensim's native parameter for max_df. If None, defaults to `max_df`.
    """
    def __init__(
            self, min_df: Optional[int] = 1, max_df: Optional[float] = 0.95,
            ngram_range: Optional[Tuple[int, int]] = (1, 1),
            ngram_separator: Optional[str] = '_'
        ):
        self.min_df = min_df
        self.max_df = max_df
        self.dictionary: Optional[corpora.Dictionary] = None
        self.ngram_range = ngram_range
        self.ngram_separator = ngram_separator

        # Validate ngram_range parameter
        if not isinstance(self.ngram_range, tuple) or len(self.ngram_range) != 2:
            raise TypeError(f"ngram_range must be a tuple of two integers, got {self.ngram_range}")
        if self.ngram_range[0] < 1 or self.ngram_range[1] < self.ngram_range[0]:
            raise ValueError(
                f"Invalid ngram_range: {self.ngram_range}. "
                "Values must be >= 1 and min_n <= max_n."
            )

    def __create_ngrams(self, X: List[List[str]]) -> List[List[str]]:
        """
        Create n-grams using gensim's Phrases and Phraser,
        mimicking the logic of the commented-out function.
        """
        min_n, max_n = self.ngram_range

        if min_n == 1 and max_n == 1:
            return X

        for _ in range(2, max_n + 1):
            phrases = Phrases(X, min_count=1, threshold=0.1, delimiter=self.ngram_separator)
            phraser = Phraser(phrases)
            X = [phraser[doc] for doc in X]

        if min_n > 1:
            X = [
                list(filter(lambda token: len(token.split(self.ngram_separator)) >= min_n, doc))
                for doc in X
            ]
        
        return X
    
    # def __create_ngrams(self, X: List[List[str]]) -> List[List[str]]:
    #     min_n, max_n = self.ngram_range

    #     if min_n == 1 and max_n == 1:
    #         return X

    #     # Process each document to include its n-grams.
    #     return [
    #         [
    #             self.ngram_separator.join(gram)
    #             for n in range(min_n, max_n + 1)
    #             for gram in nltk.ngrams(doc, n)
    #         ]
    #         for doc in X
    #     ]

    def fit(self, X: List[List[str]], y=None):
        """
        Build the vocabulary from a list of tokenized documents.

        Parameters
        ----------
        X : List[List[str]]
            A list of documents, where each document is a list of tokens.
            
        y : None
            Ignored. This parameter exists for scikit-learn compatibility.

        Returns
        -------
        self
            The fitted vectorizer instance.
        """

        X = self.__create_ngrams(X)

        self.dictionary = corpora.Dictionary(X)
        self.dictionary.filter_extremes(no_below=self.min_df, no_above=self.max_df)
        return self
        
    def transform(self, X: List[List[str]]) -> csr_matrix:
        """
        Transform tokenized documents into a sparse Bag-of-Words matrix.

        Parameters
        ----------
        X : List[List[str]]
            A list of documents to transform.

        Returns
        -------
        scipy.sparse.csr_matrix
            The document-term matrix.
        """
        if self.dictionary is None:
            raise NotFittedError("This vectorizer has not been fitted yet.")
        
        X = self.__create_ngrams(X)
        corpus = [self.dictionary.doc2bow(tokens) for tokens in X]

        num_terms = len(self.dictionary)
        num_docs = len(X)
        
        if num_docs == 0:
            return csr_matrix((num_docs, num_terms))
            
        sparse_matrix_csc = matutils.corpus2csc(corpus, num_terms=num_terms)
        return sparse_matrix_csc.transpose().tocsr()

    # The fit_transform method is automatically provided by TransformerMixin

class SpacyPosNerPreprocessorWithStemming:

    def __init__(
            self, nlp: spacy.language.Language,
            stemmer,
            pos_whitelist: Iterable[str],
            ner_to_replace: Iterable[str]
        ):
        self.nlp = nlp
        self.pos_whitelist = pos_whitelist
        self.ner_to_replace = ner_to_replace
        self.stemmer = stemmer

    def __call__(self, text: str) -> List[str]:

        doc = self.nlp(text)

        ent_map = {}
        for ent in doc.ents:
            if ent.label_ in self.ner_to_replace:
                for i in range(ent.start, ent.end + 1):
                    ent_map[i] = ent.label_

        tokens = []
        for i, token in enumerate(doc):
            if token.text in emoji.EMOJI_DATA:
                tokens.append(token.text)
            elif i in ent_map:
                tokens.append(ent_map[i])
            else:
                if token.pos_ in self.pos_whitelist:
                    if token.is_upper:
                        tokens.append(self.stemmer.stem(token.text))
                    else:
                        tokens.append(self.stemmer.stem(token.text).lower())

        return tokens

class SpacyPosNerPreprocessor:

    def __init__(
            self, nlp: spacy.language.Language,
            pos_whitelist: Iterable[str],
            ner_to_replace: Iterable[str]
        ):
        self.nlp = nlp
        self.pos_whitelist = pos_whitelist
        self.ner_to_replace = ner_to_replace

    def __call__(self, text: str) -> List[str]:

        doc = self.nlp(text)

        ent_map = {}
        for ent in doc.ents:
            if ent.label_ in self.ner_to_replace:
                for i in range(ent.start, ent.end + 1):
                    ent_map[i] = ent.label_

        tokens = []
        for i, token in enumerate(doc):
            if token.text in emoji.EMOJI_DATA:
                tokens.append(token.text)
            elif i in ent_map:
                tokens.append(ent_map[i])
            else:
                if token.pos_ in self.pos_whitelist:
                    if token.is_upper:
                        tokens.append(token.lemma_)
                    else:
                        tokens.append(token.lemma_.lower())

        return tokens