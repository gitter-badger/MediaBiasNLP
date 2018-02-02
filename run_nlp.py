from __future__ import unicode_literals
import glob
import errno
from bs4 import BeautifulSoup
import plac
import random
#import pathlib
import cytoolz
import numpy
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
#import thinc.extra.datasets
import spacy
from spacy.compat import pickle
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token
from spacy.language import Language


#@plac.annotations(
#    vectors_loc=("Path to .vec file", "positional", None, str),
#    lang=("Optional language ID. If not set, blank Language() will be used.",
#          "positional", None, str))


class PetersonRecognizer(object):
    """Example of a spaCy v2.0 pipeline component that sets entity annotations
    based on list of single or multiple-word company names. Companies are
    labelled as ORG and their spans are merged into one token. Additionally,
    ._.has_tech_org and ._.is_tech_org is set on the Doc/Span and Token
    respectively."""
    name = 'tech_companies'  # component name, will show up in the pipeline

    def __init__(self, nlp, companies=tuple(), label='ORG'):
        """Initialise the pipeline component. The shared nlp instance is used
        to initialise the matcher with the shared vocab, get the label ID and
        generate Doc objects as phrase match patterns.
        """
        self.label = nlp.vocab.strings[label]  # get entity label ID

        # Set up the PhraseMatcher – it can now take Doc objects as patterns,
        # so even if the list of companies is long, it's very efficient
        patterns = [nlp(org) for org in companies]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add('TECH_ORGS', None, *patterns)

        # Register attribute on the Token. We'll be overwriting this based on
        # the matches, so we're only setting a default value, not a getter.
        Token.set_extension('is_tech_org', default=False)

        # Register attributes on Doc and Span via a getter that checks if one of
        # the contained tokens is set to is_tech_org == True.
        Doc.set_extension('has_tech_org', getter=self.has_tech_org)
        Span.set_extension('has_tech_org', getter=self.has_tech_org)

    def __call__(self, doc):
        """Apply the pipeline component on a Doc object and modify it if matches
        are found. Return the Doc, so it can be processed by the next component
        in the pipeline, if available.
        """
        matches = self.matcher(doc)
        spans = []  # keep the spans for later so we can merge them afterwards
        for _, start, end in matches:
            # Generate Span representing the entity & set label
            entity = Span(doc, start, end, label=self.label)
            spans.append(entity)
            # Set custom attribute on each token of the entity
            for token in entity:
                token._.set('is_tech_org', True)
            # Overwrite doc.ents and add entity – be careful not to replace!
            doc.ents = list(doc.ents) + [entity]
        for span in spans:
            # Iterate over all spans and merge them into one token. This is done
            # after setting the entities – otherwise, it would cause mismatched
            # indices!
            span.merge()
        return doc  # don't forget to return the Doc!

    def has_tech_org(self, tokens):
        """Getter for Doc and Span attributes. Returns True if one of the tokens
        is a tech org. Since the getter is only called when we access the
        attribute, we can refer to the Token's 'is_tech_org' attribute here,
        which is already set in the processing step."""
        return any([t._.get('is_tech_org') for t in tokens])


def read_first_line(file):
    """Gets the first line from a file.

    Returns
    -------
    str
        the first line text of the input file
    """
    with open(file, 'rt') as fd:
        first_line = fd.readline()
    return first_line

def merge_per_folder(folder_path, output_filename):
    """Merges first lines of text files in one folder, and
    writes combined lines into new output file

    Parameters
    ----------
    folder_path : str
        String representation of the folder path containing the text files.
    output_filename : str
        Name of the output file the merged lines will be written to.
    """
    # make sure there's a slash to the folder path 
    folder_path += "" if folder_path[-1] == "/" else "/"
    # get all text files
    # txt_files = glob.glob(folder_path + "*.txt")
    input_files = glob.glob(folder_path+"/**/*.html", recursive="True")
    # get first lines; map to each text file (sorted)
    output_strings = map(read_first_line, sorted(input_files))
    output_content = "".join(output_strings)
    # write to file
    with open(folder_path + output_filename, 'wt') as outfile:
        outfile.write(output_content)


class SentimentAnalyser(object):
    @classmethod
    def load(cls, path, nlp, max_length=100):
        with (path / 'config.json').open() as file_:
            model = model_from_json(file_.read())
        with (path / 'model').open('rb') as file_:
            lstm_weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, max_length=max_length)

    def __init__(self, model, max_length=100):
        self._model = model
        self.max_length = max_length

    def __call__(self, doc):
        X = get_features([doc], self.max_length)
        y = self._model.predict(X)
        self.set_sentiment(doc, y)

    def pipe(self, docs, batch_size=1000, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            sentences = []
            for doc in minibatch:
                sentences.extend(doc.sents)
            Xs = get_features(sentences, self.max_length)
            ys = self._model.predict(Xs)
            for sent, label in zip(sentences, ys):
                sent.doc.sentiment += label - 0.5
            for doc in minibatch:
                yield doc

    def set_sentiment(self, doc, y):
        doc.sentiment = float(y[0])
        # Sentiment has a native slot for a single float.
        # For arbitrary data storage, there's:
        # doc.user_data['my_data'] = y


def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, numpy.asarray(labels, dtype='int32')


def get_features(docs, max_length):
    docs = list(docs)
    Xs = numpy.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                Xs[i, j] = vector_id
            else:
                Xs[i, j] = 0
            j += 1
            if j >= max_length:
                break
    return Xs


def train(train_texts, train_labels, dev_texts, dev_labels,
          lstm_shape, lstm_settings, lstm_optimizer, batch_size=100,
          nb_epoch=5, by_sentence=True):
    print("Loading spaCy")
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    embeddings = get_embeddings(nlp.vocab)
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)
    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts))
    dev_docs = list(nlp.pipe(dev_texts))
    if by_sentence:
        train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
        dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)

    train_X = get_features(train_docs, lstm_shape['max_length'])
    dev_X = get_features(dev_docs, lstm_shape['max_length'])
    model.fit(train_X, train_labels, validation_data=(dev_X, dev_labels),
              nb_epoch=nb_epoch, batch_size=batch_size)
    return model


def compile_lstm(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    model.add(TimeDistributed(Dense(shape['nr_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['nr_hidden'],
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(Dense(shape['nr_class'], activation='sigmoid'))
    model.compile(optimizer=Adam(lr=settings['lr']), loss='binary_crossentropy',
		  metrics=['accuracy'])
    return model


def get_embeddings(vocab):
    return vocab.vectors.data


def evaluate(model_dir, texts, labels, max_length=100):
    def create_pipeline(nlp):
        '''
        This could be a lambda, but named functions are easier to read in Python.
        '''
        return [nlp.tagger, nlp.parser, SentimentAnalyser.load(model_dir, nlp,
                                                               max_length=max_length)]

    nlp = spacy.load('en')
    nlp.pipeline = create_pipeline(nlp)

    correct = 0
    i = 0
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):
        correct += bool(doc.sentiment >= 0.5) == bool(labels[i])
        i += 1
    return float(correct) / i


def read_data(data_dir, limit=0):
    examples = []
    for subdir, label in (('pos', 1), ('neg', 0)):
        for filename in (data_dir / subdir).iterdir():
            with filename.open() as file_:
                text = file_.read()
            examples.append((text, label))
    random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]
    return zip(*examples) # Unzips into two lists


#@plac.annotations(
#    train_dir=("Location of training file or directory"),
#    dev_dir=("Location of development file or directory"),
#    model_dir=("Location of output model directory",),
#    is_runtime=("Demonstrate run-time usage", "flag", "r", bool),
#    nr_hidden=("Number of hidden units", "option", "H", int),
#    max_length=("Maximum sentence length", "option", "L", int),
#    dropout=("Dropout", "option", "d", float),
#    learn_rate=("Learn rate", "option", "e", float),
#    nb_epoch=("Number of training epochs", "option", "i", int),
#    batch_size=("Size of minibatches for training LSTM", "option", "b", int),
#    nr_examples=("Limit to N examples", "option", "n", int)
#)


def main():
    
    # Analyze article with spacy
    # Load the model, set up the pipeline
    nlp = English()
    keywords = ['Peterson', 'Jordan', 'Jordan Peterson', 'Jordan B. Peterson']  
    component = PetersonRecognizer(nlp, keywords)  # initialise component
    nlp.add_pipe(component, last=True)  # add last to the pipeline
    
    # define folder to be loaded
    folder_in = "/home/stefano/Code/JordanPetersonScrape/HTML"
    
    # Create titles .txt file 
    # merge_per_folder(folder_in, "Titles.txt")
    
    # Load all files
    path = folder_in + "/**/*.html"
    files = glob.glob(path, recursive="True")
    i = 1
    #file = open("Titles.txt", "w")
    for name in files[:2]:     # selected only first 2 files for speed
        try:
            with open(name) as f:
                #print(f.read())
                
                soup = BeautifulSoup(f.read())

#                for h in soup.find_all('h1'):
#                    print(h.text)
#                for h in soup.find_all('h2'):
#                    print(h.text)
#                for h in soup.find_all('h3'):
#                    print(h.text)
#                for h in soup.find_all('a'):
#                    print(h.text)
                
                
                # Save h1 tags to Titles.txt file
                if False:
                    for h in soup.find_all('h1'):
                        text = h.text
                        text = ' '.join(text.split())
                        #file.write(str(i) + ";" + text + ";" + "\n")
                
                
                links = [e.get_text() for e in soup.find_all('p')]
                article = '\n'.join(links)
                print (len(article))
                

                doc = nlp(article)
                print('Pipeline', nlp.pipe_names)  # pipeline contains component name
                print('Tokens', [t.text for t in doc])  # company names from the list are merged
                print('Doc has_tech_org', doc._.has_tech_org)  # Doc contains tech orgs
                print('Token 0 is_tech_org', doc[0]._.is_tech_org)  # "Alphabet Inc." is a tech org
                print('Token 1 is_tech_org', doc[1]._.is_tech_org)  # "is" is not
                print('Entities', [(e.text, e.label_) for e in doc.ents]) # all orgs are entities
                
                print(name)
                print("Analysis of Document " + str(i))
                print("Number of HTML Tags :" ,len(soup.find_all('a')))
                print("Number of h1 :" ,len(soup.find_all('h1')))
                print("Number of h2 :" ,len(soup.find_all('h2')))
                print("Number of h3 :" ,len(soup.find_all('h3')))
                
                
                nlp_sent = spacy.load("en")
                doc = nlp_sent(article)
                # Get tags of the text
                all_tags = {w.pos: w.pos_ for w in doc}       
                print(all_tags)
                
#                # load vectors
#                path_to_vectors = "/path/to/glove/vectors"
#                nlp_sent.vocab.vectors.from_glove(path_to_vectors)
#                
#                
#                print("DEBUG!")
#                
#                
#                # ADDITION (USE WORDS VECTORS!)   # [TO DO]
#                vectors_loc = "/path/to/fastText/Vectors" 
#                nlp_sentiment = Language()
#                with open(vectors_loc, 'rb') as file_:
#                    header = file_.readline()
#                    nr_row, nr_dim = header.split()
#                    nlp_sentiment.vocab.reset_vectors(width=int(nr_dim))
#                    for line in file_:
#                        line = line.rstrip().decode('utf8')
#                        pieces = line.rsplit(' ', int(nr_dim))
#                        word = pieces[0]
#                        vector = numpy.asarray([float(v) for v in pieces[1:]], dtype='f')
#                        nlp_sentiment.vocab.set_vector(word, vector)  # add the vectors to the vocab
#                # test the vectors and similarity
#                text = 'class colspan'
#                doc = nlp_sentiment(text)
#                print(text, doc[0].similarity(doc[1]))
#
#                
#                
                
                
                
                
                
                
                
                
                
                
                
                
                i = i + 1
                input("Press any key to get to the next one...")

                
                
                
                
                
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    #file.close()


if __name__== "__main__": 
    main()
# END!



































