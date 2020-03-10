import os
import re
import csv
import sys
import tqdm
import time
import torch
import random
import librosa
import warnings
import unicodedata
import numpy as np
import tensorflow as tf
import torchaudio
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu as bleu
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    unicode_csv_reader,
    walk_files)

warnings.filterwarnings('ignore')


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    # Add <padding> word to the vocabulary
    lang_tokenizer.index_word[0] = '<padding>'
    lang_tokenizer.word_index['<padding>'] = 0

    return lang_tokenizer#tensor, lang_tokenizer

def transform(batch, lang_tokenizer):
    
    tensor = lang_tokenizer.texts_to_sequences(lang)
    #tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return torch.tensor(tensor)


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
          if unicodedata.category(c) != 'Mn')



def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z0-9_*?.!,¿?åäöèÅÄÖÈÉçëË]+<>", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w



def tokenizer_librispeech(limit=10, path="../librispeech/LibriSpeech/", version = "train-clean-360"):
    ext_txt = ".trans.txt"
    ext_audio = ".flac"
    path = os.path.join(path, version)
    
    walker = walk_files(path, suffix=ext_audio, prefix=False, remove_suffix=True)
    walker = list(walker)
    all_sentences = []
    for i, fileid in enumerate(walker):
        if i==limit:
            break
        sentence = load_librispeech_item(fileid, path=path, ext_audio=ext_audio, ext_txt=ext_txt, text_only=True)
        sentence = preprocess_sentence(sentence)
        all_sentences.append(sentence)

    print(" ===== They are {} transcriptions in the dataset. ===== ".format(len(all_sentences)))  
    return tokenize(all_sentences)


def load_librispeech_item(fileid, path, ext_audio, ext_txt, text_only=False):
    
    speaker_id, chapter_id, utterance_id = fileid.split("-")
    file_text = speaker_id + "-" + chapter_id + ext_txt
    file_text = os.path.join(path, speaker_id, chapter_id, file_text)
    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    file_audio = fileid_audio + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)
    # Load audio
    if text_only is False:
        wav, sr = torchaudio.load(file_audio)
        wav = wav.numpy()[0]     
    # Load text
    with open(file_text) as ft:
        for line in ft:
            fileid_text, utterance = line.strip().split(" ", 1)
            if fileid_audio == fileid_text: 
                break
        else:
            # Translation not found
            raise FileNotFoundError("Translation not found for " + fileid_audio)
    if text_only is False:
        return wav, sr, utterance
    else:
        return utterance
    


class LibriSpeechDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    
    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"
    
    
    def __init__(self, tokenizer, limit=None, n_channels=1, n_frames=128, sr=16000, n_fft=2048, max_target_length=40,
                 n_mels=40, hop_length=512, power=1.0, n_mfcc=39, duration=10, path="../librispeech/LibriSpeech/", version = "train-clean-360"):
        'Initialization'
        self.tokenizer = tokenizer
        self.limit = limit 
        self.n_channels = n_channels
        self.n_frames = n_frames
        self.sr = sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.power = power
        self.n_mfcc = n_mfcc
        self.duration = duration
        self._path = os.path.join(path, version)
        walker = walk_files(self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True)
        self._walker = list(walker)[:limit]
        self.max_length = max_target_length
        


    def __len__(self):
        'Denotes the total number of samples'
        return len(self._walker)
    
    
    def wave_augmenter(self, wave):
        """ Choose to randomly apply an augmentation to a wave sequence."""
        return wave
    
    def spec_augmenter(self, spec):
        """ Perform spectogram level's augmentation for audio data."""
        return spec
    
    
    def wave2mfcc(self, wave):
        """ 
        Opens an wav audio file with librosa and converts it in mfccs features.
        
        :param path_wav:
        
        """
        # We create the mfcc
        mfccs = librosa.feature.mfcc(y=wave, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
                                     power=self.power, n_mels=self.n_mels, n_mfcc=self.n_mfcc)
        # Normalization of the mfccs
        mfccs = ((mfccs.T - mfccs.mean(axis=1)) / mfccs.std(axis=1)).T 
        
        return mfccs 
    
    
    def str2num(self, sentence, lang_tokenizer):
        
        tensor = lang_tokenizer.texts_to_sequences([sentence])
        
        return tensor
    
    
    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        fileid = self._walker[index]
        # Load data and get label
        wav, sr, sentence = load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)
        # Random augmentation at wav level
        wav = self.wave_augmenter(wav)
        # transformer en spec
        mat = self.wave2mfcc(wav)
        # Random augmentation at sepectogram level
        mat = self.spec_augmenter(mat)
        # Padding to have the same number of frame in each mfccs
        if mat.shape[1] < self.n_frames:
            mat = np.array(np.pad(mat, ((0,0), (0, self.n_frames - mat.shape[1])), 'constant', constant_values=0)) 
        else:
            mat = mat[:, :self.n_frames]
        # 
        mat = mat.reshape((1, *mat.shape))
        sentence = preprocess_sentence(sentence)
        sentence = self.str2num(sentence, self.tokenizer)[0]
        if len(sentence) < self.max_length:
            sentence = np.array(np.pad(sentence, (0, self.max_length - len(sentence)), 'constant', constant_values=0)) 
        else:
            sentence = sentence[:self.max_length]        
        
        return torch.tensor(mat), torch.tensor(sentence)
    
    
def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, device, batch_sz, targ_lang, teacher_forcing_ratio=0.5):
    
    # Initialize the encoder
    encoder_hidden = encoder.initialize_hidden_state()
    # Put all the previously computed gradients to zero
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    target_length = target_tensor.size(1)
    # Encode the input sentence
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    
    
    loss = 0
    decoder_input = torch.tensor([[targ_lang.word_index['<start>']]] * batch_sz, device=device)
    decoder_hidden =  encoder_hidden
    # Use randomly teacher forcing
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True  
    else:  
        use_teacher_forcing = False

    #if use_teacher_forcing:
    # Teacher forcing: Feed the target as the next input to help the model
    # in case it starts with the wrong word.
    for di in range(1, target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor[:, di])
        if use_teacher_forcing:
            decoder_input = torch.unsqueeze(target_tensor[:, di], 1)  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            topv, topi = decoder_output.data.topk(1)
            # the predicted ID is fed back into the model
            decoder_input = topi.detach()

    batch_loss = (loss.item() / int(target_tensor.shape[1]))
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return batch_loss



def global_trainer(nbr_epochs, dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer,
                                    criterion, device, batch_sz, tokenizer ):
    
    nb_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    print(" ======"*6)
    print("      The model has {} parameters".format(nb_params))
    print(" ======"*6)
    print("\n")
    
    start = time.time()
    for epoch in range(nbr_epochs):
        #
        total_loss = 0


        with tqdm.tqdm(total=len(dataloader), file=sys.stdout, leave=True, desc='Epoch ', \
                       bar_format="{l_bar}{bar:20}{r_bar}{bar:-15b}") as pbar:    
            for batch, (inp, targ) in enumerate(dataloader):

                pbar.set_description('Epoch {}'.format(epoch + 1))
                inp, targ = inp.to(device), targ.to(device)
                batch_loss = train_step(inp, targ, encoder, decoder, encoder_optimizer,
                                        decoder_optimizer, criterion,
                                        device, batch_sz, targ_lang=tokenizer)
                total_loss += batch_loss
                pbar.set_postfix_str('Loss {:.4f}'.format(total_loss / (batch + 1)))
                pbar.update(1)
                time.sleep(1)

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            torch.save(encoder, 'encoder-s2t.pt')
            torch.save(decoder, 'decoder-s2t.pt')

    print('\nTime taken for the training {:.5} hours\n'.format((time.time() - start) / 3600))
    
    
        
def greedy_decode(mfccs, max_length_targ, encoder, decoder, targ_lang, device, hidden_size=64):

    
    # Send the inputs matrix to device
    mfccs = torch.tensor(mfccs).to(device)
    
    #print("mfccs", mfccs.shape)

    result = ''

    with torch.no_grad():
        enc_hidden = torch.zeros(2, 1, hidden_size, device=device)
        enc_out, enc_hidden = encoder(mfccs, enc_hidden)
        
        #print("EO", enc_out.shape)

        dec_hidden = enc_hidden
        dec_input = torch.tensor([[targ_lang.word_index['<start>']]], device=device)

        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

            # storing the attention weights to plot later on
            topv, topi = predictions.data.topk(1)
            result += targ_lang.index_word[topi.item()] + ' '

            if targ_lang.index_word[topi.item()] == '<end>':
                return result, sentence

            # the predicted ID is fed back into the model
            dec_input = torch.tensor([topi.item()], device=device).unsqueeze(0)

        return result
    

    
class BeamTreeNode(object):
    "Generic tree node."
        
    def __init__(self, name, hidden_state, wordid=1, logp=1,  children=None, parent=None):
        self.name = name
        self.h = hidden_state
        self.wordid = wordid
        self.logp = logp
        self.children = []
        self.parent = parent
        self.is_leaf = True
        self.is_end = False
        self.length = 0
        self.inv_path = [self.wordid.item()]

    @property
    def path(self):
        return self.inv_path[::-1]
    
    def __lt__(self, other):
        return self.logp.item() < other.logp.item() 
    
    def __eq__(self, other):
        return self.logp.item() == other.logp.item() 

    def __repr__(self):
        return self.name
    
    def is_child(self, node):
        return node in self.children
        
    def add_child(self, node):
        assert isinstance(node, BeamTreeNode)
        assert self.is_child(node) == False
        self.children.append(node)
        self.is_leaf = False

    def add_parent(self, node):
        assert isinstance(node, BeamTreeNode)
        self.parent = node
        self.length = node.length + 1
        self.inv_path += node.inv_path
    
        
def beam_search_decode(sentence, max_length_targ, max_length_inp, encoder, decoder, inp_lang, 
                       targ_lang, device, nb_candidates, beam_width, alpha):

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = torch.tensor(inputs).long().to(device)

    result = ''

    with torch.no_grad():
        hidden = torch.zeros(1, 1, 1024, device=device)
        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = torch.tensor([[targ_lang.word_index['<start>']]], device=device)
        
        candidates = []
        # Créer la racinne (le noeud de départ de l'arbre)
        node = BeamTreeNode(name='root', hidden_state=dec_hidden, wordid=dec_input, logp=torch.tensor(0, device=device))
        candidates.append(node)
        
        count = 0
        endnodes = []
        for t in range(max_length_targ):
            all_nodes = PriorityQueue()
            for n in candidates:
                if n.is_leaf and not n.is_end:
                    # étendre le noeud (faire les prédictions dessus)
                    #print(n.wordid)
                    predictions, dec_hidden, attention_weights = decoder(n.wordid, n.h, enc_out)
                    # Pour signaler que le noeud est déjà étendu (utilisé)
                    n.is_leaf = False
                    # prendre le nombre de candidats choisis 
                    top_width_v, top_width_i = predictions.data.topk(nb_candidates)
                    # Créer beam width noeuds pour stocker les prédictions et les rajouter à 
                    # la liste de noeuds à scorer
                    for val, ind in zip(top_width_v[0], top_width_i[0]):
                        count += 1
                        dec_input = torch.tensor([[ind.item()]], device=device)
                        logproba = -val + n.logp
                        node = BeamTreeNode(name=str(count), hidden_state=dec_hidden, wordid=dec_input, logp=logproba)
                        # Rajouter le noeud à la priority queue
                        all_nodes.put(node)
                        # Indiquer que les nouveaux noeuds sont des enfants du noeud initial 
                        n.add_child(node)
                        node.add_parent(n)
                        # Si on prédit la fin ou que la longueur maximale est atteinte
                        if targ_lang.index_word[ind.item()] == '<end>':
                            node.is_end = True 
                            endnodes.append(node)
       
            # Retenir que les beam width meilleurs           
            candidates = [all_nodes.get() for step in range(beam_width)]
            #candidates = [node for _, node in candidates]
            candidates = [node for node in candidates]
            
        # Last step before the result 
        final_queue = PriorityQueue()
        final_candidates = candidates + endnodes
        # Put all final candidates nodes in a priority queue and choose the best one based 
        # on the score and not on the logp
        for n in final_candidates:
            score = n.logp / (n.length ** alpha)
            final_queue.put((score, n))
        # Choose the best node  
        _, node = final_queue.get()
        # Find the path
        for elem in node.path:
            if elem != 0:
                result += targ_lang.index_word[elem] + ' '

        return result, sentence         
        
        

def evaluate(mfccs, references, max_length_targ, encoder, decoder, targ_lang, 
              device, beam_search=False, beam_width=3, alpha=0.3, nb_candidates=50):
    
    if beam_search == False:
        result= greedy_decode(mfccs, max_length_targ, encoder, decoder, targ_lang, device)
    else:
        result, sentence = beam_search_decode(sentence, max_length_targ, max_length_inp, 
                                              encoder, decoder, inp_lang, targ_lang, device,
                                              beam_width=beam_width, nb_candidates=nb_candidates, alpha=alpha)
    result = result.split()    
    BLEUscore = bleu([references], result, weights = (0.5, 0.5))
    
    print("Input: {}".format(references))
    print("\n")
    print("Predicted translation: {}".format(result))
    print("\n")
    print("Bleu score: {}".format(BLEUscore))
    
    