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
    
    
    def wave2mfcc(self, wave):
        """ 
        Opens an wav audio file with librosa and converts it in mfccs features.
        
        :param path_wav:
        
        """
        # Open the audio file
        #wave, srate = librosa.load(path_wav, duration=self.duration, mono=True, sr=self.sr)
        # Random augmentation
        # wave = self.wave_augmenter(wave)
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
        # Augmenter le wave
        
        # transformer en spec
        mat = self.wave2mfcc(wav)
        # Augmenter le spec
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
    #encoder_hidden = encoder.initialize_hidden_state().to(device)
    # Put all the previously computed gradients to zero
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_tensor.size(1)
    
    # Encode the input sentence
    encoder_outputs, encoder_hidden = encoder(input_tensor)#, encoder_hidden)
    
    
    loss = 0
    decoder_input = torch.tensor([[targ_lang.word_index['<start>']]] * batch_sz, device=device)
    decoder_hidden =  encoder_hidden#encoder.last_state#encoder_hidden

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
    
    start = time.time()
    for epoch in range(nbr_epochs):
        #
        
        total_loss = 0


        with tqdm.tqdm(total=len(dataloader), file=sys.stdout, leave=True, desc='Epoch ', bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}") as pbar:    
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