import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils import data
from model.basic_net import BasicNet
from dataloader.midi_loader import MidiLoader
from dataloader.midi_utils import *

def _generate(net, dataset, config):
    """
    Produces notes from a sample. After a note is generated it is appended to the
    end of the segment and the last element is removed.
    :param net:
    :param dataset:
    :param config:
    :return: List of note/chords class indices.
    """
    # Which song to pick a segment from
    seed = config['song_seed']
    song = dataset.data[dataset.song_paths[seed]][()]
    gen_song = []

    # Segment is always chosen at the start of the song.
    # Modify the starting point and compare outputs
    segment = (dataset.note_2_idx(song[:config['window']]))
    # Normalize here
    segment = segment
    segment = torch.from_numpy(segment).unsqueeze(dim=0).double()

    with torch.no_grad():
        # Predict until sufficient notes have been made
        while len(gen_song) < config['song_length']:
            prediction = torch.argmax(net(segment.to(config['device'])))
            # Add note to our song
            gen_song.append(prediction.cpu().numpy())
            # Shift back one place
            segment[0, 0:-1] = segment[0, 1:].clone()
            # Add prediction at end
            # Normalize here
            segment[0, -1] = prediction

    return gen_song

def _network_to_midi(gen, note_dict):
    """
    Translate input from class_id to note/chord, then create midi file.
    :param gen:
    :param note_dict:
    :return:
    """
    """
    Translates from class index to note/chord
    :param gen:
    :param note_dict:
    :return:
    """
    id_2_note = {note_dict[f]: f for f in note_dict}
    # Decoding necessary as we encoded utf-8 into ascii
    gen_notes = [id_2_note[int(f)].decode('utf-8') for f in gen]
    midi = create_midi(gen_notes)
    return midi


def test(config_file):
    """
    Generates a song according to a config file
    :param config_file:
    :return:
    """
    # Load configuration
    with open(config_file, 'rb') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
    # Make output path
    out = os.path.join(config['out_path'], 'songs')
    make_path(out)

    # Initialize the dataset, net,
    dataset = MidiLoader(config['window'], check_notes(config_file),
                         config_file, start=0)
    net = BasicNet(
        config['window'], config['hidden_size'],
         len(dataset.unique_notes), config['num_layers'],
         config['dropout']).to(config['device']).double()
    net.eval()
    # Loop for each epoch to make a song
    for ep in config['ckpt_epoch']:
        # Load the checkpoint
        ckpt_path = os.path.join(config['out_path'], 'checkpoints')
        state = torch.load(os.path.join(get_target_ckpt(ckpt_path, ep)))
        net.load_state_dict(state['model_state_dict'])
        # Generate samples from a song segment
        gen = _generate(net, dataset, config)
        # Turn output into song and save
        midi = _network_to_midi(gen, dataset.unique_notes)
        midi_name = '{}_{}_{}.mid'.format(config['song_seed'], config['song_length'], ep)
        midi.write('midi', os.path.join(config['out_path'], 'songs', midi_name))
    return None

if __name__ == '__main__':
    test('/home/rimbriaco/PycharmProjects/rnn_music/configs/std_config.yaml')