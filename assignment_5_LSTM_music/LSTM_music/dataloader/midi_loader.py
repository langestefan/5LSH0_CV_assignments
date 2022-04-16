import torch
from torch.utils import data
from dataloader.midi_utils import *

class MidiLoader:
    def __init__(self, window, data, config_file, start=None):
        """
        Initialize midi dataloader.
        :param window: Sample length
        :param data: Hdf5 file object
        :param config_file:
        :param start: Hand picked starting point for the segments can be
        selected or set during training. If 'None' music segments are chosen
        randomly as long as they fit the window. Otherwise, each segment is equal
        to song[start:start+window]. If start is manually set, make sure to
        increase/reset its value each time data is loaded via 'set_start'.
        """
        self.window = window
        self.data = data
        with open(config_file, 'rb') as f:
            self.config = yaml.load(f, Loader=yaml.UnsafeLoader)
        self.unique_notes = self._note_dict()
        self.song_paths = [f for f in self.data.keys() if '.mid' in f]
        self.start = start

    def _note_dict(self):
        """
        Dictionary of notes useful for note-to-class_id conversion.
        :return:
        """
        notes = self.data['unique'][()]
        note_dict = {i: en_i for en_i, i in enumerate(notes)}
        return note_dict

    def note_2_idx(self, note_list):
        """
        Converts a list of notes/chords into class_ids.
        :param note_list:
        :return:
        """
        return np.array([self.unique_notes[f] for f in note_list])

    def set_start(self, start):
        """
        Setter for 'start' attribute.
        :param start:
        :return:
        """
        self.start = start

    def __len__(self):
        """
        Dataset length
        :return:
        """
        return len(self.song_paths)

    def __getitem__(self, item):
        """
        Returns a song segment.
        :param item:
        :return:
        """
        notes = self.data[self.song_paths[item]][()]

        # Randomize if not set
        if self.start is None:
            start = torch.randint(0, len(notes) - self.window, (1, 1))
        else:
            start = self.start
        # Randomize if segment left is too short
        if start + self.window >= len(notes):
            start = torch.randint(0, len(notes) - self.window, (1, 1))

        input_notes = notes[start:start+self.window]
        input_notes = np.array(self.note_2_idx(input_notes))
        output_note = np.array(self.unique_notes[notes[start + self.window]])
        # Return window + Following note
        return torch.from_numpy(input_notes), torch.from_numpy(output_note)

if __name__ == '__main__':
    ml = MidiLoader(50, '/home/rimbriaco/PycharmProjects/rnn_music/configs/std_config.yaml')
    ml.set_start(0)
    dl = data.DataLoader(ml, 20, shuffle=True, num_workers=1)
    for inp, tgt in dl:
        print(inp)
        print(tgt)