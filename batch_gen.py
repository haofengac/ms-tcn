#!/usr/bin/python2.7

import torch
import numpy as np
import random
import os


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, base_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.base_path = base_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy').T
            split_info = vid.split('_')
            if 'stereo' in vid:
                file_path = os.path.join(self.base_path, split_info[0], split_info[1][:-2], '_'.join(split_info[2:] + ['ch{}'.format(int(split_info[1][-2:])-1)]) + '.avi.labels')
            else:
                file_path = os.path.join(self.base_path, split_info[0], split_info[1], '_'.join(split_info[2:]) + '.avi.labels')
            file_ptr = open(file_path, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(np.shape(features)[1])
            for l in content:
                se, c = l.split()
                s, e = se.split('-')
                s, e = int(s), int(e)
                if (s < e) and (c in self.actions_dict):
                    classes[s-1:e-1] = self.actions_dict[c]

            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
