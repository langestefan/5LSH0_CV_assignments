import os
import yaml
import torch
from torch.utils import data
from torch.optim import SGD, Adam
from model.basic_net import BasicNet
from torch.nn import CrossEntropyLoss
from dataloader.midi_loader import MidiLoader
from dataloader.midi_utils import make_path, check_notes, get_latest_ckpt


def _train_epoch(net, dataloader, loss, opt, config):
    """
    Train a single epoch and computes the loss. Since the data loader only
    outputs a single segment this process is repeated N times with randomly
    selected segments.
    :param net:
    :param dataloader:
    :param loss:
    :param opt:
    :param config:
    :return: Current loss
    """
    for i in range(config['windows_per_epoch']):
        for inp, tgt in dataloader:
            inp = inp.type(torch.LongTensor)   # casting to long
            inp = inp.to(config['device']).double()

            tgt = tgt.type(torch.LongTensor)   # casting to long
            tgt = tgt.to(config['device'])

            # inp = inp.to(config['device']).double()
            # tgt = tgt.to(config['device'])

            opt.zero_grad()
            output = net(inp)
            loss_out = loss(output.squeeze(), tgt)
            loss_out.backward()
            opt.step()
    return loss_out.data

def train(config_file):
    """
    Trains a network based on a configuration file.
    :param config_file:
    :return:
    """
    # Initialize configs
    with open(config_file, 'rb') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
    # Initialize dataset, net, loss and optimizer
    dataset = MidiLoader(config['window'], check_notes(config_file), config_file)
    train_loader = data.DataLoader(dataset, config['batch_size'],
                                   shuffle=True, num_workers=0)
    net = BasicNet(config['window'], config['hidden_size'],
                   len(dataset.unique_notes), config['num_layers'],
                   config['dropout']).to(config['device']).double()
    loss = CrossEntropyLoss()
    opt = Adam(net.parameters(), config['lr'])

    start_epoch = 0
    # If resuming from checkpoint load the weights
    if config['resume']:
        state = torch.load(
            os.path.join(get_latest_ckpt(os.path.join(config['out_path'], 'checkpoints')))
        )
        net.load_state_dict(state['model_state_dict'])
        start_epoch = state['epoch']
    # Train and save loop
    for ep in range(start_epoch, config['epochs']):
        net.train()
        ep_loss = _train_epoch(net, train_loader, loss, opt, config)
        print(ep, ep_loss.data)
        if (ep+1) % config['ckpt_step'] == 0:
            save_dict = {
                    'epoch': (ep+1),
                    'model_state_dict': net.state_dict()
            }
            out = os.path.join(config['out_path'], 'checkpoints')
            make_path(out)
            torch.save(save_dict, os.path.join(out, 'epoch_{}.pth'.format(ep + 1)))


if __name__ == '__main__':
    train('/home/rimbriaco/PycharmProjects/rnn_music/configs/std_config.yaml')
