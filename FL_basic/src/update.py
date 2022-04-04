#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
import random
import numpy as np


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, user_index):
        self.args = args
        self.logger = logger
        self.trainloader = dataset
        # self.trainloader, self.testloader = self.train_val_test(
        #    dataset, list(idxs) )
        self.device = 'cuda'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        self.user_index = user_index

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_test = idxs[int(0.8 * len(idxs)):]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test) / 10), shuffle=False)
        return trainloader, testloader

    def update_weights(self, model, global_round):

        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args["optimizer"] == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args["lr"],
                                        momentum=0.5)
        elif self.args["optimizer"] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args["lr"],
                                         weight_decay=1e-4)

        for iter in range(self.args["local_ep"]):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                """
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                """
                # self.logger.info('loss {0}'.format(loss.item()))
                batch_loss.append(loss.item())
            """
            if self.args.verbose:
                print('| Global Round : {} | Local Epoch : {}, Client {}, \tLoss: {:.6f}'.format(
                    global_round, iter, self.user_index, sum(batch_loss) / len(batch_loss)))
            """
        """
        print('| Global Round : {} | Local Epoch : {}, Client {}, \tLoss: {:.6f}'.format(
            global_round, iter, self.user_index, sum(batch_loss) / len(batch_loss)))
        """
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return sum(batch_loss) / len(batch_loss)  # loss.item()

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)

                loss += self.criterion(outputs, labels).item()
                pred_labels = outputs.argmax(dim=1, keepdim=True)
                correct += pred_labels.eq(labels.view_as(pred_labels)).sum().item()

        loss /= len(self.testloader.dataset)
        accuracy = correct / len(self.testloader.dataset)

        return accuracy, loss

    def reduce_data_number(self, number):
        labels = [self.trainloader.dataset[i][1] for i in range(len(self.trainloader.dataset))]
        labels = np.array(labels)
        select_index = []
        for k in number.keys():
            tmp = list(np.where(k == labels))[0]
            print(len(tmp), number[k])
            tmp_index = list(np.random.choice(tmp, int(number[k]), replace=False))
            select_index.extend(tmp_index)
            # print(k, len(tmp_index))

    def reduce_data(self, reduce):
        labels = [self.trainloader.dataset[i][1] for i in range(len(self.trainloader.dataset))]
        labels = np.array(labels)
        select_index = []
        for i in np.unique(labels):
            tmp = list((np.where(i == labels)))[0]
            tmp_index = list(np.random.choice(tmp, int(len(tmp) * reduce), replace=False))
            select_index.extend(tmp_index)

        my_subset = Subset(self.trainloader.dataset, select_index)
        self.trainloader = DataLoader(my_subset, batch_size=32, shuffle=True)

        print("Finish reduce dataset", len(self.trainloader.dataset))


def test_inference(model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)

            loss += criterion(outputs, labels).item()
            pred_labels = outputs.argmax(dim=1, keepdim=True)
            correct += pred_labels.eq(labels.view_as(pred_labels)).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)

    return accuracy, loss
