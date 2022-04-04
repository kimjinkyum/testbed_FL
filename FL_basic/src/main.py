"""
with open("server.log", 'r') as f:
    logs = f.readlines()
    for log in logs:
        print(log.split("_"))

"""
import requests
import os
import numpy as np
import torch.utils.data
from collections import Counter
from tqdm import tqdm
from options import *
from FL_server import FL_server
from concurrent.futures import ThreadPoolExecutor
from utils import *
from models import *
import logging
from update import *

# 로그 생성
logger = logging.getLogger("Server")
logger.setLevel(logging.INFO)

# log 출력 형식
formatter = logging.Formatter(' %(name)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler('randoms_0118_5.log', mode="w")
logger.addHandler(stream_handler)

formatter_float = logging.Formatter('(message)%s')
stream_handler.setFormatter(formatter)

ip_lists = []
application_name = "fl-ex"


def server_aggregate(global_model, client_models):
    # FedAvg
    global_model.train()
    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        # print(global_dict[k].mean())

        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))],
                                     0).mean(0)

    global_model.load_state_dict(global_dict)

    for model in client_models:
        model.load_state_dict(global_model.state_dict())

    return global_model, client_models


class DatasetSplit(Dataset):
    def __init__(self, dataset_train, idxs):
        self.dataset_train = dataset_train
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset_train[self.idxs[item]]
        return image, label


def iid(data_size, num_user, dataset_train, args):
    num_items = data_size
    dict_users, all_idxs = {}, [i for i in range(len(dataset_train))]
    for i in range(num_user):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    logger.info("Data length of users {0} : {1}".format(i, len(dict_users[0])))

    client_model, client_model_ob = data_split(dict_users, num_user, dataset_train, global_model, args)

    return client_model, client_model_ob


def data_split(dict_users, num_user, dataset_train, global_model, args):
    client_model_ob = []
    client_model = [CNNCifar(args).to('cuda') for _ in range(num_user)]
    local_datasets = []

    for i in range(num_user):
        local_datasets.append(DataLoader(DatasetSplit(dataset_train, dict_users[i]),
                                         batch_size=32, shuffle=True))
        temp = [label for _, (_, label) in enumerate(local_datasets[i])]
        temp = [a.numpy() for a in temp]
        temp = np.concatenate(temp, 0)
        unique, count = np.unique(temp, return_counts=True)
        client_model_ob.append(
            LocalUpdate(args=args, dataset=local_datasets[i], idxs=dict_users[i], logger=logger, user_index=i))
        client_model[i].load_state_dict(global_model.state_dict())
    # logger.info("Data Distribution of users {0} : {1}".format(i, dict(zip(unique, count))))
    return client_model, client_model_ob


# 여기서 distributed_class = 10 으로 두면 IID 케이스 가능
def non_iid(dataset_train, distributed_class, num_user, global_model, data_size, args):
    num_class = 10
    data_size_total = 50000
    num_label = data_size_total / num_class
    dict_users_train = {i: np.array([]) for i in range(num_user)}

    labels = np.array(dataset_train.targets)

    # Sorting
    idxs_labels = np.vstack((np.arange(data_size_total), labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    for i in range(num_user):
        random_class = np.random.choice(np.arange(num_class), distributed_class, replace=False)
        for rand in random_class:
            random_select = np.arange(num_label * rand, num_label * rand + num_label)
            random_selected = np.random.choice(random_select, int(data_size / distributed_class), replace=False).astype(
                "int")

            T = [idxs[i] for i in random_selected]

            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], T), axis=0)
            dict_users_train[i] = list(map(int, dict_users_train[i]))

    client_model, client_model_ob = data_split(dict_users_train, num_user, dataset_train, global_model, args)
    return client_model, client_model_ob


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dataset_train, test = get_dataset("cifar")
    args = args_parser()
    global_model = CNNCifar(args=args)
    global_model.to('cuda')
    global_model.train()
    num_user = 20

    # start_data = args["size_start"]
    # end_data = args["size_end"]
    # distributed_class = args["class_non"]

    # TODO: 파일 이름 꼭 변경!
    file_name = "log/ test_now.log"

    logger_file = logging.getLogger("File")
    logger_file.setLevel(logging.INFO)

    file_handler = logging.FileHandler(file_name, mode="w")
    logger.addHandler(stream_handler)

    formatter_float = logging.Formatter('(message)%s')
    stream_handler.setFormatter(formatter)

    logger_file.addHandler(stream_handler)
    logger_file.addHandler(file_handler)

    # dict_users_train = split_non_iid(dataset_train, 2, 5, global_model)
    data_size_list = [1000, 2000]
    for data_size in tqdm(data_size_list):
        logger.info("Data Size {0} start".format(data_size))
        for mc in range(1):
            global_model = CNNCifar(args=args)
            global_model.to('cuda')
            global_model.train()

            # TODO: distriubted_class = 10 으로 두면 IID 경우
            # Distributed_class가 각 클라이언트가 몇개 label 가지고 있을건지 --> 10개 라벨중 10개 가지면 IID
            distributed_class = 10
            client_model, client_model_ob = non_iid(dataset_train, distributed_class, num_user, global_model, data_size,
                                                    args)
            total_loss = []
            total_acc_test = []
            total_acc_train = []
            for rounds in range(1, 50 + 1):
                loss = 0
                acc = 0
                global_model.train()

                for user in range(num_user):
                    loss += client_model_ob[user].update_weights(model=client_model[user], global_round=rounds)

                total_loss.append(loss / num_user)

                global_model, client_model = server_aggregate(global_model, client_model)

                acc, test_loss = test_inference(global_model, test)
                total_acc_test.append(acc)
                logger_file.info("{0}_{1}_{2}".format(data_size, rounds, total_acc_test[-1]))
                if rounds % 10 == 0:
                    logger.info("Round {0} Loss {1}".format(rounds, loss / num_user))

            if mc % 10 == 0:
                logger.info("{0} - {1} Test Accuracy {2}".format(data_size, mc, acc))
