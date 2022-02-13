# coding=utf-8
import numpy as np
import json
from options import args_parser
import requests
from models import *
import pickle
import torch
from io import BytesIO
from utils import *
import logging

logger = logging.getLogger('Communication')

logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def write_weight(global_weights):
    file_name = "global_test.pth"
    # copy weights & Save weights
    torch.save(global_weights, file_name)

    data = {'file_name': file_name}

    files = {
        'json': ('json_data', json.dumps(data), 'application/json'),
        'model': (file_name, open(file_name, "rb"), 'application/octet-stream')}

    return files


def select_model(args):
    if args['dataset'] == 'mnist':
        global_model = CNNMnist(args=args)
    elif args['dataset'] == 'cifar':
        global_model = CNNCifar(args=args)

    return global_model.to('cuda')


def request_update(ip, id):
    url = "http://" + ip + ':8585/'
    send_url = url + "update"
    res = requests.post(send_url)
    file_names = 'client_model' + str(id) + ".pkl"

    # 파일로 온 weight 받기
    with open(file_names, 'wb') as f:
        for chunk in res.iter_content(chunk_size=128):
            f.write(chunk)


class FL_server:

    def __init__(self):
        self.args = args_parser()
        self.global_model = select_model(self.args)

    def send_training_info(self, ip, id):
        url = "http://" + ip + ':8585/'
        send_url = url + "send_info"
        logger.info('Send to client id {}'.format(id))
        res = requests.post(send_url, data=json.dumps(self.args))
        logger.info(res.text + str(id))
        return res.text

    def start(self, ip, id, opt):

        self.global_model.to('cuda')
        files = write_weight(self.global_model.state_dict())

        url = "http://" + ip + ':8585/'
        send_url = url + "send_model"

        logger.info("Send weight to client id {}".format(id))
        res = requests.post(send_url, files=files, data=opt)

        train_loss = res.text
        logger.info("client id {0} Loss {1}".format(id, train_loss))
        print(train_loss)
        return train_loss

    def aggregate(self):
        self.global_model.train()
        global_dict = self.global_model.state_dict()

        clients = []
        for i in range(self.args["num_users"]):
            file_name = 'client_model' + str(i) + ".pkl"
            clients.append(torch.load(file_name))

        for k in global_dict.keys():
            global_dict[k] = torch.stack([clients[i][k].float() for i in range(len(clients))],
                                         0).mean(0)

        self.global_model.load_state_dict(global_dict)


"""

class FL_client:
    def __init__(self):
        # self.url = ""
        self.args = args_parser()
        self.urls = ""
        self.id = ""
        self.param = {}
        self.global_model = None
        self.flag = True
        self.weight = ""
        self.train_loss = ""

    def initialize(self, id, url):
        print("ININININ")
        self.id = id
        self.args["id"] = id
        self.urls = "http://" + url + ':8585/'

        send_url = self.urls + "init"

        logger.info('Send to client id {}'.format(self.id))
        res = requests.post(send_url, data=json.dumps(self.args))
        print(res.text)
        return res.text

    def start(self, server_global_model):
        logger.info("Start training client id {}".format(self.id))
        # Global model --> client 에게 보내줄 모델
        # send_weight 를 통해 global model 보냄 --> receive_weight 를 통해 본인 weight 업데이트
        # 업데이트 된 weight --> mains.py 에서 aggregate
        self.global_model = server_global_model
        self.send_weight()
        self.receive_weight()
        return None

    def send_weight(self):

        self.global_model.to('cuda')
        self.weight = self.global_model.state_dict()
        files = write_weight(self.weight)


        print("Download")
        send_url = self.urls + "download"
        res = requests.post(send_url, files=files)
        print(res.text)
        logger.info("Send weight to client id {}".format(self.id))
        self.train_loss = res.text
        logger.info("client id {0} Loss {1}".format(self.id, self.train_loss))

        return "Fin"

    def receive_weight(self):
        send_url = self.urls + "update"
        res = requests.post(send_url)
        file_names = 'client_model' + str(self.id) + ".pkl"

        # 파일로 온 weight 받기
        with open(file_names, 'wb') as f:
            for chunk in res.iter_content(chunk_size=128):
                f.write(chunk)


def get_optimal(arg, param):
    # TODO: Get optimal
    # print(param['local_ep'])
    arg['local_ep'] = 10 * param['local_ep']

    return arg


def select_model(args):
    if args['dataset'] == 'mnist':
        global_model = CNNMnist(args=args)
    elif args['dataset'] == 'cifar':
        global_model = CNNCifar(args=args)

    return global_model.to('cuda')



"""
