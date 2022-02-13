from kubernetes import client, config
import requests

from options import *
from FL_server import *
from FL_client import FL_client
from concurrent.futures import ThreadPoolExecutor
from utils import *

import logging

if __name__ == "__main__":

    # 로그 생성
    logger = logging.getLogger("Server")
    logger.setLevel(logging.INFO)

    # log 출력 형식
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    config.load_kube_config()
    v1 = client.CoreV1Api()

    print("AAAListing pods with their IPs")

    # 쿠버네티스에서 마스터와 연동되어 있는 모든 Pod 가져오기
    ret = v1.list_pod_for_all_namespaces(watch=False)
    ip_lists = []
    application_name = "torch-example"

    for i in ret.items:

        # 그중에서도 Pod 배포 할 때 지정한 Name 과 동일한 pod IP 리스트 업
        if application_name in i.metadata.name:
            ip_lists.append(i.status.pod_ip)
            logger.info("Pod IP {}".format(i.status.pod_ip))

    print(ip_lists)

    ret = None
    v1 = None

    fl_server = FL_server()
    num_users = fl_server.args["num_users"]
    # Synchronize 통신
    pool = ThreadPoolExecutor()

    jobs = []

    accs = []
    losses = []
    train_losses = []
    _, test_dataset = get_dataset(fl_server.args["dataset"])

    # Send metadata & model
    # Jobs -> Wait_finish : 동시다발적으로 url 전송 그 후 기다림

    for i in range(num_users):
        jobs.append(pool.submit(fl_server.send_training_info, ip_lists[i], i))
        if wait_finish(jobs) == "Finish":
            jobs = []
            pass

    # Server : Send_model
    # Client : Start training
    # Server : Request update
    opt = {'test': "1"}
    for r in range(fl_server.args["epochs"]):
        for i in range(num_users):

            jobs.append(pool.submit(fl_server.start(ip_lists[i], i, opt)))
            if wait_finish(jobs) == "Finish":
                jobs = []

            jobs.append(pool.submit(request_update(ip_lists[i], i)))
            if wait_finish(jobs) == "Finish":
                jobs = []

        # 모델 Aggregate
        fl_server.aggregate()
        logger.info("Round {}".format(r))
        acc, loss = test_inference(fl_server.global_model, test_dataset)

        print(acc, loss)

"""
# Initialize
for i in range(num_users):
    print(i)
    fl_clients[i].initialize(i, ip_lists[i])
    # jobs.append(pool.submit(fl_clients[i].initialize, i, ip_lists[i]))

# 모든 Client 에게 응답 받기 전까지 기다림
if wait_finish(jobs) == "Finish":
    pass

# Training (start로 트레이닝 시작, aggregate 로 모델 업데이트)
for r in range(fl_server.args["epochs"]):
    jobs = []
    for i in range(num_users):
        jobs.append(pool.submit(fl_clients[i].start, fl_server.global_model))
        # jobs.append(pool.submit(fl_clients[i].send_weight))



    if wait_finish(jobs) == "Finish":
        fl_server.global_model = aggregate(fl_server.global_model, num_users)

    temp = [float(fl_clients[i].train_loss) for i in range(num_users)]
    train_losses.append(sum(temp) / len(temp))

    acc, loss = test_inference(fl_server.global_model, test_dataset)

    accs.append(acc)
    losses.append(loss)

    print("Accuracy after {}".format(r), acc)


write_text_file(accs, losses, train_losses)

"""
