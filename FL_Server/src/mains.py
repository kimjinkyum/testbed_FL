# from kubernetes import client, config
import requests
from optimal import get_optimal_value
from options import *
from FL_server import *
# from FL_client import FL_client
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
    logger.info("Listing pods with their IPs")
    """
    # 쿠버네티스 API
    config.load_kube_config()
    v1 = client.CoreV1Api()
    
    # 쿠버네티스에서 마스터와 연동되어 있는 모든 Pod 가져오기
    ret = v1.list_pod_for_all_namespaces(watch=False)
    ip_lists = []
    application_name = "torch-example"

    for i in ret.items:

        # 그중에서도 Pod 배포 할 때 지정한 Name 과 동일한 pod IP 리스트 업
        if application_name in i.metadata.name:
            ip_lists.append(i.status.pod_ip)
            logger.info("Pod IP {}".format(i.status.pod_ip))
    """
    ip_lists = ["192.9.200.161"]
    logger.info(ip_lists)

    ret = None
    v1 = None

    fl_server = FL_server()
    num_users = fl_server.args["num_users"]

    # TODO: Control 할 Optimal Value 변경
    fl_server.args["opt_name"] = ["f_n", "v_n"]

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

    for r in range(fl_server.args["epochs"]):
        opts = get_optimal_value(num_users, fl_server.args["opt_name"])
        for i in range(num_users):
            # TODO : 각 optimal 함수에 따라서 변경 적용
            # Optimal resource 전송 및 model 전송 후 Client -> 바로 training
            jobs.append(pool.submit(fl_server.start(ip_lists[i], i, opts[i])))
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
