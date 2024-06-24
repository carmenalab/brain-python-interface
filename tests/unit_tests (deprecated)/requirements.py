# Software requirements, in code form
from reqlib import Req

all_requirements = []

req_source = Req("DataSource instances shall interact with specific streaming data generators and yield data using multiprocessing", requirements=all_requirements)
req_kf = Req("Kalman Filter (KF) algorithm shall be implemented correctly", requirements=all_requirements)
req_kf_mle = Req("Kalman Filter (KF) training by maximum likelihood estimation shall match hand-verifiable reference outputs", requirements=all_requirements)