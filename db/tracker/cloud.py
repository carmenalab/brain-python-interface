"""Upload data to google cloud platfom via IoT core. This code is based
on an 'end-to-end' example provided by the GCP documentation"""
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'

import argparse
import datetime
import json
import ssl
import time
import requests

import jwt
import paho.mqtt.client as mqtt


def create_jwt(project_id, private_key_file, algorithm):
    """Create a JWT (https://jwt.io) to establish an MQTT connection.
    Function copied from GCP documentation"""
    token = {
        'iat': datetime.datetime.utcnow(),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=60),
        'aud': project_id
    }
    with open(private_key_file, 'r') as f:
        private_key = f.read()
    print('Creating JWT using {} from private key file {}'.format(
        algorithm, private_key_file))
    return jwt.encode(token, private_key, algorithm=algorithm)

def error_str(rc):
    """Convert a Paho error to a human readable string."""
    return '{}: {}'.format(rc, mqtt.error_string(rc))


class CloudConfig(object):
    def __init__(self, cloud_info_fname):
        cloud_info = json.load(open(cloud_info_fname))
        for key in cloud_info:
            setattr(self, key, cloud_info[key])

        self.connected = False

def make_client(args):
    # Create the MQTT client and connect to Cloud IoT.
    client = mqtt.Client(
        client_id='projects/{}/locations/{}/registries/{}/devices/{}'.format(
            args.project_id,
            args.cloud_region,
            args.registry_id,
            args.device_id))
    client.username_pw_set(
        username='unused',
        password=create_jwt(
            args.project_id,
            args.private_key_file,
            args.algorithm))
    client.tls_set(ca_certs=args.ca_certs, tls_version=ssl.PROTOCOL_TLSv1_2)

    def on_connect(unused_client, unused_userdata, unused_flags, rc):
        """Callback for when a device connects."""
        print('Connection Result:', error_str(rc))
        args.connected = True

    def on_publish(unused_client, unused_userdata, unused_mid):
        """Callback when the device receives a PUBACK from the MQTT bridge."""
        print('Published message acked.')

    def on_message(unused_client, unused_userdata, message):
        """Callback when the device receives a message on a subscription."""
        payload = str(message.payload.decode('utf-8'))
        data = json.loads(payload)

        if 'action' not in data:
            print('Received message \'{}\' on topic \'{}\' with Qos {}'.format(
                payload, message.topic, str(message.qos)))
            return

        if data['action'] == 'upload_file':
            print("Uploading file {}".format(data['filename']))
            headers = {'Content-type': 'application/octet-stream'}
            r = requests.put(data['url'], data=open(data['full_filename'], 'rb'), headers=headers)
        elif data['action'] == 'check_status':
            print("Checking status of file {}".format(data['filename']))
            import django
            django.setup()
            from . import models
            df = models.DataFile.objects.get(path=data['filename'])
            df.backup_status = data['status'] + ' on {}'.format(str(datetime.datetime.now()))
            # df.archived = data['status'] == 'present'
            df.save()
            print(df.archived, df.backup_status, data)

    client.on_connect = on_connect
    client.on_publish = on_publish
    client.on_message = on_message
    return client

default_cloud_info_fname = os.path.join(os.path.dirname(__file__), '../../config/cloud-config.json')
def upload_json(json_data, cloud_info_fname=default_cloud_info_fname):
    if not os.path.exists(cloud_info_fname):
        print("Cloud upload configuration not found, skipping!")
        return

    args = CloudConfig(cloud_info_fname)
    client = make_client(args)
    client.connect(args.mqtt_bridge_hostname, args.mqtt_bridge_port)
    client.loop_start()

    timeout = 5
    total_time = 0
    while not args.connected and total_time < timeout:
        time.sleep(1)
        total_time += 1

    if not args.connected:
        print('Data not uploaded! Could not connect to MQTT bridge.')

    # This is the topic that the device will publish data to
    mqtt_telemetry_topic = '/devices/{}/events'.format(args.device_id)
    payload = json.dumps(json_data)
    print('Publishing payload', payload)
    client.publish(mqtt_telemetry_topic, payload, qos=1)
    client.disconnect()
    client.loop_stop()
    args.connected = False


def send_message_and_wait(file_data, cloud_info_fname=default_cloud_info_fname):
    """Upload a file by posting a message, retreiving a secret URL, and initiating a resumable upload"""
    if not os.path.exists(cloud_info_fname):
        print("Cloud upload configuration not found, skipping!")
        return

    args = CloudConfig(cloud_info_fname)
    client = make_client(args)
    client.connect(args.mqtt_bridge_hostname, args.mqtt_bridge_port)

    # subscribe to commands topic
    client.subscribe("/devices/{}/commands/#".format(args.device_id), 0)

    client.loop_start()

    timeout = 5
    total_time = 0
    while not args.connected and total_time < timeout:
        time.sleep(1)
        total_time += 1

    if not args.connected:
        print('Data not uploaded! Could not connect to MQTT bridge.')

    # This is the topic that the device will publish data to
    mqtt_telemetry_topic = '/devices/{}/events'.format(args.device_id)
    payload = json.dumps(file_data)
    print('Publishing payload', payload)
    client.publish(mqtt_telemetry_topic, payload, qos=1)

    # wait for signed URL reply
    time.sleep(10)

    client.disconnect()
    client.loop_stop()
    args.connected = False
