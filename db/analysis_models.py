#!/usr/bin/python
import tracker.models as models
import json

class TaskEntry():
    def __init__(self, record):
        self.record = record
        self.params = json.loads(record.params)
        try:
            self.assist_level = self.params['assist_level']
        except:
            self.assist_level = None
