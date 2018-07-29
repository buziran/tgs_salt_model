# -*- coding: utf-8 -*-

from __future__ import print_function

import os
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.metadata',
    'https://www.googleapis.com/auth/drive.appdata',
    'https://www.googleapis.com/auth/drive.apps.readonly'
]

class ServiceManager(object):

    def __init__(self, credentials):
        self.credentials = credentials
        store = file.Storage('token.json')
        self.creds = store.get()
        if not self.creds or self.creds.invalid:
            flow = client.flow_from_clientsecrets(os.path.expanduser(credentials), SCOPES)
            self.creds = tools.run_flow(flow, store)

    def get_drive_service(self):
        service = build('drive', 'v3', http=self.creds.authorize(Http()))
        return service

    def get_sheets_service(self):
        service = build('sheets', 'v4', http=self.creds.authorize(Http()))
        return service
