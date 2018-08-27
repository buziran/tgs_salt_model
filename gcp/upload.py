#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pprint import pprint
import json

import datetime
from absl import app, flags
import git
from googleapiclient.http import MediaFileUpload

from .service_manager import ServiceManager

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'credentials', os.path.join(os.path.dirname(__file__), 'credentials.json'), 'path to GCP token file.')

flags.DEFINE_string(
    'config', os.path.join(os.path.dirname(__file__), 'config.json'), 'path to GCP token file.')


def result_upload(name, path, summary, datetime_str=None):
    if datetime_str is None:
        datetime_str = str(datetime.datetime.now())
    service_manager = ServiceManager(FLAGS.credentials)
    config = get_config()
    service = service_manager.get_drive_service()
    drive_path = upload_outputs(service, name, path, config)

    job_info = create_job_info(name, datetime_str, summary, drive_path)
    service = service_manager.get_sheets_service()
    append_row(service, job_info, config)


def get_config():
    with open(FLAGS.config) as f:
        config = json.load(f)
    return config


def create_job_info(name, datetime_str, summary, drive_path):
    command = " ".join(sys.argv)
    repo = git.Repo(os.path.dirname(__file__), search_parent_directories=True)
    commit = repo.head.object.hexsha
    job_info = [name, datetime_str, commit, command, drive_path, summary]
    return job_info


def append_row(service, job_info, config):
    range_name = '{}!A:A'.format(config['sheet_name'])
    result = service.spreadsheets().values().get(spreadsheetId=config['spreadsheet_id'],
                                                 range=range_name).execute()
    values = result.get('values', [])

    next_row = len(values) + 1

    target_range = '{}!A{}'.format(config['sheet_name'], next_row)

    body = {"values": [job_info]}

    value_input_option = 'USER_ENTERED'
    insert_data_option = 'INSERT_ROWS'
    request = service.spreadsheets().values().append(spreadsheetId=config['spreadsheet_id'], range=target_range,
                                                     valueInputOption=value_input_option,
                                                     insertDataOption=insert_data_option, body=body)
    response = request.execute()
    pprint(response)


def upload_outputs(service, name, output_dir, config):
    def create_dir(name, parent_id=None):
        file_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id is not None:
            file_metadata['parents'] = [parent_id]

        file = service.files().create(body=file_metadata, fields='id').execute()
        return file.get('id')

    def upload_file(name, source_path, parent_id=None):
        file_metadata = {'name': name}
        if parent_id is not None:
            file_metadata['parents'] = [parent_id]

        media = MediaFileUpload(source_path)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')

    object_ids = {}

    id = create_dir(name, config['drive_root_id'])
    object_ids[output_dir] = id
    drive_path = config['drive_url_prefix'] + id

    for root, dirs, files in os.walk(output_dir):
        for d in dirs:
            id = create_dir(d, object_ids[root])
            object_ids[os.path.join(root, d)] = id

        for f in files:
            source_path = os.path.join(root, f)
            id = upload_file(f, source_path, object_ids[root])
            object_ids[source_path] = id

    return drive_path


