#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import shlex
import subprocess
import datetime

from gcp.upload import result_upload
from git import Repo
from absl import app, flags

flags.DEFINE_string("preprocess", "rm -rf ../output", """preprocess command.""")
flags.DEFINE_string("train", "", """train command.""")
flags.DEFINE_string("eval", "", """evaluation command.""")
flags.DEFINE_string("name", "no-name", """name of job""")
flags.DEFINE_bool("dry-run", False, """Do not upload anything""", short_name="n")
flags.DEFINE_bool("force", False, """Ignore un-committed files""", short_name="f")

FLAGS = flags.FLAGS

OUTPUT_PATH = "../output"


def check_commit():
    repo = Repo()
    not_committed_files = repo.index.diff(None)
    untracked_files = repo.untracked_files

    if len(untracked_files) != 0 or len(list(not_committed_files)) != 0:
        print("There are some untracked file or not-commited file in repository")
        subprocess.run(["git", "status"])
        assert yn("Are you sure you want to run the job?")


def yn(message):
    while True:
        answer = input(message + ' [y/N]: ')
        if len(answer) > 0 and answer[0].lower() in ('y', 'n'):
            return answer[0].lower() == 'y'


def main(argv):

    if not FLAGS.force:
        check_commit()

    datetime_str = str(datetime.datetime.now())

    if FLAGS.preprocess != "":
        subprocess.run(shlex.split(FLAGS.preprocess), check=False)

    if FLAGS.train != "":
        proc_tb = subprocess.Popen(["tensorboard", "--logdir", "../output", "--port", "6699"])
        subprocess.run(shlex.split(FLAGS.train), check=True)
        proc_tb.kill()

    summary = ""
    if FLAGS.eval != "":
        proc = subprocess.Popen(FLAGS.eval, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        last_line = ""
        while True:
            # バッファから1行読み込む.
            line = proc.stdout.readline()
            # バッファが空 + プロセス終了.
            if not line and proc.poll() is not None:
                break
            if line != b"":
                last_line = line.decode('utf-8')
                sys.stdout.write(line.decode('utf-8'))

        summary = last_line.replace('\n', '').replace('\r', '')
        print("summary: \"{}\"".format(summary))


    if not FLAGS["dry-run"].value:
        result_upload(FLAGS.name, OUTPUT_PATH, summary)


if __name__ == '__main__':
    app.run(main)
