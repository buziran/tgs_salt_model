#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shlex
import subprocess
from gcp.upload import result_upload

import sys
from absl import app, flags

flags.DEFINE_string("train", None, """train command.""")
flags.DEFINE_string("eval", None, """evaluation command.""")
flags.DEFINE_string("name", "no-name", """name of job""")
flags.DEFINE_bool("dry-run", False, """Do not upload anything""", short_name="n")

FLAGS = flags.FLAGS

OUTPUT_PATH = "./output"


def main(argv):

    if FLAGS.train is not None:
        subprocess.run(shlex.split(FLAGS.train), check=True)

    summary = ""
    if FLAGS.eval is not None:
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
