#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import shlex
import subprocess
import datetime
import yaml

from gcp.upload import result_upload
from git import Repo
from absl import app, flags
from skopt import gp_minimize, dummy_minimize
from skopt.space import Categorical, Integer, Real

flags.DEFINE_string("yaml", "", """path to yaml file.""", short_name='y')
flags.DEFINE_bool("dry-run", False, """Do not upload anything""", short_name="n")
flags.DEFINE_bool("force", False, """Ignore un-committed files""", short_name="f")
flags.DEFINE_bool("verbose", True, """whether to print job commands""")
flags.DEFINE_bool("maximize", True, """whether to maximize the score""")
flags.DEFINE_integer("calls", 100, """Number of calls to func""")

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


class Job(object):
    def __init__(self, name_templ, train_templ, eval_templ, params, preprocess="", upload=True, verbose=True):
        self.name_templ = name_templ
        self.train_templ = train_templ
        self.eval_templ = eval_templ
        self.params = params
        self.preprocess_args = preprocess
        self.upload = upload
        self.verbose = verbose

    def __call__(self, param_vals):
        name = self.set_param_val(self.name_templ, param_vals)
        train_args = self.set_param_val(self.train_templ, param_vals)
        eval_args = self.set_param_val(self.eval_templ, param_vals)

        if self.verbose:
            print("name: {}".format(name))
            print("train command: {}".format(train_args))
            print("eval command: {}".format(eval_args))

        if self.preprocess_args != "":
            subprocess.run(shlex.split(self.preprocess_args), check=False)

        if train_args != "":
            proc_tb = subprocess.Popen(["tensorboard", "--logdir", "../output", "--port", "6699"])
            subprocess.run(shlex.split(train_args), check=True)
            proc_tb.kill()

        summary = ""
        if eval_args != "":
            proc = subprocess.Popen(eval_args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

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
            score = float(summary.split(',')[0].split(':')[1])

        if self.upload:
            result_upload(name, OUTPUT_PATH, summary, command=train_args + " " + eval_args)

        return score

    def set_param_val(self, template, param_vals):
        args = template
        for param, val in zip(self.params, param_vals):
            if param in args:
                args = args.replace(param, str(val))
        return args


def parse_parameters(parameter_dict):
    list_param = []
    list_space = []
    def _parse(space_def):
        if space_def[0] == "categorical":
            return Categorical(space_def[1:])
        elif space_def[0] == "integer":
            return Integer(space_def[1], space_def[2])
        elif space_def[0] == "real":
            if len(space_def) == 4:
                prior = space_def[3]
            else:
                prior = 'uniform'
            return Real(space_def[1], space_def[2], prior)
    for param, space_def in parameter_dict.items():
        list_param.append(param)
        list_space.append(_parse(space_def))
    return list_param, list_space


def parse_yaml(path_yaml):
    with open(path_yaml) as f:
        yaml_dict = yaml.load(f)

    if "preprocess" in yaml_dict:
        preprocess = yaml_dict["preprocess"]
    else:
        preprocess = ""
    template = yaml_dict["template"]
    name_templ = template["name"]
    train_templ = template["train"]
    eval_templ = template["eval"]

    params_dict = yaml_dict["parameter"]

    params, spaces = parse_parameters(params_dict)

    if "minimizer" in yaml_dict:
        minimizer = yaml_dict["minimizer"]
    else:
        minimizer = "gp_minimize"

    return name_templ, train_templ, eval_templ, params, spaces, preprocess, minimizer


def main(argv):
    if not FLAGS.force:
        check_commit()

    name_templ, train_templ, eval_templ, params, spaces, preprocess, minimizer = parse_yaml(FLAGS.yaml)

    if FLAGS.verbose:
        for param, space in zip(params, spaces):
            print("{}: {}".format(param, space))

    upload = not FLAGS['dry-run'].value
    job = Job(name_templ, train_templ, eval_templ, params, preprocess, upload=upload, verbose=FLAGS.verbose)

    if not FLAGS.maximize:
        func = job
    else:
        func = lambda x: -1 * job(x)

    if minimizer == "gp_minimize":
        res = gp_minimize(func, spaces, n_calls=FLAGS.calls)
    elif minimizer == "dummy_minimize":
        res = dummy_minimize(func, spaces, n_calls=FLAGS.calls)
    else:
        raise ValueError("minimizer {} is invalid".format(minimizer))

    print(res.x, res.fun)


if __name__ == '__main__':
    app.run(main)
