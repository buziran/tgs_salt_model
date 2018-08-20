#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime

from .upload import result_upload

def main(argv=None):
    result_upload(
        name="test", datetime_str=str(datetime.datetime.now()), path="./output", summary="this is upload test")


if __name__ == '__main__':
    main()
