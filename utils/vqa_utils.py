#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/3/13
"""


def unify_size(h, w, ms):
    """
    统一最长边的尺寸
    :h 高
    :w 宽
    :ms 最长尺寸
    """
    # 最长边修改为标准尺寸
    if w > h:
        r = ms / w
    else:
        r = ms / h
    h = int(h * r)
    w = int(w * r)

    return h, w
