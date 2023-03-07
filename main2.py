#!/usr/bin/env python
# coding: utf-8
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task')
parser.add_argument('--gpu')
args = parser.parse_args()
print(args)