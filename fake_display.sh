#!/bin/bash

set -x
set -e

sudo apt-get install -y xvfb

xvfb-run -s "-screen 0 1400x900x24" bash
