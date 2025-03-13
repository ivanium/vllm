#!/bin/bash

echo $(($(($1+2)) * 1024 * 1024 * 1024)) > /tmp/kvcached_mem
python online-app.py $1
