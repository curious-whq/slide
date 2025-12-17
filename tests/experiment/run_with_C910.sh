#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"



HOSTNAME=$1
PORT=$2
USERNAME=$3
PASSWORD=$4
HOSTPATH=$5


if [ $# -ne 5 ]; then
    echo "Usage: $0 <hostname> <port> <username> <password> <hostpath>"
    exit 1
fi

python3 exp1_inject_test_with_C910.py \
    --hostname "$HOSTNAME" \
    --port "$PORT" \
    --username "$USERNAME" \
    --password "$PASSWORD" \
    --hostpath "$HOSTPATH"
