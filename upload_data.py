"""
Uses drive linux/ anythgin to upload results to a specified dir
"""

import argparse
import os
import subprocess

CMD = '/usr/bin/rsync -p'
TARGETS = ['l2_{}.npz', 'param_hist_{}.npz']

p = argparse.ArgumentParser()
p.add_argument('-d', '--dir', help='The upload directory')
p.add_argument('data', nargs='+', help='The data directorys')
p.add_argument('-i', '--iteration', help='The iteration to count to', default=1000, type=int)
p.add_argument('-u', '--upload_cmd', default=CMD)

args = p.parse_args()

transfers = []
for d in args.data:
    for i in range(0, args.iteration, 100):
        for t in TARGETS:
            src = os.path.join(d, t.format(i))
            dest = args.dir + t.format(i)
            transfers.append([src, dest])
for t in transfers:
    cmd_to_run = args.upload_cmd.split(' ') + t
    subprocess.run(cmd_to_run)
