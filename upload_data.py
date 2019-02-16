"""
Uses drive linux/ anythgin to upload results to a specified dir
"""

import argparse
import subprocess

CMD = '/usr/bin/rsync -r'


p = argparse.ArgumentParser()
p.add_argument('-d', '--dir', help='The upload directory')
p.add_argument('data', nargs='+', help='The data directorys')
p.add_argument('-i', '--iteration', help='The iteration to count to', default='1000')
p.add_argument('-u', '--upload_cmd', default=CMD)

args = p.parse_args()
cmd_to_run = args.upload_cmd.split(' ') + args.data + args.dir
subprocess.run(cmd_to_run)
