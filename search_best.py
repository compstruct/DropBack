import json
import argparse

def main(log: str, field: str, error: bool):
    highest = 0
    with open(log) as f:
        data = json.load(f)
    for d in data:
        df = float(d[field])
        if df > highest:
            highest = df
    print(1-highest if error else highest)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("log_file", help="The log file to find the field in")
    p.add_argument("json_field", help="The JSON field to find the best of")
    p.add_argument("-e", "--error", help="Report value as error from 1.", action='store_true', default=False)
    args = p.parse_args()
    main(args.log_file, args.json_field, args.error)