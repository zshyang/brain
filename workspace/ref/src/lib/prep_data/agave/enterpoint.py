import argparse
import os


def call_process(process, index, interval):
    start = interval * index
    end = interval * (index + 1)
    for i in range(start, end):
        cmd = f"python {process} --index {i}"
        os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description='Decimated mesh processor')
    parser.add_argument('--process', type=str, help='the name of the process')
    parser.add_argument('--index', type=int, help='The start index')
    parser.add_argument('--interval', type=int, help='The interval')
    args = parser.parse_args()

    call_process(args.process, args.index, args.interval)


if __name__ == '__main__':
    main()
