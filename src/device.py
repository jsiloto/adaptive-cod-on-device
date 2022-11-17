import argparse
import os

def get_argparser():
    argparser = argparse.ArgumentParser(description='Device Side Full System')
    argparser.add_argument('--name', type=str, default="Test", help='Experiment Name')
    argparser.add_argument('--seconds', type=int, default=10, help='Total Runtime')
    argparser.add_argument('--mode', type=str, default=10, help='Mode of operation (11 12 .. 44) or dynamic')
    #argparser.add_argument('-multithread', type=str, default=10, help='separate compute/communicate threads')
    return argparser


# Load Model
# Connect BT

# Start Loop
    # COmpute
    # Send Data
