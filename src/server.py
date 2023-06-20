import argparse
import time
import torch
from btcomm.bt import BTServer
from literature_models.assine_2022b.decoder import get_decoder

def get_argparser():
    argparser = argparse.ArgumentParser(description='Device Side Full System')
    argparser.add_argument('-ideal', action="store_true", help='Experiment Name')
    argparser.add_argument('-fake', action="store_true", help='Experiment Name')
    return argparser


def main():
    # Load Model

    args = get_argparser().parse_args()
    if not args.ideal and not args.fake:
        device = "cuda"
        device = torch.device(device)
        model = get_decoder(device)
        model.to(device)
        input_shape = (1, 6, 96, 96)
        dummy_input = torch.randn(input_shape, dtype=torch.float).to(device)
        model(dummy_input)
    def callback():
        if args.ideal:
            print("Ideal Server")
        elif args.fake:
            time.sleep(0.075)
        else:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()

    # set up server
    bt_server = BTServer(callback=callback)
    bt_server.run()

if __name__ == "__main__":
    main()
