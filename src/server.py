import torch

from btcomm.bt import BTServer
from literature_models.assine_2022b.decoder import get_decoder

def main():
    # Load Model
    device = "cuda"
    device = torch.device(device)
    model = get_decoder()
    model.to(device)
    input_shape = (6, 96, 96)
    dummy_input = torch.randn(input_shape, dtype=torch.float).to(device)
    def callback():
        model(dummy_input)

    # set up server
    bt_server = BTServer(callback=callback)
    bt_server.run()

if __name__ == "__main__":
    main()