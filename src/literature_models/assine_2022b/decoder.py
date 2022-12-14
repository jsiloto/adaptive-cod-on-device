import os
import sys
import hashlib
import gdown
import torch

def sha1_checksum(filepath):
    # BUF_SIZE is totally arbitrary, change for your app!
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    md5 = hashlib.md5()
    sha1 = hashlib.sha1()

    with open(filepath, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
            sha1.update(data)

    return sha1.hexdigest()


def get_decoder(map_location=torch.device('gpu')):
    model_filepath = "./models/effd2_decoder.pt"
    sha1sum = "12e0c79f444b78dd38f5ba2bff4cd5062f0b4ccb"
    if not os.path.isfile(model_filepath):
        gdown.download('https://drive.google.com/uc?id=1t4kPsQxV_kgmSL-mXWq4kW5KbG1Sd-U8&export=download',
                       output=model_filepath)

    if sha1_checksum(model_filepath) != sha1sum:
        os.remove(model_filepath)
        print("Download Failed")
        exit(1)

    model = torch.jit.load(model_filepath, map_location=map_location)
    return model
