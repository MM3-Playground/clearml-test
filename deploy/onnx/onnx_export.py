import argparse
import os
import sys

import torch.onnx 

from models.A import *

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('--load_path', type=str, help='path to the pretrained model')
    parser.add_argument("--image_size", type=int, default=512, help="size of the images for prediction")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)

    onnx_file_name = "model.onnx"

    model = Attributor(args.image_size)

    if os.path.exists(args.load_path):
        checkpoint = torch.load(args.load_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        print("load %s finish" % (os.path.basename(args.load_path)))
    else:
        print("%s not exist" % args.load_path)
        sys.exit()

    model.eval()

    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, args.image_size, args.image_size, requires_grad=True) 
    
    torch.onnx.export(model,
                        dummy_input,
                        onnx_file_name,
                        export_params=True,
                        do_constant_folding=True,
                        input_names = ['input'],
                        output_names = ['output'],
                        dynamic_axes={'input' : {0 : 'batch_size'},    
                                        'output' : {0 : 'batch_size'},
                                        })