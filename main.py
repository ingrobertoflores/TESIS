
import argparse

import web
from network import train, test, infer_single_image
from web import app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", help="checkpoint",
                        type=str, default=None)
    parser.add_argument("-m", "--mode", help="train or test",
                        type=str, default="train")
    parser.add_argument("-r", "--runtime", help="train with runtime-generated images",
                        action='store_true')
    parser.add_argument("--img", help="image fullpath to test",
                        type=str, default=None)
    parser.add_argument("--port", help="Port to serve application",
                        type=str, default=5000)
    parser.add_argument("--host", help="Port to serve application",
                        type=str, default="localhost")

    args = parser.parse_args()

    if args.mode == 'train':
        train(checkpoint=args.ckpt, runtime_generate=args.runtime)
    elif args.mode == 'test':
        if args.img is None:
            test(checkpoint=args.ckpt)
        else:
            infer_single_image(checkpoint=args.ckpt, fname=args.img)
    elif args.mode == 'serve':
        web.init_lprnet(args.ckpt)
        app.run(host=args.host, port=args.port)
    else:
        print('unknown mode:', args.mode)
