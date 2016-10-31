import argparse


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Process optional settings.'
        )
        self.add_argument()
        self.parse_argument()

    def add_argument(self):
        self.parser.add_argument(
            '--epoch',
            action='store',
            dest='epoch',
            default=50,
            type=int,
            help='Input epoch number'
        )
        self.parser.add_argument(
            '--gpuid',
            action='store',
            dest='gpuid',
            default=-1,
            type=int,
            help='Input GPUID you\'d like to use'
        )
        self.parser.add_argument(
            '--batchsize',
            action='store',
            dest='batchsize',
            default=100,
            type=int,
            help='Input minibatch size'
        )
        self.parser.add_argument(
            '--dims',
            action='store',
            dest='dims',
            default=1,
            type=int,
            help='Input dimention of encoded vector'
        )

    def parse_argument(self):
        self.args = vars(self.parser.parse_args())

    def get_argument(self, arg):
        try:
            return self.args[arg]
        except:
            return -1
