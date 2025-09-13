# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" Define the Logger class to print log"""
import os
import sys
import logging
from datetime import datetime
from os.path import join
import inspect
import os


log = None

def init(config):
    global log
    if log is None:
        log = Logger(config)


class Logger:
    def __init__(self, args, time = None):
        if time is None:
            time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.time = time
        ins_stack = inspect.stack()
        p = (ins_stack[2])[1]
        abs_path = os.path.abspath(p)
        f_name = os.path.basename(p)
        self.log_dir = os.path.dirname(abs_path)
        self.log_dir = self.log_dir.replace("RunScriptsAndConfigs",os.path.join("Data","Output"))
        self.log_dir = os.path.join(self.log_dir,f_name[:-3])
        self.log_dir = os.path.join(self.log_dir,time)

        #creating log directory
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        logger = logging.getLogger(self.log_dir)
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(os.path.join(self.log_dir,('log.txt')))
            fh.setLevel(logging.INFO)
            ch = ProgressHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(ch)
        self.logger = logger
        self.writer = None

    def setup(self, args):
        if args:
            # setup TensorBoard
            if args.tensorboard:
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir)

            if args.log_per_updates:
                self.log_per_updates = args.log_per_updates
            else:
                self.log_per_updates = 50
            self.epoch = 0

    def set_progress(self, epoch, total):
        self.logger.info(f'Epoch: {epoch}')
        self.epoch = epoch
        self.i = 0
        self.total = total
        self.start = datetime.now()

    def update(self, stats):
        self.i += 1
        if self.i % self.log_per_updates == 0:
            remaining = str((datetime.now() - self.start) / self.i * (self.total - self.i))
            remaining = remaining.split('.')[0]
            updates = stats.pop('updates')
            stats_str = ' '.join(f'{key}[{val:.8f}]' for key, val in stats.items())
            self.logger.info(f'> epoch [{self.epoch}] updates[{updates}] {stats_str} eta[{remaining}]')
            if self.writer:
                for key, val in stats.items():
                    self.writer.add_scalar(f'train/{key}', val, updates)
        if self.i == self.total:
            self.logger.debug('\n')
            self.logger.debug(f'elapsed time: {str(datetime.now() - self.start).split(".")[0]}')

    def log_eval(self, stats, metrics_group=None):
        stats_str = ' '.join(f'{key}: {val:.8f}' for key, val in stats.items())
        self.logger.info(f'eval {stats_str}')
        if self.writer:
            for key, val in stats.items():
                self.writer.add_scalar(f'eval/{key}', val, self.epoch)
        # for mode, metrics in metrics_group.items():
        #     self.log.info(f'evaluation scores ({mode}):')
        #     for key, (val, _) in metrics.items():
        #         self.log.info(f'\t{key} {val:.4f}')
        # if self.writer and metrics_group is not None:
        #     for key, val in stats.items():
        #         self.writer.add_scalar(f'valid/{key}', val, self.epoch)
        #     for key in list(metrics_group.values())[0]:
        #         group = {}
        #         for mode, metrics in metrics_group.items():
        #             group[mode] = metrics[key][0]
        #         self.writer.add_scalars(f'valid/{key}', group, self.epoch)

    def __call__(self, msg):
        self.logger.info(msg)


class ProgressHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        log_entry = self.format(record)
        if record.message.startswith('> '):
            sys.stdout.write('{}\r'.format(log_entry.rstrip()))
            sys.stdout.flush()
        else:
            sys.stdout.write('{}\n'.format(log_entry))


