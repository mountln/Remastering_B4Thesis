import argparse
import os
import random
import time

import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import rgb_to_grayscale, rotate

from dataloaders import DALIVideoLoader, DALISequenceLoader
import utils
from utils import print_log as print
# from model.remasternet import NetworkR
from model.remasternet5 import NetworkR


def get_perceptual_loss_fn(layers, local_rank):
    """Return a function which compute the perceptual loss for sequences."""
    perceptual_loss_batch_frames = utils.PerceptualLoss(
        layers=layers).to(local_rank)

    def perceptual_loss(outputs, targets):
        """Perceptual loss for sequences. Input shape: (B, C, T, H, W)."""
        # Change shape from (B, C, T, H, W) to (B * T, C, H, W)
        _, c, _, h, w = outputs.size()
        outputs = outputs.transpose(1, 2).reshape(-1, c, h, w)
        targets = targets.transpose(1, 2).reshape(-1, c, h, w)

        return perceptual_loss_batch_frames(outputs, targets)

    return perceptual_loss


def get_preprocess_fn(args, transform_pair=None, transform_input=None):
    """Return a function for data pre-processing."""
    if args.remaster:
        def preprocess(input_sequences):
            # Permute: from (B, T, H, W, C) to (B, T, C, H, W)
            input_sequences = rgb_to_grayscale(input_sequences.permute(0, 1, 4, 2, 3))
            # Permute: from (B, T, C, H, W) to (B, C, T, H, W)
            input_sequences = input_sequences.permute(0, 2, 1, 3, 4)
            return input_sequences.div(255.).clamp(0., 1.)
        return preprocess

    noise_img_dir = os.path.join(args.data_root, 'noise_data')
    noise_img_loader = DALISequenceLoader(
        image_dir=noise_img_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        size=args.size,
        local_rank=args.local_rank,
        auto_reset=True,
        random_seed=12 + args.rank
    )
    noise_img_iterator = iter(noise_img_loader)

    def preprocess(target_sequences, noise_k=(0.5, 1.2)):
        # Read noise and target sequences
        # Permute: (B, T, H, W, C) to (B, T, C, H, W)
        noise_sequences = next(noise_img_iterator)[0]['sequence'].to(
            device=args.local_rank, dtype=torch.float).permute(0, 1, 4, 2, 3)
        target_sequences = target_sequences.permute(0, 1, 4, 2, 3)

        # Rotate noise sequences
        if random.random() < 0.8:
            random_angle = random.uniform(-5, 5)
            b, t, c, h, w = noise_sequences.size()
            noise_sequences = rotate(noise_sequences.view(-1, c, h, w),
                                     random_angle).view(b, t, c, h, w)

        # Composite input sequences
        k = random.uniform(*noise_k) * (1 if random.random() < 0.5 else -1)
        input_sequences = ((target_sequences + noise_sequences * k) / 255.).clamp(0., 1.)
        target_sequences.div_(255.).clamp_(0., 1.)

        # Data augmentation
        inputs, targets = [], []
        for input, target in zip(input_sequences, target_sequences):
            # torch.cat: Ensure input and target have the same transform state
            #            (same random value for data augmentation).
            pair = transform_pair(torch.cat([input, target]))
            input, target = torch.chunk(pair, 2)
            input = transform_input(input)
            inputs.append(input)
            targets.append(target)
        input_sequences = rgb_to_grayscale(torch.stack(inputs))
        target_sequences = rgb_to_grayscale(torch.stack(targets))
        # Permute: from (B, T, C, H, W) to (B, C, T, H, W)
        input_sequences = input_sequences.permute(0, 2, 1, 3, 4)
        target_sequences = target_sequences.permute(0, 2, 1, 3, 4)

        return input_sequences, target_sequences

    return preprocess


def train(model, criterion, optimizer, train_loader, preprocess_fn, args):
    if args.rank == 0:
        log_writer = SummaryWriter()
        prev_time = time.monotonic()
        avg_loss = 0
        best_val_loss = float('inf')
        print("=> start training")

    model.train()
    for i, data in enumerate(train_loader):
        inputs, targets = preprocess_fn(data[0]['sequence'])

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Log
        reduced_loss = loss.detach().clone()
        dist.reduce(reduced_loss, dst=0)
        if args.rank == 0:
            reduced_loss = reduced_loss.item() / args.world_size
            log_writer.add_scalar('Loss/Train', reduced_loss, i)
            lr = optimizer.param_groups[0]['lr']
            log_writer.add_scalar('lr', lr, i)
            avg_loss += reduced_loss
            if i % args.print_freq == 0 and i != 0:
                curr_time = time.monotonic()
                avg_time = (curr_time - prev_time) / args.print_freq
                avg_loss = avg_loss / args.print_freq
                print(f"iter{i}: avg_time={avg_time} avg_loss={avg_loss}")
                prev_time = curr_time
                avg_loss = 0
            if i % args.save_img_freq == 0:
                # Save images
                h, w = inputs.size()[-2:]
                utils.save_images(inputs.transpose(1, 2).view(-1, 1, h, w),
                                  args.img_dir, filename_prefix='input')
                utils.save_images(outputs.transpose(1, 2).view(-1, 1, h, w),
                                  args.img_dir, filename_prefix='output')
                utils.save_images(targets.transpose(1, 2).view(-1, 1, h, w),
                                  args.img_dir, filename_prefix='target')

            if i % args.save_checkpoint_freq == 0 and i != 0:
                # Save checkpoint
                state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                utils.save_checkpoint(state, path=args.checkpoint_path)

        if i % args.eval_freq == 0 and i != 0:
            if args.rank == 0:
                state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                checkpoint_path = os.path.join("checkpoints", "iter{}.tar".format(i))
                utils.save_checkpoint(state, path=checkpoint_path)
            # Validation
            val_loss = validate(model, torch.nn.L1Loss(), preprocess_fn, args)
            if args.rank == 0:
                log_writer.add_scalar('Loss/Val', val_loss, i)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    state = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }
                    utils.save_checkpoint(state, args.checkpoint_path, is_best=True)

            # Reset the module to training mode
            model.train()

        if args.dry_run and i == args.print_freq:
            break

    if args.rank == 0:
        log_writer.close()


def remaster(model, preprocess_fn, args):
    # Get video loader
    video_dir = os.path.join(args.data_root, 'remaster')
    video_loader = DALIVideoLoader(
        video_root=video_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        size=args.size,
        rank=args.rank,
        local_rank=args.local_rank,
        world_size=args.world_size,
        reader_name='RemasterReader',
        random_shuffle=False,
        random_seed=12 + args.rank
    )

    input_frame_dir = os.path.join(args.output_dir, 'inputs')
    output_frame_dir = os.path.join(args.output_dir, 'outputs')
    diff_frame_dir = os.path.join(args.output_dir, 'diff')
    if args.rank == 0:
        print("=> start remastering")
        os.makedirs(input_frame_dir)
        os.makedirs(output_frame_dir)
        os.makedirs(diff_frame_dir)
        print("=> save frames to", os.path.join(args.output_dir))

    model.eval()
    for i, data in enumerate(video_loader):
        sequences = data[0]['sequence']

        with torch.no_grad():
            inputs = preprocess_fn(sequences)
            outputs = model(inputs)
            _, c, _, h, w = inputs.size()
            # (B, C, T, H, W) to (B * T, C, H, W)
            inputs = inputs.transpose(1, 2).reshape(-1, c, h, w)
            outputs = outputs.transpose(1, 2).reshape(-1, c, h, w)
            diff = torch.abs((inputs - outputs) * 5).clamp(0., 1.)
        filename_prefix = str(args.rank) + "_" + str(i)
        utils.save_frames(inputs, input_frame_dir, filename_prefix)
        utils.save_frames(outputs, output_frame_dir, filename_prefix)
        utils.save_frames(diff, diff_frame_dir, filename_prefix)
        if args.rank == 0 and i % args.print_freq == 0:
            print("... iter", i)


def validate(model, criterion, preprocess_fn, args):
    # Get validation loader
    val_dir = os.path.join(args.data_root, 'val')
    val_loader = DALIVideoLoader(
        video_root=val_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        size=args.size,
        rank=args.rank,
        local_rank=args.local_rank,
        world_size=args.world_size,
        reader_name='ValReader',
        random_shuffle=False,
        random_seed=12 + args.rank
    )

    if args.save_val_data:
        data_root = args.save_val_data
        input_dir = os.path.join(data_root, "input")
        output_dir = os.path.join(data_root, "output")
        target_dir = os.path.join(data_root, "target")
        diff_dir = os.path.join(data_root, "diff")

    # Variables for logging
    if args.rank == 0:
        start_time = prev_time = time.monotonic()
        avg_loss = 0.0
        cnt = 0
        print("=> start evaluation on the validation set")
        if args.save_val_data:
            os.makedirs(input_dir)
            os.makedirs(output_dir)
            os.makedirs(target_dir)
            os.makedirs(diff_dir)

    model.eval()
    for i, data in enumerate(val_loader):
        sequences = data[0]['sequence']
        with torch.no_grad():
            inputs, targets = preprocess_fn(sequences)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            dist.reduce(loss, dst=0)
            if args.save_val_data:
                # (B, C, T, H, W) to (B * T, C, H, W)
                _, c, _, h, w = inputs.size()
                inputs = inputs.transpose(1, 2).reshape(-1, c, h, w)
                outputs = outputs.transpose(1, 2).reshape(-1, c, h, w)
                targets = targets.transpose(1, 2).reshape(-1, c, h, w)
                diff = torch.abs((inputs - outputs) * args.gamma).clamp(0., 1.)
                filename_prefix = str(args.rank) + "_" + str(i)
                utils.save_frames(inputs, input_dir, filename_prefix)
                utils.save_frames(outputs, output_dir, filename_prefix)
                utils.save_frames(targets, target_dir, filename_prefix)
                utils.save_frames(diff, diff_dir, filename_prefix)

        if args.rank == 0:
            avg_loss += loss.item() / args.world_size
            cnt += 1
            if cnt % args.print_freq == 0:
                curr_time = time.monotonic()
                d_time = curr_time - prev_time
                print(f"val_iter{cnt}: avg_time={d_time / args.print_freq}")
                prev_time = curr_time

    if args.rank == 0:
        avg_loss /= cnt + 1
        total_time = time.monotonic() - start_time
        print("Evaluation result: avg_loss: {}, total_time: {}".format(avg_loss,
                                                                       total_time))
        return avg_loss


def test(model, criterion, preprocess_fn, args):
    # Get test video loader
    test_dir = os.path.join(args.data_root, 'test')
    test_video_loader = DALIVideoLoader(
        video_root=test_dir,
        batch_size=2,
        sequence_length=10,
        size=args.size,
        rank=args.rank,
        local_rank=args.local_rank,
        world_size=args.world_size,
        reader_name='testReader',
        random_shuffle=False,
        random_seed=12 + args.rank
    )

    # Variables for logging
    if args.rank == 0:
        sum_loss = 0.0
        cnt = 0
        print("=> start test on the test video set")

    model.eval()
    for i, data in enumerate(test_video_loader):
        sequences = data[0]['sequence']
        with torch.no_grad():
            inputs, targets = preprocess_fn(sequences)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            dist.reduce(loss, dst=0)

        if args.rank == 0:
            sum_loss += loss.item()
            cnt += args.world_size
            if cnt % args.print_freq == 0:
                print(f"cnt={cnt}, sum_loss={sum_loss}")
            if i % 3 == 2:
                # worldsize=5
                # TODO
                psnr = utils.psnr(sum_loss / cnt)
                print(f"PSNR: {psnr}")
                sum_loss, cnt = 0, 0

    if args.rank == 0:
        avg_loss = sum_loss / cnt
        print(f"Test result: avg_loss: {avg_loss}")
        return avg_loss


def parse_args():
    arg_parser = argparse.ArgumentParser(description="Train args for Remastering")
    arg_parser.add_argument('--data_root', type=str, default='data/',
                            help='Path to the root directory of data.')
    arg_parser.add_argument('--checkpoint_path', type=str, default='checkpoint.tar',
                            help='Path to the checkpoint file.')
    arg_parser.add_argument('--img_dir', type=str, default='runs/images/',
                            help='Path to the directory of inputs/outputs/targets '
                                 'images during training. (default: runs/images)')
    arg_parser.add_argument('--output_dir', type=str, default='outputs/',
                            help='Path to the directory of remastered outputs. '
                                 '(default: outputs/)')
    arg_parser.add_argument('--save_val_data', type=str,
                            help='Path to the directory of outputs in validation set.')
    arg_parser.add_argument('--rank', type=int, default=-1,
                            help='Node rank for distributed training.')
    arg_parser.add_argument('--num_gpus_per_node', type=int, default=4,
                            help='Number of GPUs per node. (default: 4)')
    arg_parser.add_argument('--local_rank', type=int, default=-1,
                            help='Local rank in the node.')
    arg_parser.add_argument('--world_size', type=int, default=-1,
                            help='Number of nodes for distributed training.')
    arg_parser.add_argument('--dist_url', type=str, default='tcp://192.168.0.11:31478',
                            help='Url used to set up distributed training.')
    arg_parser.add_argument('--batch_size', type=int, default=1,
                            help='Batch size per process for training. (default: 1)')
    arg_parser.add_argument('--n_grad_accum', type=int, default=4,
                            help='Number of times of gradient accumulation. (default: 4)')
    arg_parser.add_argument('--sequence_length', type=int, default=10,
                            help='Number of frames in a sequence. (default: 10)')
    arg_parser.add_argument('--size', type=int, nargs=2, default=[320, 576],
                            help='Frame size. (default: [320, 576])')
    arg_parser.add_argument('--gamma', type=float, default=5.0,
                            help='Gamma for generating diff images.')
    arg_parser.add_argument('--evaluate', action='store_true',
                            help='Evalutate model on validation set.')
    arg_parser.add_argument('--train', action='store_true',
                            help='Train preprocess.')
    arg_parser.add_argument('--test', action='store_true',
                            help='Test preprocess.')
    arg_parser.add_argument('--remaster', action='store_true',
                            help='Remaster preprocess.')
    arg_parser.add_argument('--resume', type=str, metavar='PATH',
                            help='Path to latest checkpoint. (default: none)')
    arg_parser.add_argument('--print_freq', type=int, default=100,
                            help='Frequency of result printing. (default: 100)')
    arg_parser.add_argument('--save_img_freq', type=int, default=1000,
                            help='Frequency of saving images of inputs/outputs/targets '
                                 'during training. (default: 1000)')
    arg_parser.add_argument('--save_checkpoint_freq', type=int, default=1000,
                            help='Frequency of checkpoint saving. (default: 1000)')
    arg_parser.add_argument('--eval_freq', type=int, default=10000,
                            help='Frequency of evaluate model. (default:10000)')
    arg_parser.add_argument('--dry_run', action='store_true',
                            help='Dry run.')
    return arg_parser.parse_args()


def main_worker(local_rank, args):
    args.local_rank = local_rank
    args.rank = args.rank * args.num_gpus_per_node + local_rank

    # Initialize
    dist.init_process_group('nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    print("=> running on rank {}".format(args.rank))

    # Create model
    model = NetworkR(train=args.train).to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank])

    # Define loss function and optimizer
    optimizer = torch.optim.Adadelta(model.parameters())
    # criterion = torch.nn.MSELoss()
    criterion = get_perceptual_loss_fn(
        layers=('relu3_3', ),
        local_rank=args.local_rank
    )

    # Resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if args.rank == 0:
                print("=> loading checkpoint '{}'".format(args.resume))
            map_location = 'cuda:{}'.format(args.local_rank)
            checkpoint = torch.load(args.resume, map_location=map_location)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.rank == 0:
                print("=> checkpoint was loaded")
        else:
            if args.rank == 0:
                print("=> no checkpoint fount at {}".format(args.resume))

    if args.remaster:
        preprocess_fn = get_preprocess_fn(args)
        remaster(model, preprocess_fn, args)
        dist.barrier()
        dist.destroy_process_group()
        return

    # Transforms for both input and target.
    transform_pair = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.25)),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.9, 1.0))],
            p=0.2
        ),
    ])
    # Transforms for input.
    transform_input = transforms.Compose([
        transforms.RandomApply(
            [transforms.Lambda(lambda x: utils.random_bicubic_blur(x, (0.25, 0.5))),
             transforms.Lambda(lambda x: x.clamp(0.0, 1.0))],
            p=0.5
        ),
        transforms.RandomApply(
            [transforms.Lambda(lambda x: utils.add_gaussian_noise(x, stddev=0.04)),
             transforms.Lambda(lambda x: x.clamp(0.0, 1.0))],
            p=0.1
        ),
        transforms.RandomApply(
            [transforms.ColorJitter(contrast=(0.6, 1.0))],
            p=0.33
        ),
        transforms.RandomApply(
            [transforms.Lambda(lambda x: utils.compress_jpeg(x, (15, 40)))],
            p=0.9
        )
    ])

    preprocess_fn = get_preprocess_fn(args, transform_pair, transform_input)

    if args.evaluate:
        validate(model, torch.nn.L1Loss(), preprocess_fn, args)
        dist.barrier()
        dist.destroy_process_group()
        return

    if args.test:
        test(model, torch.nn.MSELoss(), preprocess_fn, args)
        dist.barrier()
        dist.destroy_process_group()
        return

    train_dir = os.path.join(args.data_root, 'train')
    train_loader = DALIVideoLoader(
        video_root=train_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        size=args.size,
        rank=args.rank,
        local_rank=args.local_rank,
        world_size=args.world_size,
        reader_name='TrainReader',
        random_seed=12 + args.rank
    )

    train(model, criterion, optimizer, train_loader, preprocess_fn, args)

    dist.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    args.world_size = args.world_size * args.num_gpus_per_node
    mp.spawn(main_worker, nprocs=args.num_gpus_per_node, args=(args, ))


if __name__ == '__main__':
    main()
