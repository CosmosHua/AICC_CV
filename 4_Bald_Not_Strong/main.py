# coding:utf-8
#!/usr/bin/python3
import argparse, os
import tensorflow as tf
from model import pix2pix


parser = argparse.ArgumentParser(description='')
#parser.add_argument('--dataset', dest='dataset', default='IDFace', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=2, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=50000, help='# images used to train')
#parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
#parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
#parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
#parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
#parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
#parser.add_argument('--direction', dest='direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
#parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
#parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
#parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
#parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
#parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='takes images in order to make batches, otherwise takes them randomly')
#parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--model_dir', dest='model_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--train_dir', dest='train_dir', default='./Train', help='training data are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./Sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./Test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=200., help='weight on L1 term in objective')
parser.add_argument('--SS_lambda', dest='SS_lambda', type=float, default=5, help='weight on SSIM term in objective')
args = parser.parse_args()


def main(_):
    args.train_dir = "/home/joshua/AICC4/IDFace1W4"
    args.model_dir += "/" + os.path.basename(args.train_dir)
    if args.phase=="train": assert(os.path.isdir(args.train_dir))
    if not os.path.exists(args.model_dir):  os.makedirs(args.model_dir) # for train/test
    if not os.path.exists(args.test_dir):   os.makedirs(args.test_dir) # for train/test

    print("\nParameters :\n", args) # device_count: limit number of GPUs
    config = tf.ConfigProto(device_count={"GPU":1}, allow_soft_placement=True)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.1 # gpu_memory ratio
    config.gpu_options.allow_growth = True # dynamicly apply gpu_memory
    with tf.Session(config=config) as sess:
        model = pix2pix(sess = sess, batch_size = args.batch_size,
                        input_nc = args.input_nc, output_nc = args.output_nc,
                        L1_lambda = args.L1_lambda, SS_lambda = args.SS_lambda)
        if args.phase=="train": model.train(args) # Train
        else:                   model.test(args) # Test


if __name__ == '__main__':
    tf.app.run()


#####################################################################
# nohup python main.py --phase=train > log.txt & # Train
# rm test/*.png; python main.py --phase=test --batch_size=5; sz test/003833*.png

