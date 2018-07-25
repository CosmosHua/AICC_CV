# coding:utf-8
#!/usr/bin/python3
import numpy as np
import os, time, cv2
import tensorflow as tf
from glob import glob

from ops import *
from utils import *


class pix2pix(object):
    def __init__(self, sess, batch_size=1, image_size=256, output_size=256, gf_dim=64, df_dim=64,
                 input_nc=3, output_nc=3, L1_lambda=200, SS_lambda=5, keep=10): # old L1_lambda=100
        """ Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            image_size: The resolution in pixels of the input images. [256]
            output_size: The resolution in pixels of the output images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_nc: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_nc: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.L1_lambda = L1_lambda
        self.SS_lambda = SS_lambda

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.is_grayscale = (input_nc==1)
        self.kp = keep # max number of model
        self.ps = [] # keep [PNSR, SSIM, PNSR*SSIM]

        # batch normalization :
        # deals with poor initialization, enhances gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')
        self.build_model()


    def build_model(self):
        shape = [self.batch_size, self.image_size, self.image_size, self.input_nc + self.output_nc]
        self.real_data = tf.placeholder(dtype=TType, shape=shape, name='real_A_and_B_images')
        
        self.real_B = self.real_data[:, :, :, :self.input_nc] # clean image
        self.real_A = self.real_data[:, :, :, self.input_nc:] # mesh image
        
        #if self.image_size==250: # help(tf.pad): [batch_size,height,width,channels]
        #   paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
        #   self.real_A = tf.pad(self.real_A, paddings=paddings, mode='REFLECT')
        #   self.real_B = tf.pad(self.real_B, paddings=paddings, mode='REFLECT')
        
        self.fake_B = self.generator(self.real_A) # recover image
        self.fake_B_sample = self.sampler(self.real_A)
        
        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)

        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)
        
        # self.d_sum = tf.summary.histogram("d", self.D) # unnecessary
        # self.d__sum = tf.summary.histogram("d_", self.D_) # unnecessary
        # self.fake_B_sum = tf.summary.image("fake_B", self.fake_B) # unnecessary
        
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
       
        self.g_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
        self.g_loss_l1  = tf.reduce_mean(tf.abs(self.real_B - self.fake_B)) * self.L1_lambda
        self.g_loss_ss  = tf.reduce_mean([SSLF(self.real_B[i,:], self.fake_B[i,:]) for i in range(self.batch_size)]) * self.SS_lambda
        self.g_loss = self.g_loss_gan + self.g_loss_l1 + self.g_loss_ss # SSIM loss
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.saver = tf.train.Saver(max_to_keep=self.kp)


    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            print("\n" + "discriminator_"*4)
            if reuse: tf.get_variable_scope().reuse_variables()
            else: assert(tf.get_variable_scope().reuse == False)

            # image is (256 x 256 x (input_nc+output_nc))
            # h0 is (128 x 128 x self.df_dim)
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h1 is (64 x 64 x self.df_dim*2)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            return tf.nn.sigmoid(h4), h4


    def generator(self, image):
        with tf.variable_scope("generator") as scope:
            print("\n" + "generator_"*6)
            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = [int(s/2**i) for i in range(1,8)]

            # image is (256 x 256 x input_nc)
            # e1 is (128 x 128 x self.gf_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            print(tf.Print(e1, [e1]))

            # e2 is (64 x 64 x self.gf_dim*2)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            print(tf.Print(e2, [e2]))
            
            # e3 is (32 x 32 x self.gf_dim*4)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            print(tf.Print(e3, [e3]))

            # e4 is (16 x 16 x self.gf_dim*8)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            print(tf.Print(e4, [e4]))

            # e5 is (8 x 8 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            print(tf.Print(e5, [e5]))

            # e6 is (4 x 4 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            print(tf.Print(e6, [e6]))

            # e7 is (2 x 2 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            print(tf.Print(e7, [e7]))

            # e8 is (1 x 1 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            print(tf.Print(e8, [e8]))

            # d1 is (2 x 2 x self.gf_dim*8*2)
            shape = [self.batch_size, s128, s128, self.gf_dim*8]
            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8), shape, name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.8) # change to 0.3le
            d1 = tf.concat([d1, e7], 3)
            print(tf.Print(self.d1, [self.d1]))

            # d2 is (4 x 4 x self.gf_dim*8*2)
            shape = [self.batch_size, s64, s64, self.gf_dim*8]
            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1), shape, name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.8)
            d2 = tf.concat([d2, e6], 3)
            print(tf.Print(self.d2, [self.d2]))

            # d3 is (8 x 8 x self.gf_dim*8*2)
            shape = [self.batch_size, s32, s32, self.gf_dim*8]
            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2), shape, name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.8)
            d3 = tf.concat([d3, e5], 3)
            print(tf.Print(self.d3, [self.d3]))

            # d4 is (16 x 16 x self.gf_dim*8*2)
            shape = [self.batch_size, s16, s16, self.gf_dim*8]
            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3), shape, name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            print(tf.Print(self.d4, [self.d4]))

            # d5 is (32 x 32 x self.gf_dim*4*2)
            shape = [self.batch_size, s8, s8, self.gf_dim*4]
            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4), shape, name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            print(tf.Print(self.d5, [self.d5]))

            # d6 is (64 x 64 x self.gf_dim*2*2)
            shape = [self.batch_size, s4, s4, self.gf_dim*2]
            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5), shape, name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            print(tf.Print(self.d6, [self.d6]))

            # d7 is (128 x 128 x self.gf_dim*1*2)
            shape = [self.batch_size, s2, s2, self.gf_dim]
            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6), shape, name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            print(tf.Print(self.d7, [self.d7]))

            # d8 is (256 x 256 x output_nc)
            shape = [self.batch_size, s, s, self.output_nc]
            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), shape, name='g_d8', with_w=True)
            print(tf.Print(self.d8, [self.d8]))
            return tf.nn.tanh(self.d8)


    def sampler(self, image):
        with tf.variable_scope("generator") as scope:
            print("\n" + 'sampler_'*6)
            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = [int(self.output_size/2**i) for i in range(1,8)]

            scope.reuse_variables()
            # image is (256 x 256 x input_nc)
            # e1 is (128 x 128 x self.gf_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            print(tf.Print(e1, [e1]))

            # e2 is (64 x 64 x self.gf_dim*2)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            print(tf.Print(e2, [e2]))

            # e3 is (32 x 32 x self.gf_dim*4)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            print(tf.Print(e3, [e3]))

            # e4 is (16 x 16 x self.gf_dim*8)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            print(tf.Print(e4, [e4]))

            # e5 is (8 x 8 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            print(tf.Print(e5, [e5]))

            # e6 is (4 x 4 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            print(tf.Print(e6, [e6]))

            # e7 is (2 x 2 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            print(tf.Print(e7, [e7]))

            # e8 is (1 x 1 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            print(tf.Print(e8, [e8]))

            # d1 is (2 x 2 x self.gf_dim*8*2)
            shape = [self.batch_size, s128, s128, self.gf_dim*8]
            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8), shape, name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.8)
            d1 = tf.concat([d1, e7], 3)
            print(tf.Print(self.d1, [self.d1]))

            # d2 is (4 x 4 x self.gf_dim*8*2)
            shape = [self.batch_size, s64, s64, self.gf_dim*8]
            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1), shape, name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.8)
            d2 = tf.concat([d2, e6], 3)
            print(tf.Print(self.d2, [self.d2]))

            # d3 is (8 x 8 x self.gf_dim*8*2)
            shape = [self.batch_size, s32, s32, self.gf_dim*8]
            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2), shape, name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.8)
            d3 = tf.concat([d3, e5], 3)
            print(tf.Print(self.d3, [self.d3]))

            # d4 is (16 x 16 x self.gf_dim*8*2)
            shape = [self.batch_size, s16, s16, self.gf_dim*8]
            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),shape, name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            print(tf.Print(self.d4, [self.d4]))

            # d5 is (32 x 32 x self.gf_dim*4*2)
            shape = [self.batch_size, s8, s8, self.gf_dim*4]
            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4), shape, name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            print(tf.Print(self.d5, [self.d5]))

            # d6 is (64 x 64 x self.gf_dim*2*2)
            shape = [self.batch_size, s4, s4, self.gf_dim*2]
            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5), shape, name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            print(tf.Print(self.d6, [self.d6]))

            # d7 is (128 x 128 x self.gf_dim*1*2)
            shape = [self.batch_size, s2, s2, self.gf_dim]
            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6), shape, name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            print(tf.Print(self.d7, [self.d7]))

            # d8 is (256 x 256 x output_nc)
            shape = [self.batch_size, s, s, self.output_nc]
            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), shape, name='g_d8', with_w=True)
            print(tf.Print(self.d8, [self.d8]))
            return tf.nn.tanh(self.d8)


    def train(self, args): # Train pix2pix
        train_dir = args.train_dir; data = []
        for dirpath, dirnames, filenames in os.walk(train_dir):
            data += glob(dirpath + "/*_*.jpg")
        print("\n" + "#"*50 + "\nThe Number of Training =", len(data))
        batch_num = min(len(data), args.train_size) // self.batch_size
        #iter_num = batch_num*args.epoch # iter_num <= epoch*train_size

        self.g_sum = tf.summary.merge([self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_loss_real_sum, self.d_loss_sum])
        # self.g_sum = tf.summary.merge([self.d__sum, self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        # self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        # self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.g_loss, var_list=self.g_vars)

        counter = 0; start_time = time.time()
        self.sess.run(tf.global_variables_initializer())
        self.load_model(args.model_dir) # restore model
        self.test_model(args, counter) # test initial model
        for epoch in range(args.epoch):
            np.random.shuffle(data) # shuffle data
            for id in range(batch_num):
                batch = data[id*self.batch_size:(id+1)*self.batch_size]
                images = self.load_data(batch, is_test=False)

                # Update D network:
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.real_data: images})
                # self.writer.add_summary(summary_str, counter)

                # Update G network:
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.real_data: images})
                # self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.real_data: images})
                # self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.real_data: images})
                errD_real = self.d_loss_real.eval({self.real_data: images})
                errG = self.g_loss.eval({self.real_data: images})

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % \
                      (epoch, id, batch_num, time.time()-start_time, errD_fake+errD_real, errG))

                counter += 1
                if counter % 500 == 0: self.test_model(args, counter) # save model
                #if counter % 500 == 0: self.sample_model(args.sample_dir, counter)
                #if counter % 1000 == 0: self.save_model(args.model_dir, counter)
        if counter%500!=0: self.test_model(args, counter) # test/save the final model


    def test(self, args): # Test pix2pix
        start_time = time.time(); batch = args.batch_size
        data = glob(os.path.join(args.test_dir, "*.jpg"))
        data.sort(); num = (len(data)//batch)*batch # multiple of batch
        images = self.load_data(data, is_test=True) # [N,height,width,channel]
        images = [images[i:i+batch] for i in range(0,num,batch)] # repack
        print("\n"+"#"*50+"\nTesting = %d images" % len(images))
        
        # self.sess.run(tf.global_variables_initializer())
        if not self.load_model(args.model_dir): return # Load model
        print("Load time: %f s" % (time.time()-start_time)); load_time = time.time()
        
        for i,image in enumerate(images):
            im = self.sess.run(self.fake_B_sample, feed_dict={self.real_data: image})
            save_images(im, data[i*batch][:-4]+".png")
        print("Test time: %f s" % (time.time()-load_time))
        if batch==1: print("[PSNR SSIM] =", BatchPS(args.test_dir))


    # similiar to test: save model, without load_model
    def test_model(self, args, counter, id=2, ff="ps.log"):
        mesh = glob(os.path.join(args.test_dir, "*.jpg"))
        # insert batch_dim=1 after axis=0 (where None->1)
        images = self.load_data(mesh, is_test=True)[:,None]
        for i,image in enumerate(images): # output *.png to args.test_dir
            im = self.sess.run(self.fake_B_sample, feed_dict={self.real_data: image})
            save_images(im, mesh[i][:-4]+".png") # batch_size=1
        
        ps = self.ps; out = "" # ps is a reference of self.ps
        pp = BatchPS(args.test_dir) # [PNSR, SSIM, PNSR*SSIM]
        # Other Strategy: {mean/max/min/...} + {[id]/all/any/...}
        if len(ps)<2: out = "@" # enqueue: initialize ps[:2]
        elif (pp>np.mean(ps,axis=0))[id]: # mean/max/min, [id]/all/any
            if len(ps)<self.kp: out = "@" # enqueue if underfill
            elif (pp>ps[0])[id]: ps.pop(0); out = "@" # dequeue oldest
        '''
        if len(ps)<2: out = "@" # enqueue: initialize ps[:2]
        elif len(ps)<self.kp: # enqueue: when exceed mean/max/min/...
            if (pp>np.mean(ps,axis=0))[id]: out = "@" # [id]/all()/any()
        else: # dequeue: when exceed the oldest
            if (pp>ps[0])[id]: ps.pop(0); out = "@" # [id]/all()/any()
        '''
        if counter<5: out = "" # ignore initial models
        if out!="": ps.append(pp); self.save_model(args.model_dir,counter)
        
        out += "num=%d\t[PSNR SSIM] = "%counter + str(pp)
        with open(ff,"a+") as f: f.write(out+"\n") # append+read-write
        print(out); return pp


    def save_model(self, model_dir, step):
        model_name = os.path.join(model_dir, "pix2pix")
        self.saver.save(self.sess, model_name, global_step=step)


    def load_model(self, model_dir):
        print(" [*] Loading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            model = os.path.join(model_dir, ckpt_name)
            self.saver.restore(self.sess, model)
            print(" [*] Load SUCCESS: %s" % model); return True
        else: print(" [!] Load FAIL!"); return False


    def sample_model(self, sample_dir, counter): # validation
        data = glob(sample_dir + "/*_*.jpg")
        data = np.random.choice(data, self.batch_size)
        images = self.load_data(data, is_test=False)
        im, d_loss, g_loss = self.sess.run([self.fake_B_sample, self.d_loss, self.g_loss], feed_dict={self.real_data: images})
        save_images(im, './{}/train_{:08d}.png'.format(sample_dir, counter))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))
    
    
    def load_data(self, data, is_test=False): # batch images
        size = (self.image_size, self.image_size) # for resize
        images = [load_image(im, size, is_test) for im in data]
        if not self.is_grayscale: # for color images
            return np.array(images).astype(np.float32)
        else: # grayscale: insert channel dim/axis at None->1
            return np.array(images).astype(np.float32)[:,:,:,None]

