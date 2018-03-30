import tensorflow as tf
import numpy as np

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

class VartiationalRNNCell(tf.contrib.rnn.RNNCell):
    """Variational RNN cell."""

    def __init__(self, x_dim, h_dim, z_dim = 100):
        """For our purposes, we need to ensure that the whole code
        works when x_dim is a matrix of 35x128"""
        self.n_h = h_dim
        self.n_x = x_dim
        self.n_z = z_dim
        self.n_x_1 = x_dim
        self.n_z_1 = z_dim
        self.n_enc_hidden = z_dim
        self.n_dec_hidden = x_dim
        self.n_prior_hidden = z_dim
        self.lstm = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)


    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return self.n_h

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            h, c = state

            with tf.variable_scope("Prior"):
                with tf.variable_scope("hidden"):
                    prior_hidden = tf.nn.relu(linear(h, self.n_prior_hidden))
                with tf.variable_scope("mu"):
                    prior_mu = linear(prior_hidden, self.n_z)
                with tf.variable_scope("sigma"):
                    prior_sigma = tf.nn.softplus(linear(prior_hidden, self.n_z))

            with tf.variable_scope("phi_x"):
                x_1 = tf.nn.relu(linear(x, self.n_x_1))

            #import ipdb; ipdb.set_trace()
            with tf.variable_scope("Encoder"):
                with tf.variable_scope("hidden"):
                    enc_hidden = tf.nn.relu(linear(tf.concat(axis=1,values=(x_1, h)), self.n_enc_hidden))
                with tf.variable_scope("mu"):
                    enc_mu    = linear(enc_hidden, self.n_z)
                with tf.variable_scope("sigma"):
                    enc_sigma = tf.nn.softplus(linear(enc_hidden, self.n_z))
            eps = tf.random_normal((tf.shape(x)[0], self.n_z), 0.0, 1.0, dtype=tf.float32)
            # z = mu + sigma*epsilon
            z = tf.add(enc_mu, tf.multiply(enc_sigma, eps))
            with tf.variable_scope("phi_z"):
                z_1 = tf.nn.relu(linear(z, self.n_z_1))

            with tf.variable_scope("Decoder"):
                with tf.variable_scope("hidden"):
                    dec_hidden = tf.nn.relu(linear(tf.concat(axis=1,values=(z_1, h)), self.n_dec_hidden))
                with tf.variable_scope("mu"):
                    dec_mu = linear(dec_hidden, self.n_x)
                with tf.variable_scope("sigma"):
                    dec_sigma = tf.nn.softplus(linear(dec_hidden, self.n_x))


            output, state2 = self.lstm(tf.concat(axis=1,values=(x_1, z_1)), state)
        return (enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma), state2




class VRNN():
    def __init__(self, args, sample=False):

        
        def tf_normal(y, mu, s):
            """This function returns negative log of pdf of a sample y,
            assuming Y follows a gaussian dist with mu, s as parameters"""
            #s = sigma
            #mu = mean
            with tf.variable_scope('normal'):
                #ss = sigma squared
                ss = tf.maximum(1e-10,tf.square(s)) 
                norm = tf.subtract(y[:,:args.num_features], mu)
                z = tf.div(tf.square(norm), ss)
                denom_log = tf.log(2*np.pi*ss, name='denom_log')
                result = tf.reduce_sum(z+denom_log, 1)/2# -

            return result

        def tf_kl_gaussgauss(mu_1, sigma_1, mu_2, sigma_2):
            """returns kl divergence of the two distributions
                https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            """
            with tf.variable_scope("kl_gaussgauss"):
                return tf.reduce_sum(0.5 * (
                    2 * tf.log(tf.maximum(1e-9,sigma_2),name='log_sigma_2') 
                  - 2 * tf.log(tf.maximum(1e-9,sigma_1),name='log_sigma_1')
                  + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9,(tf.square(sigma_2))) - 1
                ), 1)

        def get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma, y):
            """Obtain total loss: kl_loss + likelihood loss"""
            kl_loss = tf_kl_gaussgauss(enc_mu, enc_sigma, prior_mu, prior_sigma)
            likelihood_loss = tf_normal(y, dec_mu, dec_sigma)

            return tf.reduce_mean(kl_loss + likelihood_loss)
            #return tf.reduce_mean(likelihood_loss)

        self.args = args
        if sample:
            args.batch_size = 1
            #args.seq_length = 1

        cell = VartiationalRNNCell(args.num_features, args.rnn_size, args.latent_size)

        self.cell = cell

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, args.num_features], name='input_data')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, args.num_features],name = 'target_data')
        #self.initial_state_c, self.initial_state_h = cell.zero_state(batch_size=None, dtype=tf.float32)


        # input shape: (batch_size, n_steps, n_input)
        with tf.variable_scope("inputs"):
            inputs = tf.transpose(self.input_data, [1, 0, 2])  # permute n_steps and batch_size
            inputs = tf.reshape(inputs, [-1, args.num_features]) # (n_steps*batch_size, n_input)

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            inputs = tf.split(axis=0, num_or_size_splits=args.seq_length, value=inputs) # n_steps * (batch_size, n_hidden)
        flat_target_data = tf.reshape(self.target_data,[-1, args.num_features])

        self.target = flat_target_data
        self.flat_input = tf.reshape(tf.transpose(tf.stack(inputs),[1,0,2]),[-1, args.num_features])
        self.input = tf.stack(inputs)
        # Get vrnn cell output
        #outputs, last_state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=(self.initial_state_c,self.initial_state_h))
        outputs, last_state = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32) 
        #print outputs
        #outputs = map(tf.pack,zip(*outputs))
        outputs_reshape = []
        names = ["enc_mu", "enc_sigma", "dec_mu", "dec_sigma", "prior_mu", "prior_sigma"]
        for n,name in enumerate(names):
            with tf.variable_scope(name):
                x = tf.stack([o[n] for o in outputs])
                x = tf.transpose(x,[1,0,2])
                x = tf.reshape(x,[-1, tf.shape(x)[2]])
                outputs_reshape.append(x)

        enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma = outputs_reshape
        self.final_state_c,self.final_state_h = last_state
        self.mu = dec_mu
        self.sigma = dec_sigma

        lossfunc = get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma, flat_target_data)
        self.sigma = dec_sigma
        self.mu = dec_mu
        with tf.variable_scope('cost'):
            self.cost = lossfunc 
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('mu', tf.reduce_mean(self.mu))
        tf.summary.scalar('sigma', tf.reduce_mean(self.sigma))


        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        for t in tvars:
            print t.name
        grads = tf.gradients(self.cost, tvars)
        #grads = tf.cond(
        #    tf.global_norm(grads) > 1e-20,
        #    lambda: tf.clip_by_global_norm(grads, args.grad_clip)[0],
        #    lambda: grads)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        #self.saver = tf.train.Saver(tf.all_variables())

    def sample(self, sess, args,start=None):

        def sample_gaussian(mu, sigma):
            return mu + (sigma*np.random.randn(*sigma.shape))

        if start is None:
            prev_x = np.random.randn(1, args.seq_length, args.num_features)
        elif len(start.shape) == 1:
            prev_x = start[np.newaxis,np.newaxis,:]
        elif len(start.shape) == 2:
            for i in range(start.shape[0]-1):
                prev_x = start[i,:]
                prev_x = prev_x[np.newaxis,np.newaxis,:]
                feed = {self.input_data: prev_x,
                        self.initial_state_c:prev_state[0],
                        self.initial_state_h:prev_state[1]}
                
                [o_mu, o_sigma, prev_state_c, prev_state_h] = sess.run(
                        [self.mu, self.sigma,
                         self.final_state_c,self.final_state_h],feed)

            prev_x = start[-1,:]
            prev_x = prev_x[np.newaxis,np.newaxis,:]

        prev_state = sess.run(self.cell.zero_state(1, tf.float32))
        chunks = np.zeros((args.seq_length, args.num_features), dtype=np.float32)
        mus = np.zeros((args.seq_length, args.num_features), dtype=np.float32)
        sigmas = np.zeros((args.seq_length, args.num_features), dtype=np.float32)

        [o_mu, o_sigma] = sess.run([self.mu, self.sigma],{self.input_data: prev_x})
        sample_x = sample_gaussian(o_mu, o_sigma)
        return sample_x, o_mu, o_sigma
        #for i in xrange(num):
        #    feed = {self.input_data: prev_x,
        #            self.initial_state_c:prev_state[0],
        #            self.initial_state_h:prev_state[1]}
        #    [o_mu, o_sigma, next_state_c, next_state_h] = sess.run([self.mu, self.sigma,
        #        self.final_state_c, self.final_state_h],feed)

        #    next_x = np.hstack(sample_gaussian(o_mu, o_sigma))
        #    chunks[i] = next_x
        #    mus[i] = o_mu
        #    sigmas[i] = o_sigma

        #    prev_x = np.zeros((1, 1, args.num_features), dtype=np.float32)
        #    prev_x[0][0] = next_x
        #    prev_state = next_state_c, next_state_h

        #return chunks, mus, sigmas
