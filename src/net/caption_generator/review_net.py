import sys
sys.path.append('./src/common/linker')
sys.path.append('./src/caption_generator')
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from attention import Attention
from mod_lstm import ModLSTM


class Fire(chainer.Chain):
    def __init__(self, in_size, s1, e1, e3):
        super().__init__(
            conv1=L.Convolution2D(in_size, s1, 1),
            conv2=L.Convolution2D(s1, e1, 1),
            conv3=L.Convolution2D(s1, e3, 3, pad=1),
        )

    def __call__(self, x):
        h = F.elu(self.conv1(x))
        h_1 = self.conv2(h)
        h_3 = self.conv3(h)
        h_out = F.concat([h_1, h_3], axis=1)
        return F.elu(h_out)


class Encoder(chainer.Chain):
    def __init__(self, in_ch, n_unit):
        super().__init__(
            conv1=L.Convolution2D(in_ch, 96, 7, stride=2, pad=3),
            fire2=Fire(96, 16, 64, 64),
            fire3=Fire(128, 16, 64, 64),
            fire4=Fire(128, 16, 128, 128),
            fire5=Fire(256, 32, 128, 128),
            fire6=Fire(256, 48, 192, 192),
            fire7=Fire(384, 48, 192, 192),
            fire8=Fire(384, 64, 256, 256),  # (n_batch, 512, 44, 44), 44*44=1936
            conv9=L.Convolution2D(512*21, 4096, 1, pad=0),  # (n_batch, 1024)
            conv10=L.Convolution2D(4096, n_unit, 1, pad=0),
        )
        self.train = True

    def __call__(self, x):
        h = F.elu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fire2(h)
        h = self.fire3(h)
        h = self.fire4(h)

        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fire5(h)
        h = self.fire6(h)
        h = self.fire7(h)
        self.sorce_hidden_state = self.fire8(h)  # H in the paper.

        h = F.spatial_pyramid_pooling_2d( \
            self.sorce_hidden_state, 3, F.MaxPooling2D)
        h = F.elu(self.conv9(h))
        h = F.dropout(h, ratio=0.5, train=self.train)
        self.context_vec = self.conv10(h)


class Reviewer(chainer.Chain):
    def __init__(self, n_hid, rnn_size):  # 44*44, 1096
        super().__init__(
            image_attention = Attention(n_hid, rnn_size),
            rev_lstm = L.LSTM(rnn_size, rnn_size),  # size of h_i is ch
        )
        self.align_source = None
        self.pre_hidden_state = None

    def __call__(self):
        att_res = self.image_attention( \
                    self.align_source, self.pre_hidden_state)
        next_hidden_state = self.rev_lstm(att_res)
        self.pre_hidden_state = next_hidden_state

    def clear(self):
        self.rev_lstm.reset_state()
        self.align_source = None
        self.pre_hidden_state = None


class Decoder(chainer.Chain):
    def __init__(self, rnn_size, n_unit, voc_size):
        super().__init__(
            wrod_attention = Attention(rnn_size, n_unit),
            dec_lstm = L.LSTM(n_unit, n_unit),
            l_out = L.Linear(n_unit, voc_size),
        )
        self.align_source = None
        self.pre_hidden_state = None
        self.train = True
        self.sentence_ids = []

    def clear(self):
        self.dec_lstm.reset_state()
        self.sentence_ids = []
        self.align_source = None
        self.pre_hidden_state = None

    def __call__(self, t):
        att = self.wrod_attention(self.align_source, self.pre_hidden_state)
        h_state = self.dec_lstm(att)
        y = self.l_out(F.dropout(h_state, train=self.train))  # don't forget to change drop out into non train mode.
        self.sentence_ids.append(F.argmax(y, axis=1).data)
        self.pre_hidden_state = h_state
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        return loss, accuracy,


class ReviewNet(chainer.Chain):
    def __init__(self, n_class, in_ch, voc_size=100, n_view_steps=8, loss_balancing=10,
                        feat_dim=512, n_hid=44*44, rnn_size=1096, n_unit=512,
                        sent_len=100):  # 44*44=w*h in fire8
        super().__init__(
            encoder=Encoder(in_ch, feat_dim),
            reduce_feat_dim=L.Convolution2D(feat_dim, 1, 1, pad=0),
            reviewer=Reviewer(n_hid, rnn_size),  # hidden_state_size, prev_thought_vector_size
            disc_linear=L.Linear(rnn_size, voc_size),
            decoder=Decoder(rnn_size, n_unit, voc_size)
        )
        self.train = True
        self.n_class = n_class
        self.active_learn = False

        self.n_view_steps = n_view_steps
        self.n_hid = n_hid
        self.rnn_size = rnn_size
        self.n_unit = n_unit
        self.voc_size = voc_size
        self.loss_balancing = loss_balancing

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.reviewer.clear()
        self.decoder.clear()
        self.decoder.train = self.train
        self.encoder.train = self.train
        self.sentence_ids = None

    def compute_discriminative_loss(self, fact_vectors):
        reason_pool = F.max(fact_vectors, axis=0)  # dim of fact_vectors is (n_view_steps, batch_size, voc_size)
        batch_size, voc_size = reason_pool.data.shape
        loss = 0
        for i_word in range(voc_size):
            one_word_score = \
                F.get_item(reason_pool, [range(batch_size), i_word])
            one_word_score = F.expand_dims(one_word_score, axis=1)
            broaded_one_word_score = \
                F.broadcast_to(one_word_score, reason_pool.data.shape)
            real_value_of_hinge = 1-(reason_pool-broaded_one_word_score)
            zero_mat = Variable(self.xp.zeros_like( \
                        real_value_of_hinge.data, self.xp.float32), volatile='auto')
            hinge_loss = F.where( \
                real_value_of_hinge.data<0, zero_mat, real_value_of_hinge.data)
            loss += F.sum(hinge_loss) - voc_size
        loss /= (voc_size-1)*voc_size
        return loss

    def forward_reviewer(self, align_source):
        self.reviewer.align_source = align_source

        pre_hidden_state = self.xp.zeros((self.batch_size, self.rnn_size))
        self.reviewer.pre_hidden_state = Variable( \
            pre_hidden_state.astype(self.xp.float32), volatile='auto')

        for idx, t in enumerate(range(self.n_view_steps)):
            self.reviewer()
            t_reason = self.disc_linear(self.reviewer.pre_hidden_state)
            if idx==0:
                reason_matrix = F.expand_dims(t_reason, axis=0)
            else:
                t_reason_matrix = F.expand_dims(t_reason, axis=0)
                reason_matrix = \
                    F.concat((reason_matrix, t_reason_matrix), axis=0)
        # compute_discriminative_loss is the hinge loss
        return self.compute_discriminative_loss(reason_matrix)

    def forward_decoder(self, tokens):
        '''
        tokens:[[word11,word12,...,EOS],...,[word n1, word n2...,EOS]] as words in a text
        '''
        batch_size, n_word = tokens.data.shape
        self.decoder.align_source = self.reviewer.pre_hidden_state
        # pre_hidden_state must modify.
        self.decoder.pre_hidden_state = self.encoder.context_vec

        # extract words array(ex [w11,w21]) from words matrix(ex [[w11,w12,...],[w21,w22...]])
        # There are as many texts as there are batch size.
        first_word = F.get_item(tokens, [range(self.batch_size), 0])
        self.loss, self.accuracy = self.decoder(first_word)

        neg_one_mat = Variable(self.xp.ones_like( \
            self.decoder.pre_hidden_state.data, self.xp.float32), volatile='auto')
        neg_one_mat *= -1

        # cur_word is current word in each sentence as 1d array.
        # next_word is next word in each sentence as 1d array.
        for i_word in range(1, n_word):
            next_word = F.get_item(tokens, [range(batch_size), i_word])
            broaded_next_word = F.broadcast_to( \
                F.expand_dims(next_word, axis=1), neg_one_mat.data.shape)
            self.decoder.pre_hidden_state = F.where( \
                broaded_next_word.data==-1, neg_one_mat, self.decoder.pre_hidden_state)
            loss, acc = self.decoder( \
                F.get_item(tokens, [range(self.batch_size), i_word]))
            self.loss += loss
            self.accuracy += acc
        self.accuracy /= n_word

    def __call__(self, x, t, tokens):
        self.clear()
        x.volatile = not self.train
        self.batch_size = len(x.data)

        self.encoder(x)
        align_source = self.reduce_feat_dim(self.encoder.sorce_hidden_state)

        batch_size, _, w, h = align_source.data.shape
        align_source = F.reshape(align_source, (batch_size, w*h))

        discriminative_loss = self.forward_reviewer(align_source)
        self.forward_decoder(tokens)

        self.loss += self.loss_balancing*discriminative_loss
        chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
        return self.loss
