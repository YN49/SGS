from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras import backend as K

import matplotlib.pyplot as plt
import argparse


from PIL import Image

import math

import os

import gym
import numpy as np
import gym.spaces

import pygame
from pygame.locals import *
import pygame.gfxdraw


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



class PIC(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    MAX_STEPS = 50
    RANGE = 0.18#報酬やるときにどのくらいの距離だったら同じものだという認識に入るか
    f_model = './model'

    PIC = np.array(Image.open('強化学習/pic_env/MAP_PIC.png').convert('L'))

    WIDTH = 950
    HEIGHT = 450


    weights_filename = '強化学習/pic_env/vae.hdf5'

    epochs = 200#vaeのエポック数
    TRAIN_FREQ = 1000000000000000#何エピソードに一回学習するか100

    original_dim = 25

    # network parameters
    input_shape = (original_dim, )
    intermediate_dim = 15
    batch_size = 128
    latent_dim = 2

    MIDDLE_layer = 2

    ##############################################モデル構成##############################################

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')



    def __init__(self):
        super().__init__()
        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(4)  # 東西南北設置破壊
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(29,)
        )
        self.reward_range = [-1., 100.]

        self._reset()

        self.viewer = None

    def _reset(self):
        # 諸々の変数を初期化する
        self.pos = np.array([8, 8], dtype=int)#画面の座標x,画面のディレクトリ
        self.done = False
        self.steps = 0

        self.reset_rend = False

        self.out_train = np.zeros((2,25))

        self.encoded_obs = np.zeros(self.MIDDLE_layer, )
        
        try:
            loaded_array = np.load('強化学習/pic_env/data.npz')
            self.x_train = loaded_array['arr_0']
            self.train_data = loaded_array['arr_1']
            #self.TARGET = loaded_array['arr_2']
            
            self.update_traget()#ターゲットとりあえずランダム設定
        except FileNotFoundError:
            self.x_train = np.zeros((2,25))
            self.train_data = np.array([0,])
            self.update_traget()#ターゲットとりあえずランダム設定


        try:
            self.vae.load_weights(os.path.join(self.weights_filename))
        except:
            pass


        return self._observe()


        

    def _step(self, action):
        # 1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)
        if action == 0:
            next_pos = self.pos + np.array([1, 0])
        elif action == 1:
            next_pos = self.pos + np.array([-1, 0])
        elif action == 2:
            next_pos = self.pos + np.array([0, 1])
        else:
            next_pos = self.pos + np.array([0, -1])
        
        if 2 <= next_pos[0] < self.PIC.shape[0] - 2 and 2 <= next_pos[1] < self.PIC.shape[1] - 2:
            self.pos = next_pos


        #エンコードするぞ
        #self.encoded_obs = np.array(self.encoder.predict(np.squeeze(self.obs_encoder())[np.newaxis,:])).reshape(self.MIDDLE_layer, )
        self.encoded_obs, _, _ = self.encoder.predict(np.squeeze(self.obs_encoder())[np.newaxis,:])
        self.encoded_obs = self.encoded_obs.reshape(self.MIDDLE_layer, )

        self.action = action
        self.steps = self.steps + 1
        observation = self._observe()
        reward = self._get_reward()
        self.done = self._is_done()


        #現ステップのobservationを教師データに格納
        self.out_train = np.insert(self.out_train, self.out_train.shape[0], self.obs_encoder(), axis=0)

        if self.done:
            
            #終了時に先端の２つの余計な配列を取り除く
            self.out_train = np.delete(self.out_train, 0, 0)
            self.out_train = np.delete(self.out_train, 0, 0)

            #self.out_train = self.out_train.astype('float32') / 255.

            self.x_train = np.insert(self.out_train, self.out_train.shape[0], self.x_train, axis=0)
            if self.train_data[0] == 0:#一番最初の学習の場合
                #先端の２つの余計な配列を取り除く
                self.x_train = np.delete(self.x_train, 0, 0)
                self.x_train = np.delete(self.x_train, 0, 0)

            self.train_data[0] = self.train_data[0] + 1#エピソード数カウント

            if self.train_data[0] >= self.TRAIN_FREQ:#MAX_STEP*50ステップに一回学習
                self.train_data[0] = 0


                #オートエンコーダ実行
                self.auto_encoder()

                #ひとがくしゅう終わったからパラメータを初期化する
                #self.x_train = np.zeros((2,25))
                self.train_data = np.array([0,])

            #保存
            np.savez('強化学習/pic_env/data.npz', self.x_train, self.train_data, self.TARGET)


        return observation, reward, self.done, {}

    def _render(self, mode='human', close=False):
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        #os.system('cls')
        #print(self.pos)
        #print(self._get_reward())
        #print(self.obs())
        #print(math.sqrt(np.sum((self.encoded_obs - self.TARGET) ** 2)),'差')
        #print('範囲',self.range_calcu,'x',math.sqrt(np.sum(self.TARGET**2)))
        #print(self.encoded_obs)
        #print(self.TARGET)
        #print(self.x_train)
        #print(self.train_data)
        
        if not self.reset_rend:#一度目の処理なので描画初期化
            # Pygameを初期化
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("pic-gym-v0")              # タイトルバーに表示する文字
            self.font = pygame.font.Font(None, 15) 
            self.font_item = pygame.font.Font(None, 30)               # フォントの設定(55px)
            self.screen.fill((0,0,0))


        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        self.TARGET2d = self.TARGET.reshape(int(self.MIDDLE_layer/2),2)*255
        self.encoded2d = self.encoded_obs.reshape(int(self.MIDDLE_layer/2),2)*255

        self.screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.array([self.obs(),self.obs(),self.obs()]).transpose(1, 2, 0)), (self.obs().shape[0] * 24 , self.obs().shape[1] * 24)), (10, 10))
        self.screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.array([self.TARGET_PIC,self.TARGET_PIC,self.TARGET_PIC]).transpose(1, 2, 0)), (self.TARGET_PIC.shape[0] * 24 , self.TARGET_PIC.shape[1] * 24)), (self.obs().shape[0] * 24 + 10, 10))
        #self.screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.array([self.TARGET2d,self.TARGET2d,self.TARGET2d]).transpose(1, 2, 0)), (self.TARGET2d.shape[0] * 24 , self.TARGET2d.shape[1] * 24)), (self.obs().shape[0] * 48 + 10, 10))
        #self.screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.array([self.encoded2d,self.encoded2d,self.encoded2d]).transpose(1, 2, 0)), (self.encoded2d.shape[0] * 24 , self.encoded2d.shape[1] * 24)), (self.obs().shape[0]*48+10,  (self.TARGET2d.shape[1]+1)*24+10))
        
         # 白い線
        pygame.draw.line(self.screen, (255,255,255), (self.obs().shape[0] * 48 + 20,10), (self.obs().shape[0] * 48 + 20,210))
        pygame.draw.line(self.screen, (255,255,255), (self.obs().shape[0] * 48 + 220,210), (self.obs().shape[0] * 48 + 20,210))

        pygame.gfxdraw.pixel(self.screen, int(self.obs().shape[0] * 48 + 120+200*self.TARGET[0]/3), int(110-200*self.TARGET[1]/3), (255,255,255))

        #pygame.gfxdraw.pixel(self.screen, int(self.obs().shape[0] * 48 + 120+200*self.encoded_obs[0]/3), int(110-200*self.encoded_obs[1]/3), (255,255,255))

        if self.reset_rend:#一度目の処理じゃない場合
            pygame.draw.line(self.screen, (255,255,255), ( int(self.obs().shape[0] * 48 + 120+200*self.encoded_obs[0]/3), int(110-200*self.encoded_obs[1]/3)), (self.befo_pixpos[0],self.befo_pixpos[1]))

        self.befo_pixpos = [int(self.obs().shape[0] * 48 + 120+200*self.encoded_obs[0]/3),int(110-200*self.encoded_obs[1]/3)]

        if not math.sqrt(np.sum((self.encoded_obs - self.TARGET) ** 2)) < self.range_calcu:
            pygame.draw.circle(self.screen, (255,255,255), (int(self.obs().shape[0] * 48 + 120+200*self.TARGET[0]/3),int(110-200*self.TARGET[1]/3)), int(200*self.range_calcu/3), 1)
        else:#中に入ると塩が赤く
            pygame.draw.circle(self.screen, (255,0,0), (int(self.obs().shape[0] * 48 + 120+200*self.TARGET[0]/3),int(110-200*self.TARGET[1]/3)), int(200*self.range_calcu/3), 1)


        #MAP上の位置
        pygame.draw.line(self.screen, (255,255,255), (self.obs().shape[0] * 48 + 20,240), (self.obs().shape[0] * 48 + 20,440))#y
        pygame.draw.line(self.screen, (255,255,255), (self.obs().shape[0] * 48 + 220,440), (self.obs().shape[0] * 48 + 20,440))#x

        #pygame.gfxdraw.pixel(self.screen, int(self.obs().shape[0] * 48 + 20+200*self.pos[0]/self.PIC.shape[0]), int(440-200*self.pos[1]/self.PIC.shape[1]), (255,255,255))
        if self.reset_rend:#一度目の処理じゃない場合
            pygame.draw.line(self.screen, (255,255,255), (int(self.obs().shape[0] * 48 + 20+200*self.pos[0]/self.PIC.shape[0]), int(440-200*self.pos[1]/self.PIC.shape[1])), (self.befo_pos[0],self.befo_pos[1]))

        self.befo_pos = [int(self.obs().shape[0] * 48 + 20+200*self.pos[0]/self.PIC.shape[0]), int(440-200*self.pos[1]/self.PIC.shape[1])]
        

        pygame.display.update()  # 画面を更新

        if not self.reset_rend:
            self.reset_rend = True

    def _close(self):
        pass

    def _seed(self, seed=None):
        pass

    def _get_reward(self):
        # 報酬を返す。報酬の与え方が難しいが、ここでは
        # - ゴールにたどり着くと 100 ポイント
        # - ダメージはゴール時にまとめて計算
        # - 1ステップごとに-1ポイント(できるだけ短いステップでゴールにたどり着きたい)
        # とした
        #print(np.max(encoded_obs),'aaaaaaaaaaaa')
        #print(encoded_obs)
        if math.sqrt(np.sum((self.encoded_obs - self.TARGET) ** 2)) < self.range_calcu:
            return 100
        else:
            return -1
        '''
        if not self._is_done():#終了してないとき
            return -1
        else:#終了時
            if math.sqrt(np.sum((self.encoded_obs - self.TARGET) ** 2)) < self.RANGE:
                return 100
            else:
                return -1'''

    def obs(self):#こっちは2D
        return self.PIC[self.pos[0]-2:self.pos[0]+3,self.pos[1]-2:self.pos[1]+3]
    
    '''
    def _observe(self):
        return np.concatenate([np.ravel(self.PIC[self.pos[0]-2:self.pos[0]+3,self.pos[1]-2:self.pos[1]+3]),self.pos],0)'''

    def obs_encoder(self):#エンコーダへの入力
        return np.ravel(self.PIC[self.pos[0]-2:self.pos[0]+3,self.pos[1]-2:self.pos[1]+3]).astype('float32') / 255.

    def _observe(self):#こっちは1D+エンコード結果
        return np.concatenate([(np.concatenate([(np.ravel(self.PIC[self.pos[0]-2:self.pos[0]+3,self.pos[1]-2:self.pos[1]+3])) / 255,self.encoded_obs],0)),self.TARGET],0)

    
    def _is_done(self):
        # 今回は最大で self.MAX_STEPS までとした
        if self.steps > self.MAX_STEPS:
            return True
        else:
            if math.sqrt(np.sum((self.encoded_obs - self.TARGET) ** 2)) < self.range_calcu:#最後のディレクトリでボタンを押す
                return True
            else:
                return False


    def auto_encoder(self):#オートエンコーダ
        ##############################################学習開始##############################################
        #モデル読み込み
        #json_string = open(os.path.join(self.model_filename)).read()
        #autoencoder = model_from_json(json_string)

        
        
        parser = argparse.ArgumentParser()
        help_ = "Load h5 model trained weights"
        parser.add_argument("-w", "--weights", help=help_)
        help_ = "Use mse loss instead of binary cross entropy (default)"
        parser.add_argument("-m",
                            "--mse",
                            help=help_, action='store_true')
        args = parser.parse_args()
        models = (self.encoder, self.decoder)
        data = (self.x_train, 1)

        # VAE loss = mse_loss or xent_loss + kl_loss
        if args.mse:
            reconstruction_loss = mse(self.inputs, self.outputs)
        else:
            reconstruction_loss = binary_crossentropy(self.inputs,
                                                    self.outputs)
        #load vae
        '''
        try:
            self.vae.load_weights(os.path.join(self.weights_filename))
        except:
            pass'''

        reconstruction_loss *= self.original_dim
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam', loss=None)


        # train the autoencoder
        self.vae.fit(self.x_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(self.x_train, None))

        self.plot_results()

        self.vae.save_weights(os.path.join(self.weights_filename))



    def plot_results(self,
                    batch_size=128,
                    model_name="vae_mnist"):
        """Plots labels and MNIST digits as a function of the 2D latent vector
        # Arguments
            models (tuple): encoder and decoder models
            data (tuple): test data and label
            batch_size (int): prediction batch size
            model_name (string): which model is using this function
        """

        y_test = 1
        os.makedirs(model_name, exist_ok=True)

        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = self.encoder.predict(self.x_train,
                                    batch_size=batch_size)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1])
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()

        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 5
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = (n - 1) * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

    



    def update_traget(self):#ターゲットアップデート
        self.TARGET = np.random.randn(self.MIDDLE_layer)#ターゲットとりあえずランダム設定
        self.TARGET_PIC = self.decoder.predict(np.squeeze(self.TARGET)[np.newaxis,:]).reshape(5, 5)*255
        #計算済みの範囲を格納
        self.range_calcu = (self.RANGE/(math.sqrt(1/(math.sqrt(2*math.pi)))*math.e**(((math.sqrt(np.sum(self.TARGET**2)))**2)/-2)))/(1/(math.sqrt(1/(math.sqrt(2*math.pi)))))

