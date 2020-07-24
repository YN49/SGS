import pic

import gym

import pickle
import os
import numpy as np
import random
import math

import tensorflow as tf

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import *
from keras import backend as K

import rl.core

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class PendulumProcessorForDQN(rl.core.Processor):
    def __init__(self, enable_image=False, image_size=84):
        self.image_size = image_size
        self.enable_image = enable_image
        self.mode = "train"
    
    def process_observation(self, observation):
        if not self.enable_image:
            return observation
        return self._get_rgb_state(observation)  # reshazeせずに返す
        
    def process_action(self, action):
        ACT_ID_TO_VALUE = {
            0: [-2.0], 
            1: [-1.0], 
            2: [0.0], 
            3: [+1.0],
            4: [+2.0],
        }
        return ACT_ID_TO_VALUE[action]

    def process_reward(self, reward):
        if self.mode == "test":  # testは本当の値を返す
            return reward
        # return np.clip(reward, -1., 1.)
        return reward

        # -16.5～0 を -1～1 に正規化
        self.max = 0
        self.min = -16.5
        # min max normarization
        if (self.max - self.min) == 0:
            return 0
        M = 1
        m = -0.5
        return ((reward - self.min) / (self.max - self.min))*(M - m) + m
        

    # 状態（x,y座標）から対応画像を描画する関数
    def _get_rgb_state(self, state):
        img_size = self.image_size

        h_size = img_size/2.0

        img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        dr = ImageDraw.Draw(img)

        # 棒の長さ
        l = img_size/4.0 * 3.0/ 2.0

        # 棒のラインの描写
        dr.line(((h_size - l * state[1], h_size - l * state[0]), (h_size, h_size)), (0, 0, 0), 1)

        # 棒の中心の円を描写（それっぽくしてみた）
        buff = img_size/32.0
        dr.ellipse(((h_size - buff, h_size - buff), (h_size + buff, h_size + buff)), 
                   outline=(0, 0, 0), fill=(255, 0, 0))

        # 画像の一次元化（GrayScale化）とarrayへの変換
        pilImg = img.convert("L")
        img_arr = np.asarray(pilImg)

        # 画像の規格化
        img_arr = img_arr/255.0

        return img_arr

def clipped_error_loss(y_true, y_pred):
    err = y_true - y_pred  # エラー
    L2 = 0.5 * K.square(err)
    L1 = K.abs(err) - 0.5

    # エラーが[-1,1]区間ならL2、それ以外ならL1を選択する。
    loss = tf.where((K.abs(err) < 1.0), L2, L1)   # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)

def rescaling(x, epsilon=0.001):
    n = math.sqrt(abs(x)+1) - 1
    return np.sign(x)*n + epsilon*x

def rescaling_inverse(x):
    return np.sign(x)*( (x+np.sign(x) ) ** 2 - 1)

class RainbowRAgent(rl.core.Agent):
    def __init__(self, 
        input_shape, 
        enable_image_layer,
        nb_actions,
        input_sequence=4,          # 入力フレーム数
        memory_type="replay", # 使用するメモリ
        memory_capacity=1000000,  # 確保するメモリーサイズ
        per_alpha=0.6,            # PERの確率反映率
        per_beta_initial=0.4,     # IS反映率の初期値
        per_beta_steps=1000000,       # IS反映率の上昇step数
        per_enable_is=False,     # ISを有効にするかどうか
        nb_steps_warmup=50000,    # 初期のメモリー確保用step数(学習しない)
        target_model_update=500,  # target networkのupdate間隔
        action_interval=4,  # アクションを実行する間隔
        train_interval=4,   # 学習間隔
        batch_size=32,      # batch_size
        gamma=0.99,        # Q学習の割引率
        initial_epsilon=1.0,  # ϵ-greedy法の初期値
        final_epsilon=0.1,    # ϵ-greedy法の最終値
        exploration_steps=1000000,  # ϵ-greedy法の減少step数
        multireward_steps=3,  # multistep reward
        dence_units_num=512,  # Dence層のユニット数

        enable_double_dqn=False,
        enable_dueling_network=False,
        dueling_network_type="ave",
        enable_noisynet=False,

        lstm_type="",        # 使用するLSTMアルゴリズム
        lstm_units_num=512,  # LSTMのユニット数
        priority_exponent=0.9,   # シーケンス長priorityを計算する際のη
        enable_rescaling_priority=False,  # rescalingを有効にするか(priotiry)
        enable_rescaling_train=False,     # rescalingを有効にするか(学習)
        rescaling_epsilon=0.001,  # rescalingの定数
        burnin_length=0,     # burnin期間

        **kwargs):
        super(RainbowRAgent, self).__init__(**kwargs)
        self.compiled = False

        self.input_shape = input_shape
        self.enable_image_layer = enable_image_layer
        self.nb_actions = nb_actions
        self.input_sequence = input_sequence
        self.nb_steps_warmup = nb_steps_warmup
        self.target_model_update = target_model_update
        self.action_interval = action_interval
        self.train_interval = train_interval
        self.gamma = gamma
        self.batch_size = batch_size
        self.multireward_steps = multireward_steps
        self.dence_units_num = dence_units_num

        self.lstm_units_num = lstm_units_num
        self.enable_rescaling_priority = enable_rescaling_priority
        self.enable_rescaling_train = enable_rescaling_train
        self.rescaling_epsilon = rescaling_epsilon
        self.priority_exponent = priority_exponent
        self.lstm_type = lstm_type

        # type チェック
        lstm_types = [
            "",
            "lstm",
            "lstm_ful",
        ]
        if self.lstm_type not in lstm_types:
            raise ValueError('lstm_type is ["","lstm","lstm_ful"]')

        # lstm_ful のみburnin有効
        if self.lstm_type == "lstm_ful":
            self.burnin_length = burnin_length
        else:
            self.burnin_length = 0

        self.initial_epsilon = initial_epsilon  
        self.epsilon_step = (initial_epsilon - final_epsilon) / exploration_steps
        self.final_epsilon = final_epsilon
        
        self.per_alpha = per_alpha
        if memory_type == "replay":
            self.memory = ReplayMemory(memory_capacity)
        elif memory_type == "per_greedy":
            self.memory = PERGreedyMemory(memory_capacity)
        elif memory_type == "per_proportional":
            self.memory = PERProportionalMemory(memory_capacity, per_beta_initial, per_beta_steps, per_enable_is)
        elif memory_type == "per_rankbase":
            self.memory = PERRankBaseMemory(memory_capacity, per_alpha, per_beta_initial, per_beta_steps, per_enable_is)
        else:
            raise ValueError('memory_type is ["replay","per_proportional","per_rankbase"]')

        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_network_type = dueling_network_type
        self.enable_noisynet = enable_noisynet

        dueling_network_types = [
            "ave",
            "max",
            "naive",
        ]
        if self.dueling_network_type not in dueling_network_types:
            raise ValueError('dueling_network_type is ["ave","max","naive"]')
        self.dueling_network_type = dueling_network_type

        self.model = self.build_network()         # Q network
        self.target_model = self.build_network()  # target network
        
        assert memory_capacity > self.batch_size, "Memory capacity is small.(Larger than batch size)"
        assert self.nb_steps_warmup > self.batch_size, "Warmup steps is few.(Larger than batch size)"
        
    def reset_states(self):
        self.repeated_action = 0

        self.recent_action = [ 0 for _ in range(self.input_sequence)]
        self.recent_reward = [ 0 for _ in range(self.input_sequence + self.multireward_steps - 1)]
        obs_length = self.burnin_length + self.input_sequence + self.multireward_steps
        self.recent_observations = [np.zeros(self.input_shape) for _ in range(obs_length)]

        if self.lstm_type == "lstm_ful":
            self.model.reset_states()
            self.recent_hidden_state = [
                [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])]
                for _ in range(self.burnin_length + self.input_sequence)
            ]

    # NNモデルの作成
    def build_network(self):

        if self.lstm_type == "lstm_ful":
            # (batch_size, timesteps, width, height)
            c = input_ = Input(batch_shape=(1, 1) + self.input_shape)
        else:
            # 入力層(input_sequence, width, height)
            c = input_ = Input(shape=(self.input_sequence,) + self.input_shape)

        if self.enable_image_layer:
            if self.lstm_type == "":
                c = Permute((2, 3, 1))(c)  # (window,w,h) -> (w,h,window)

                c = Conv2D(32, (8, 8), strides=(4, 4), padding="same", name="c1")(c)
                c = Activation("relu")(c)
                c = Conv2D(64, (4, 4), strides=(2, 2), padding="same", name="c2")(c)
                c = Activation("relu")(c)
                c = Conv2D(64, (3, 3), strides=(1, 1), padding="same", name="c3")(c)
                c = Activation("relu")(c)
                c = Flatten()(c)
            else:  #lstm

                # (time steps, w, h) -> (time steps, w, h, ch)
                if self.lstm_type == "lstm_ful":
                    c = Reshape((1, ) + self.input_shape + (1,) )(c)
                else:
                    c = Reshape((self.input_sequence, ) + self.input_shape + (1,) )(c)
            
                # https://keras.io/layers/wrappers/
                c = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), padding="same"), name="c1")(c)
                c = Activation("relu")(c)
                c = TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), padding="same"), name="c2")(c)
                c = Activation("relu")(c)
                c = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), padding="same"), name="c3")(c)
                c = Activation("relu")(c)
                c = TimeDistributed(Flatten())(c)
            
        elif self.lstm_type == "":
            c = Flatten()(c)
        
        if self.lstm_type == "lstm":
            c = LSTM(self.lstm_units_num, name="lstm")(c)
        elif self.lstm_type == "lstm_ful":
            c = LSTM(self.lstm_units_num, stateful=True, name="lstm")(c)

        if self.enable_dueling_network:

            # value
            v = Dense(self.dence_units_num, activation="relu")(c)
            if self.enable_noisynet:
                v = NoisyDense(1, name="v")(v)
            else:
                v = Dense(1, name="v")(v)

            # advance
            adv = Dense(self.dence_units_num, activation='relu')(c)
            if self.enable_noisynet:
                adv = NoisyDense(self.nb_actions, name="adv")(adv)
            else:
                adv = Dense(self.nb_actions, name="adv")(adv)

            # 連結で結合
            c = Concatenate()([v,adv])
            if self.dueling_network_type == "ave":
                c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(self.nb_actions,))(c)
            elif self.dueling_network_type == "max":
                c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True), output_shape=(self.nb_actions,))(c)
            elif self.dueling_network_type == "naive":
                c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(self.nb_actions,))(c)

        else:
            c = Dense(self.dence_units_num, activation="relu")(c)
            if self.enable_noisynet:
                c = NoisyDense(self.nb_actions, activation="linear", name="adv")(c)
            else:
                c = Dense(self.nb_actions, activation="linear", name="adv")(c)
        
        return Model(input_, c)


    def compile(self, optimizer=None, metrics=[]):
        # target networkは更新がないので optimizerとlossは何でもいい
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(loss=clipped_error_loss, optimizer=optimizer, metrics=metrics)

        # lstm ful では lstmレイヤーを使う
        if self.lstm_type == "lstm_ful":
            self.lstm = self.model.get_layer("lstm")
            self.target_lstm = self.target_model.get_layer("lstm")

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.target_model.load_weights(filepath)

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def forward(self, observation):
        # windowサイズ分observationを保存する
        self.recent_observations.append(observation)  # 最後に追加
        self.recent_observations.pop(0)  # 先頭を削除

        # 学習(次の状態が欲しいのでforwardで学習)
        self.forward_train()

        # フレームスキップ(action_interval毎に行動を選択する)
        action = self.repeated_action
        if self.step % self.action_interval == 0:

            if self.lstm_type == "lstm_ful":
                # 状態を復元
                self.lstm.reset_states(self.recent_hidden_state[-1])
            
            # 行動を決定
            action = self.select_action()

            if self.lstm_type == "lstm_ful":
                # 状態を保存
                self.recent_hidden_state.append([K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])])
                self.recent_hidden_state.pop(0)

            # リピート用
            self.repeated_action = action
        
        self.recent_action.append(action)  # 最後に追加
        self.recent_action.pop(0)  # 先頭を削除
        return action

    # 長いので関数に
    def select_action(self):
        # noisy netが有効の場合はそちらで探索する
        if self.training and not self.enable_noisynet:
            
            # ϵ をstepで減少。
            epsilon = self.initial_epsilon - self.step*self.epsilon_step
            if epsilon < self.final_epsilon:
                epsilon = self.final_epsilon
            
            # ϵ-greedy法
            if epsilon > np.random.uniform(0, 1):
                # ランダム
                action = np.random.randint(0, self.nb_actions)
            else:
                action = self._get_qmax_action()
        else:
            action = self._get_qmax_action()

        return action

    # 2箇所あるので関数に、現状の最大Q値のアクションを返す
    def _get_qmax_action(self):

        if self.lstm_type == "lstm_ful":
            # 最後の状態のみ
            state1 = [self.recent_observations[-1]]
            q_values = self.model.predict(np.asarray([state1]), batch_size=1)[0]
        else:
            # sequence分の入力
            state1 = self.recent_observations[-self.input_sequence:]
            q_values = self.model.predict(np.asarray([state1]), batch_size=1)[0]

        return np.argmax(q_values)


    # 長いので関数に
    def forward_train(self):
        if not self.training:
            return

        if self.lstm_type == "lstm_ful":
            # Multi-Step learning
            rewards = []
            for i in range(self.input_sequence):
                r = 0
                for j in range(self.multireward_steps):
                    r += self.recent_reward[i+j] * (self.gamma ** j)
                rewards.append(r)
            
            self.memory.add((
                self.recent_observations[:],
                self.recent_action[:],
                rewards,
                self.recent_hidden_state[0]
            ))
        else:
            # Multi-Step learning
            reward = 0
            for i, r in enumerate(self.recent_reward):
                reward += r * (self.gamma ** i)
            
            state0 = self.recent_observations[self.burnin_length:self.burnin_length+self.input_sequence]
            state1 = self.recent_observations[-self.input_sequence:]
            
            self.memory.add((
                state0, 
                self.recent_action[-1],
                reward,
                state1
            ))

        # ReplayMemory確保のため一定期間学習しない。
        if self.step <= self.nb_steps_warmup:
            return

        # 学習の更新間隔
        if self.step % self.train_interval != 0:
            return

        # memory から優先順位に基づき状態を取得
        (indexes, batchs, weights) = self.memory.sample(self.batch_size, self.step)

        # 学習(長いので関数化)
        if self.lstm_type == "lstm_ful":
            self.train_model_ful(indexes, batchs, weights)
        else:
            self.train_model(indexes, batchs, weights)

    # ノーマルの学習
    def train_model(self, indexes, batchs, weights):
        state0_batch = []
        action_batch = []
        reward_batch = []
        state1_batch = []
        for batch in batchs:
            state0_batch.append(batch[0])
            action_batch.append(batch[1])
            reward_batch.append(batch[2])
            state1_batch.append(batch[3])

        # 更新用に現在のQネットワークを出力(Q network)
        outputs = self.model.predict(np.asarray(state0_batch), self.batch_size)

        if self.enable_double_dqn:
            # TargetNetworkとQNetworkのQ値を出す
            state1_model_qvals_batch = self.model.predict(np.asarray(state1_batch), self.batch_size)
            state1_target_qvals_batch = self.target_model.predict(np.asarray(state1_batch), self.batch_size)
        else:
            # 次の状態のQ値を取得(target_network)
            target_qvals = self.target_model.predict(np.asarray(state1_batch), self.batch_size)

        for i in range(self.batch_size):
            if self.enable_double_dqn:
                action = np.argmax(state1_model_qvals_batch[i])  # modelからアクションを出す
                maxq = state1_target_qvals_batch[i][action]  # Q値はtarget_modelを使って出す
            else:
                maxq = np.max(target_qvals[i])

            # priority計算
            if self.enable_rescaling_priority:
                tmp = rescaling_inverse(maxq)
            else:
                tmp = maxq
            tmp = reward_batch[i] + (self.gamma ** self.multireward_steps) * tmp
            tmp *= weights[i]
            if self.enable_rescaling_priority:
                tmp = rescaling(tmp, self.rescaling_epsilon)
            priority = abs(tmp - outputs[i][action_batch[i]]) ** self.per_alpha

            # Q値の更新
            if self.enable_rescaling_train:
                maxq = rescaling_inverse(maxq)
            td_error = reward_batch[i] + (self.gamma ** self.multireward_steps) * maxq
            td_error *= weights[i]
            if self.enable_rescaling_train:
                td_error = rescaling(td_error, self.rescaling_epsilon)
            outputs[i][action_batch[i]] = td_error

            # priorityを更新
            self.memory.update(indexes[i], batchs[i], priority)

        # 学習
        self.model.train_on_batch(np.asarray(state0_batch), np.asarray(outputs))
    
    # ステートフルLSTMの学習
    def train_model_ful(self, indexes, batchs, weights):

        # 各経験毎に処理を実施
        for batch_i, batch in enumerate(batchs):
            states = batch[0]
            action = batch[1]
            reward = batch[2]
            hidden_state = batch[3]
            prioritys = []

            # burn-in
            self.lstm.reset_states(hidden_state)
            for i in range(self.burnin_length):
                self.model.predict(np.asarray([[states[i]]]), 1)
            # burn-in 後の結果を保存
            hidden_state = [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])]
        
            # 以降は1sequenceずつ更新させる
            for i in range(self.input_sequence):
                state0 = [states[self.burnin_length + i]]
                state1 = [states[self.burnin_length + i + self.multireward_steps]]

                # 現在のQネットワークを出力
                self.lstm.reset_states(hidden_state)
                output = self.model.predict(np.asarray([state0]), 1)[0]

                # TargetネットワークとQネットワークの値を出力
                if self.enable_double_dqn:
                    self.lstm.reset_states(hidden_state)
                    self.target_lstm.reset_states(hidden_state)
                    state1_model_qvals = self.model.predict(np.asarray([state1]), 1)[0]
                    state1_target_qvals = self.target_model.predict(np.asarray([state1]), 1)[0]

                    action_q = np.argmax(state1_model_qvals)
                    maxq = state1_target_qvals[action_q]

                else:
                    self.target_lstm.reset_states(hidden_state)
                    target_qvals = self.target_model.predict(np.asarray([state1], 1))[0]
                    maxq = np.max(target_qvals)

                # priority計算
                if self.enable_rescaling_priority:
                    tmp = rescaling_inverse(maxq)
                else:
                    tmp = maxq
                tmp = reward[i] + (self.gamma ** self.multireward_steps) * tmp
                tmp *= weights[batch_i]
                if self.enable_rescaling_priority:
                    tmp = rescaling(tmp, self.rescaling_epsilon)
                priority = abs(tmp - output[action[i]]) ** self.per_alpha
                prioritys.append(priority)
                
                # Q値 update用
                if self.enable_rescaling_train:
                    maxq = rescaling_inverse(maxq)
                td_error = reward[i] + (self.gamma ** self.multireward_steps) * maxq
                td_error *= weights[batch_i]
                if self.enable_rescaling_train:
                    td_error = rescaling(td_error, self.rescaling_epsilon)
                output[action[i]] = td_error

                # 学習
                self.lstm.reset_states(hidden_state)
                self.model.fit(
                    np.asarray([state0]), 
                    np.asarray([output]), 
                    batch_size=1, 
                    epochs=1, 
                    verbose=0, 
                    shuffle=False
                )

                # 次の学習用に hidden state を保存
                hidden_state = [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])]

            # 今回使用したsamplingのpriorityを更新
            priority = self.priority_exponent * np.max(prioritys) + (1-self.priority_exponent) * np.average(prioritys)
            self.memory.update(indexes[batch_i], batch, priority)
        

    def backward(self, reward, terminal):
        
        self.recent_reward.append(reward)  # 最後に追加
        self.recent_reward.pop(0)  # 先頭を削除
        
        # 一定間隔でtarget modelに重さをコピー
        if self.step % self.target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        return []

    @property
    def layers(self):
        return self.model.layers[:]



class ReplayMemory():
    def __init__(self, capacity):
        self.capacity= capacity
        self.index = 0
        self.memory = []

    def add(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = experience
        self.index = (self.index + 1) % self.capacity

    def update(self, idx, experience, priority):
        pass

    def sample(self, batch_size, steps):
        batchs = random.sample(self.memory, batch_size)

        indexes = np.empty(batch_size, dtype='float32')
        weights = [ 1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

import heapq
class _head_wrapper():
    def __init__(self, data):
        self.d = data
    def __eq__(self, other):
        return True

class PERGreedyMemory():
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

        self.max_priority = 1
        
    def add(self, experience):
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は最後の要素を削除
            self.buffer.pop()
        
        # priority は最初は最大を選択
        experience = _head_wrapper(experience)
        heapq.heappush(self.buffer, (-self.max_priority, experience))

    def update(self, idx, experience, priority):
        # heapqは最小値を出すためマイナス
        experience = _head_wrapper(experience)
        heapq.heappush(self.buffer, (-priority, experience))

        # 最大値を更新
        if self.max_priority < priority:
            self.max_priority = priority
    
    def sample(self, batch_size, step):
        # 取り出す(学習後に再度追加)
        batchs = [heapq.heappop(self.buffer)[1].d for _ in range(batch_size)]

        indexes = np.empty(batch_size, dtype='float32')
        weights = [ 1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

#copy from https://github.com/jaromiru/AI-blog/blob/5aa9f0b/SumTree.py
import numpy

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class PERProportionalMemory():
    def __init__(self, capacity, beta_initial, beta_steps, enable_is):
        self.capacity = capacity
        self.tree = SumTree(capacity)

        self.beta_initial = beta_initial
        self.beta_steps = beta_steps
        self.enable_is = enable_is
        
        self.max_priority = 1

    def add(self, experience):
        self.tree.add(self.max_priority, experience)

    def update(self, index, experience, priority):
        self.tree.update(index, priority)

        if self.max_priority < priority:
            self.max_priority = priority

    def sample(self, batch_size, step):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')

        if self.enable_is:
            # βは最初は低く、学習終わりに1にする
            beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
    
        # 合計を均等に割り、その範囲内からそれぞれ乱数を出す。
        total = self.tree.total()
        section = total / batch_size
        for i in range(batch_size):
            r = section*i + random.random()*section
            (idx, priority, experience) = self.tree.get(r)

            indexes.append(idx)
            batchs.append(experience)

            if self.enable_is:
                # 重要度サンプリングを計算
                weights[i] = (self.capacity * priority / total) ** (-beta)
            else:
                weights[i] = 1  # 無効なら1

        if self.enable_is:
            # 安定性の理由から最大値で正規化
            weights = weights / weights.max()

        return (indexes ,batchs, weights)


import bisect
class _bisect_wrapper():
    def __init__(self, data):
        self.d = data
        self.priority = 0
        self.p = 0
    def __lt__(self, o):  # a<b
        return self.priority > o.priority

class PERRankBaseMemory():
    def __init__(self, capacity, alpha, beta_initial, beta_steps, enable_is):
        self.capacity = capacity
        self.buffer = []
        self.alpha = alpha
        
        self.beta_initial = beta_initial
        self.beta_steps = beta_steps
        self.enable_is = enable_is

        self.max_priority = 1

    def add(self, experience):
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は最後の要素を削除
            self.buffer.pop()
        
        experience = _bisect_wrapper(experience)
        experience.priority = self.max_priority
        bisect.insort(self.buffer, experience)

    def update(self, index, experience, priority):
        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)

        if self.max_priority < priority:
            self.max_priority = priority

    def sample(self, batch_size, step):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')

        if self.enable_is:
            # βは最初は低く、学習終わりに1にする。
            beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps

        total = 0
        for i, o in enumerate(self.buffer):
            o.index = i
            o.p = (len(self.buffer) - i) ** self.alpha 
            total += o.p
            o.p_total = total

        # 合計を均等に割り、その範囲内からそれぞれ乱数を出す。
        index_lst = []
        section = total / batch_size
        rand = []
        for i in range(batch_size):
            rand.append(section*i + random.random()*section)
        
        rand_i = 0
        for i in range(len(self.buffer)):
            if rand[rand_i] < self.buffer[i].p_total:
                index_lst.append(i)
                rand_i += 1
                if rand_i >= len(rand):
                    break

        for i, index in enumerate(reversed(index_lst)):
            o = self.buffer.pop(index)  # 後ろから取得するのでindexに変化なし
            batchs.append(o.d)
            indexes.append(index)

            if self.enable_is:
                # 重要度サンプリングを計算
                priority = o.p
                weights[i] = (self.capacity * priority / total) ** (-beta)
            else:
                weights[i] = 1  # 無効なら1

        if self.enable_is:
            # 安定性の理由から最大値で正規化
            weights = weights / weights.max()

        return (indexes, batchs, weights)


# from : https://github.com/LuEE-C/Noisy-A3C-Keras/blob/master/NoisyDense.py
# from : https://github.com/keiohta/tf2rl/blob/atari/tf2rl/networks/noisy_dense.py
class NoisyDense(Layer):

    def __init__(self, units,
                 sigma_init=0.02,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.sigma_init = sigma_init
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        self.kernel_shape = tf.constant((self.input_dim, self.units))
        self.bias_shape = tf.constant((self.units,))

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.sigma_kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=initializers.Constant(value=self.sigma_init),
                                      name='sigma_kernel'
                                      )


        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.sigma_bias = self.add_weight(shape=(self.units,),
                                        initializer=initializers.Constant(value=self.sigma_init),
                                        name='sigma_bias')
        else:
            self.bias = None
            self.epsilon_bias = None

        self.epsilon_kernel = K.zeros(shape=(self.input_dim, self.units))
        self.epsilon_bias = K.zeros(shape=(self.units,))

        self.sample_noise()
        super(NoisyDense, self).build(input_shape)


    def call(self, X):
        #perturbation = self.sigma_kernel * self.epsilon_kernel
        #perturbed_kernel = self.kernel + perturbation
        perturbed_kernel = self.sigma_kernel * K.random_uniform(shape=self.kernel_shape)

        output = K.dot(X, perturbed_kernel)
        if self.use_bias:
            #bias_perturbation = self.sigma_bias * self.epsilon_bias
            #perturbed_bias = self.bias + bias_perturbation
            perturbed_bias = self.bias + self.sigma_bias * K.random_uniform(shape=self.bias_shape)
            output = K.bias_add(output, perturbed_bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def sample_noise(self):
        K.set_value(self.epsilon_kernel, np.random.normal(0, 1, (self.input_dim, self.units)))
        K.set_value(self.epsilon_bias, np.random.normal(0, 1, (self.units,)))

    def remove_noise(self):
        K.set_value(self.epsilon_kernel, np.zeros(shape=(self.input_dim, self.units)))
        K.set_value(self.epsilon_bias, np.zeros(shape=self.units,))

#----------------------
# NN可視化用
#----------------------
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import cv2

class ObservationLogger(rl.callbacks.Callback):
    def __init__(self):
        self.observations = []

    def on_step_end(self, step, logs):
        self.observations.append(logs["observation"])

agent = None
logger = None

def grad_cam(c_output, c_val, img, shape):
    global agent
    if agent.lstm_type == "":
        c_output = c_output[0]
        c_val = c_val[0]
    else:
        c_output = c_output[0][-1]
        c_val = c_val[0][-1]
    weights = np.mean(c_val, axis=(0, 1))
    cam = np.dot(c_output, weights)
    cam = cv2.resize(cam, shape, cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    rate = 0.4
    cam = cv2.addWeighted(src1=img, alpha=(1-rate), src2=cam, beta=rate, gamma=0)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    return cam


def plot(frame):
    if frame % 50 == 0:  # debug
        print(frame)

    global agent, logger
    observations = logger.observations
    input_sequence = agent.input_sequence
    model = agent.model

    # 入力分の frame がたまるまで待つ
    if frame < input_sequence:
        return

    # 入力用の変数を作成
    # 入力は input_sequence の長さ分必要(DQN編を参照)
    input_state = observations[frame - input_sequence:frame]

    # ついでに shape も取得
    shape = np.asarray(observations[0]).shape

    # 出力用のオリジナル画像を作成
    # 形式は(w,h)でかつ0～1で正規化されているので画像形式に変換
    img = np.asarray(observations[frame])  # (w,h)
    img *= 255
    img = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2BGR)  # (w,h) -> (w,h,3)

    c1_output = model.get_layer("c1").output
    c2_output = model.get_layer("c2").output
    c3_output = model.get_layer("c3").output
    if agent.enable_dueling_network:
        v_output = model.get_layer("v").output
    adv_output = model.get_layer("adv").output

    # 予測結果を出す
    prediction = model.predict(np.asarray([input_state]), 1)[0]
    class_idx = np.argmax(prediction)
    class_output = model.output[0][class_idx]

    # 各勾配を定義
    # adv層は出力と同じ(action数)なので予測結果を指定
    # v層はUnit数が1つしかないので0を指定
    grads_c1 = K.gradients(class_output, c1_output)[0]
    grads_c2 = K.gradients(class_output, c2_output)[0]
    grads_c3 = K.gradients(class_output, c3_output)[0]
    if agent.enable_dueling_network:
        grads_v = K.gradients(v_output[0][0], model.input)[0]
    grads_adv = K.gradients(adv_output[0][class_idx], model.input)[0]

    # functionを定義、１度にすべて計算
    if agent.enable_dueling_network:
        grads_func = K.function([model.input, K.learning_phase()],
            [c1_output, grads_c1, c2_output, grads_c2, c3_output, grads_c3, grads_adv, grads_v])

        # 勾配を計算
        (c1_output, c1_val, c2_output, c2_val, c3_output, c3_val, adv_val, v_val) = grads_func([np.asarray([input_state]), 0])
        adv_val = adv_val[0][input_sequence-1]
        v_val = v_val[0][input_sequence-1]

        # SaliencyMap
        adv_val = np.abs(adv_val.reshape(shape))
        v_val = np.abs(v_val.reshape(shape))

        # Grad-CAMの計算と画像化、3回も書きたくないので関数化
        cam1 = grad_cam(c1_output, c1_val, img, shape)
        cam2 = grad_cam(c2_output, c2_val, img, shape)
        cam3 = grad_cam(c3_output, c3_val, img, shape)

        imgs = [img, cam1, cam2, cam3, adv_val, v_val]
        names = ["original", "c1", "c2", "c3", "advance", "value"]
        cmaps = ["", "", "", "", "gray", "gray"]

    else:
        grads_func = K.function([model.input, K.learning_phase()],
            [c1_output, grads_c1, c2_output, grads_c2, c3_output, grads_c3, grads_adv])
        
        # 勾配を計算
        (c1_output, c1_val, c2_output, c2_val, c3_output, c3_val, adv_val) = grads_func([np.asarray([input_state]), 0])
        adv_val = adv_val[0][input_sequence-1]

        # SaliencyMap
        adv_val = np.abs(adv_val.reshape(shape))

        # Grad-CAMの計算と画像化、3回も書きたくないので関数化
        cam1 = grad_cam(c1_output, c1_val, img, shape)
        cam2 = grad_cam(c2_output, c2_val, img, shape)
        cam3 = grad_cam(c3_output, c3_val, img, shape)

        imgs = [img, cam1, cam2, cam3, adv_val]
        names = ["original", "c1", "c2", "c3", "advance"]
        cmaps = ["", "", "", "", "gray"]
    

    # plot
    for i in range(len(imgs)):
        plt.subplot(2, 3, i+1)
        plt.gca().tick_params(labelbottom="off",bottom="off") # x軸の削除
        plt.gca().tick_params(labelleft="off",left="off") # y軸の削除
        plt.title(names[i]).set_fontsize(12)
        if cmaps[i] == "":
            plt.imshow(imgs[i])
        else:
            plt.imshow(imgs[i], cmap=cmaps[i])



#-----------------------------------------------------
# ムービー用
#-----------------------------------------------------

import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
class MovieLogger(rl.callbacks.Callback):
    def __init__(self):
        self.frames = []
        self.history = []

    def on_action_end(self, action, logs):
        self.frames.append(self.env.render(mode='rgb_array'))
        
        
    def on_step_end(self, step, logs):
        self.history.append(logs)

    #-----------------------
    def view(self, interval=10, start_frame=0, end_frame=0, gifname="", mp4name=""):
        assert start_frame<len(self.frames), "start frame is over frames({})".format(len(self.frames))
        if end_frame == 0:
          end_frame = len(self.frames)
        elif end_frame > len(self.frames):
            end_frame = len(self.frames)
        self.start_frame = start_frame
        self.t0 = time.time()
        
        self.patch = plt.imshow(self.frames[0])
        plt.axis('off')
        ani = matplotlib.animation.FuncAnimation(plt.gcf(), self._plot, frames=end_frame - start_frame, interval=interval)

        if gifname != "":
            #ani.save(gifname, writer="pillow", fps=5)
            ani.save(gifname, writer="imagemagick", fps=60)
        if mp4name != "":
            ani.save(mp4name, writer="ffmpeg")
        #plt.show()
    
    def _plot(self, frame):
        if frame % 50 == 0:
            print("{}f {}m".format(frame, (time.time()-self.t0)/60))
        
        #plt.imshow(self.frames[frame + self.start_frame])
        self.patch.set_data(self.frames[frame + self.start_frame])


#-----------------------------------------------------------
# main    
#-----------------------------------------------------------
def main(image=False, lstm_type="lstm"):
    global agent, logger

    env = gym.make("pic-v0")
    nb_actions = 5  # PendulumProcessorで5個と定義しているので5

    if image:
        processor = PendulumProcessorForDQN(enable_image=True, image_size=84)
        input_shape = (84, 84)
    else:
        processor = PendulumProcessorForDQN(enable_image=False)
        input_shape = env.observation_space.shape

    # 引数が多いので辞書で定義して渡しています。
    args={
        "input_shape": input_shape, 
        "enable_image_layer": image, 
        "nb_actions": nb_actions, 
        "input_sequence": 4,         # 入力フレーム数
        "memory_capacity": 1_000_000,  # 確保するメモリーサイズ
        "nb_steps_warmup": 200,     # 初期のメモリー確保用step数(学習しない)
        "target_model_update": 500, # target networkのupdate間隔
        "action_interval": 1,  # アクションを実行する間隔
        "train_interval": 1,   # 学習する間隔
        "batch_size": 16,   # batch_size
        "gamma": 0.99,     # Q学習の割引率
        "initial_epsilon": 1.0,  # ϵ-greedy法の初期値
        "final_epsilon": 0.01,    # ϵ-greedy法の最終値
        "exploration_steps": 10000,  # ϵ-greedy法の減少step数
        "processor": processor,

        "memory_type": "per_proportional",  # メモリの種類
        "per_alpha": 0.8,            # PERの確率反映率
        "per_beta_initial": 0.0,     # IS反映率の初期値
        "per_beta_steps": 5000,   # IS反映率の上昇step数
        "per_enable_is": False,      # ISを有効にするかどうか
        "multireward_steps": 1,    # multistep reward
        "enable_double_dqn": True,
        "enable_dueling_network": True,
        "dueling_network_type": "ave",  # dueling networkで使うアルゴリズム
        "enable_noisynet": False,
        "dence_units_num": 64,    # Dence層のユニット数

        # 今回追加分
        "lstm_type": "",
        "lstm_units_num": 64,
        "priority_exponent": 0.9,   # priority優先度
        "enable_rescaling_priority": True,   # rescalingを有効にするか(priotrity)
        "enable_rescaling_train": True,      # rescalingを有効にするか(train)
        "rescaling_epsilon": 0.001,  # rescalingの定数
        "burnin_length": 40,        # burn-in期間
    }

    if lstm_type == "lstm":
        args["lstm_type"] = "lstm"
    elif lstm_type == "lstm_ful":
        args["lstm_type"] = "lstm_ful"
        args["batch_size"] = 1

    agent = RainbowRAgent(**args)
    agent.compile(optimizer=Adam(lr=0.0002))
    print(agent.model.summary())

    # 訓練
    print("--- start ---")
    print("'Ctrl + C' is stop.")
    history = agent.fit(env, nb_steps=50_000, visualize=False, verbose=1)
    weights_file = "lstm_weight.h5"
    agent.save_weights(weights_file, overwrite=True)
    agent.load_weights(weights_file)

    # 結果を表示
    plt.subplot(1,1,1)
    plt.plot(history.history["episode_reward"])
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()

    # 訓練結果を見る
    processor.mode = "test"  # env本来の報酬を返す
    agent.test(env, nb_episodes=5, visualize=True)
    view = MovieLogger()   # 動画用
    logger = ObservationLogger()
    agent.test(env, nb_episodes=1, visualize=False, callbacks=[logger,view])
    view.view(interval=1, gifname="anim1.gif")  # 動画用

    #--- NNの可視化
    if image:
        plt.figure(figsize=(8.0, 6.0), dpi = 100)  # 大きさを指定
        plt.axis('off')
        ani = matplotlib.animation.FuncAnimation(plt.gcf(), plot, frames=150, interval=5)
        #ani = matplotlib.animation.FuncAnimation(plt.gcf(), plot, frames=len(logger.observations), interval=5)

        #ani.save('anim2.mp4', writer="ffmpeg")
        ani.save('anim2.gif', writer="imagemagick", fps=60)
        #plt.show()


    

# コメントアウトで切り替え
#main(image=False, lstm_type="")
main(image=False, lstm_type="lstm")
#main(image=False, lstm_type="lstm_ful")