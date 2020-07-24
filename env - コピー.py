from PIL import Image

import os

import gym
import numpy as np
import gym.spaces

class PIC(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    MAX_STEPS = 500

    PIC = np.array(Image.open('強化学習/pic_env/MAP_PIC.jpg').convert('L'))

    def __init__(self):
        super().__init__()
        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(4)  # 東西南北設置破壊
        self.observation_space = gym.spaces.Box(
            low=0,
            high=256,
            shape=(25,)
        )
        self.reward_range = [-1., 100.]
        self._reset()

    def _reset(self):
        # 諸々の変数を初期化する
        self.pos = np.array([10, 20], dtype=int)#画面の座標x,画面のディレクトリ
        self.done = False
        self.steps = 0
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

        self.action = action
        self.steps = self.steps + 1
        observation = self._observe()
        reward = self._get_reward()
        self.done = self._is_done()
        return observation, reward, self.done, {}

    def _render(self, mode='human', close=False):
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        os.system('cls')
        print(self.pos)
        print(self._get_reward())
        print(self.obs())
        '''
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n'.join(''.join(
                self.FIELD_TYPES[elem] for elem in row
                ) for row in self.obs()
            ) + '\n'
        )
        return outfile'''

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
        if not self._is_done():#終了してないとき
            return -1
        else:#終了時
            if self.pos[0] == 21 and self.pos[1] == 21:
                return 100
            else:
                return -1

    def obs(self):
        return self.PIC[self.pos[0]-2:self.pos[0]+3,self.pos[1]-2:self.pos[1]+3]
    
    '''
    def _observe(self):
        return np.concatenate([np.ravel(self.PIC[self.pos[0]-2:self.pos[0]+3,self.pos[1]-2:self.pos[1]+3]),self.pos],0)'''

    def _observe(self):
        return np.ravel(self.PIC[self.pos[0]-2:self.pos[0]+3,self.pos[1]-2:self.pos[1]+3])

    
    def _is_done(self):
        # 今回は最大で self.MAX_STEPS までとした
        if self.steps > self.MAX_STEPS:
            return True
        else:
            if self.pos[0] == 21 and self.pos[1] == 21:#最後のディレクトリでボタンを押す
                return True
            else:
                return False