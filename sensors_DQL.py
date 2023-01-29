from typing import List, Any, Union

import pyodbc  # to read table from sql
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pickle  # for saving model
from joblib import Parallel, delayed  # for saving model
import joblib  # for saving model
from numba import jit, cuda  # for running on gpu
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import time as time
# Guide Notes:
# This is a DQL (Deep Q Learning) agent, it has a neural net to predict Q values for actions in states
# notice that DQL is different from DQN(Deep Q Network) which consists of a neural net and a target net.
# each record of input data is like:
# normalizedTime  V21  V22  V23  S137  S138  S139  S140  B5 
# where normalizedTime is normal time of event and other columns are sensor values in that timestamp
# and last column (i.e B5 ) is label(i.e it's value must be predicted by our algorithm)

# model predicts the action index

from sklearn.model_selection import train_test_split
from tensorflow import keras

EPISODES = 100  # gradually change it to 1000. must be large enough to be trained

name = "dqn_in7_out1"
SaveLocation = "savedModel/dqn_in7_out1.h5"  # max reward is 18 of 20
train_start_column = 1  # for train86 set 4   for train7 set 1
test_start_column = 3  # for train86 set     for test7 set 3


class DQNAgent:
    def __init__(self, state_size, action_size: int, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.1  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
        self.learning_rate = 0.0005  # 0.01 - 0.00001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(5, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(2, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        # model.compile(loss='sparse_categorical_crossentropy',
        #               optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate), )
        model.compile(
            # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=[
                # tf.metrics.SparseCategoricalAccuracy()
                tf.metrics.CategoricalAccuracy()
            ],
        )
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # returns the action index.
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # print(".")
            return random.randrange(self.action_size)  # randrange is from [start - last)

        act_values = self.model.predict(state)

        # act_values is ndarray (1,7) like:  [[0,0,0,0,0,0,0,0]]
        # print("np.argmax(act_values[0]): ", np.argmax(act_values[0])) # argmax gives index of max value
        return np.argmax(act_values[0])  # returns action index

    # this is the train part
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path_weights):
        self.model.load_weights(path_weights)
        # newModel = keras.models.load_model(path_model)
        # newModel.load_weights(path_weights)
        # return newModel

    def save(self, path_weights):
        # self.model.save(path_model)
        self.model.save_weights(path_weights)

    def setEpsilon(self, eps):
        self.epsilon = eps


class MySmartHome:

    def __init__(self):
        self.current_record_index = 0
        self.action_space = [[0], [1]]
        # I you have more actuators to predict:
        # self.action_space = [[0, 0, 0],
        #                      [0, 0, 1],
        #                      [0, 1, 0],
        #                      [1, 0, 0],
        #                      [0, 1, 1],
        #                      # [1, 1, 0],
        #                      # [1, 0, 1],
        #                      [1, 1, 1],
        #                      ]
        # states_labels is the whole records and columns
        self.states_labels: pd.DataFrame = self._get_sensors_data_from_sql()
        self.states_labels_test: pd.DataFrame = self._get_test_data_from_sql()

        actuators_count = 1
        self.states_labels: pd.DataFrame = self.states_labels.iloc[0:, train_start_column:]  # select from second column

        self.all_states: pd.DataFrame = self.states_labels.iloc[:, :-actuators_count]  # igonore last 3 columns
        self.all_labels: pd.DataFrame = self.states_labels.iloc[:, -actuators_count:]  # last 3 columns are labels

        self.states_labels = self.states_labels.apply(pd.to_numeric)

        self.all_states_test: pd.DataFrame = self.states_labels_test.iloc[:,
                                             test_start_column:-actuators_count]  # = self.x_test
        self.all_labels_test: pd.DataFrame = self.states_labels_test.iloc[:, -actuators_count:]  # = self.y_test
        # in case of using 7 sensors, state_size is 8(7 sensor and a timestamp)
        # ignore last 1 columns for state since they are labels.
        self.state_size = self.all_states.columns.__len__()  # might be 7-8  86-87
        # print("test 4s:", self.all_states_test.head(10))
        # print("test 4l:", self.all_labels_test.head(10))

    def _get_test_data_from_sql(self) -> pd.DataFrame:
        server = '.'
        database = 'Your Database Name'
        username = 'Database username'
        password = 'Database password'
        cnxn = pyodbc.connect(
            'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
        cursor = cnxn.cursor()
        # query = "sp_getRandomTests86"
        query = "sp_getRandomTests7"
        df = pd.read_sql(query, cnxn)
        # data initialization done
        # print("Test Data received.first record is:")
        # print(df.head(1))

        return df

    def _get_sensors_data_from_sql(self) -> pd.DataFrame:
        server = '.'
        database = 'Your Database Name'
        username = 'Database username'
        password = 'Database password'
        cnxn = pyodbc.connect(
            'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
        cursor = cnxn.cursor()
        query = "SELECT *  FROM tb2_7train;"  # train dataset used to train in case of 7 sensors as input
        # query = "sp_getRandomRows;"  # train dataset used to train in case of 86 sensors as input
        df = pd.read_sql(query, cnxn)
        # data initialization done
        # print("data initialization done.first record is:")
        # print(df.head(1))

        return df

    # return the first record of train
    def reset(self, test=False):
        self.current_record_index = 0
        if test:
            return self.all_states_test.iloc[0, :]  # row 0  and all columns

        return self.all_states.iloc[0, :]  # row 0  and all columns

    ###
    # action is the predicted label's index
    # do the action and then
    # return next_state, reward, done, info
    def step(self, actionIndex, test=False):
        done = False
        # Reward
        rewardd: int = -1
        # print("%%%%%%%% step method %%%%%%%%%%%%%")
        # print("actionIndex is:", actionIndex)
        # In this problem, doing action does not change the dataset
        # It just compares the predicted action and real actions and gives a reward.
        self.current_record_index += 1
        # if isDone, so we are in the last record
        if self.isDone(self.current_record_index, test):
            # rewardd = 0
            done = True
            if test:
                current_record: pd.Series = self.all_states_test.iloc[self.current_record_index - 1, :]
                actual_labels: pd.Series = self.all_labels_test.iloc[self.current_record_index - 1, :]
                actual_labels_list = actual_labels.tolist()
            else:
                current_record: pd.Series = self.all_states.iloc[self.current_record_index - 1, :]
                actual_labels: pd.Series = self.all_labels.iloc[self.current_record_index - 1, :]
                actual_labels_list = actual_labels.tolist()
            predicted_actions = self.action_space[actionIndex]
            y_true_index = self.action_space.index(actual_labels_list)
            if actual_labels_list == predicted_actions:
                rewardd = 1
            return current_record, rewardd, done, y_true_index
        # print("     real labels:", actual_labels_list)
        # print("predicted labels:", predicted_actions)
        # print("          reward: ", rewardd)
        # calculate reward:------------------
        if test:
            current_record: pd.Series = self.all_states_test.iloc[self.current_record_index - 1, :]
            actual_labels: pd.Series = self.all_labels_test.iloc[self.current_record_index - 1, :]
            actual_labels_list = actual_labels.tolist()
        else:
            current_record: pd.Series = self.all_states.iloc[self.current_record_index - 1, :]
            actual_labels: pd.Series = self.all_labels.iloc[self.current_record_index - 1, :]
            actual_labels_list = actual_labels.tolist()

        predicted_actions = self.action_space[actionIndex]
        y_true_index = self.action_space.index(actual_labels_list)

        if actual_labels_list == predicted_actions:
            rewardd = 1
        # -------------------------------------
        return current_record, rewardd, done, y_true_index

    def isDone(self, current_record_index, test=False):
        # 1>1  is false
        if test:
            if current_record_index > self.all_states_test.__len__() - 1:
                return True
        else:
            if current_record_index > self.all_states.__len__() - 1:
                return True


def test_agent(model: DQNAgent, env: MySmartHome, n_episodes=20, test=True) -> [int, List, List]:
    start_time_test = time.time()

    predicted_labels_index: List = []
    y_true_index_labels_index = []
    done = False
    episode_reward = 0
    state = env.reset(test=True)
    state = np.reshape(state.values, [1, state_size])
    while not done:
        action = model.act(state)
        state, reward, done, y_true_index = env.step(action, test=test)
        state = np.reshape(state.values, [1, state_size])
        episode_reward += reward
        # store predicted labels to use for confusion matrix at the end
        predicted_labels_index.append(action)
        y_true_index_labels_index.append(y_true_index)

    end_time_test = time.time()
    return [episode_reward, predicted_labels_index, y_true_index_labels_index, start_time_test, end_time_test]


def train_agent(env: MySmartHome, agent: DQNAgent) -> List[int]:
    reward_per_episode: List[int] = []
    batch_size = 32

    for e in range(1, EPISODES + 1):  # (i,j) start from i to j-1
        episode_reward = 0
        # state type is Series
        state = env.reset()  # go to first record of dataset
        done = False
        state = np.reshape(state.values, [1, state_size])
        # while not reached the terminal state:
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # reward = reward if not done else -10
            reward = reward
            # store episode reward
            episode_reward += reward
            # print("--------------next_state.shape---------------")
            # before reshape, shape is (8,)
            next_state = np.reshape(next_state.values, [1, state_size])
            # print(next_state.shape)  # is: (1,8)  like: [['timestamp',0,0,0,0,0,0,0]]

            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{} , epsilon: {:.2} , rewards:{}"
                      .format(e, EPISODES, agent.epsilon, episode_reward))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)  # train the agent

        # after finishing an episode, log the reward for this episode
        reward_per_episode.append(episode_reward)
        # todo tune save time
        if e % 2 == 0:
            print(f"episode {e}, agent saved.")
            agent.save(SaveLocation)
            # todo here early exit
            # compare last two 20 consequtive episode rewards
            if e > 10:
                if episode_reward >= 20 or (np.mean(reward_per_episode[-5:]) >= np.mean(reward_per_episode[-10:-5])):
                    print(f"episode {e}, agent saved.")
                    agent.save(SaveLocation)
                    print("Early Exit")
                    break

    return reward_per_episode


def get_test_results() -> [int, List, List]:
    predictedd_labels_index: List
    y_truee_index_labels_index: List

    newAgent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        epsilon=0)
    newAgent.load(SaveLocation)
    one_episode_reward, \
    predictedd_labels_index, \
    y_truee_index_labels_index, start_time_test, end_time_test = test_agent(model=newAgent, env=env, n_episodes=20,
                                                                            test=True)

    return [one_episode_reward, predictedd_labels_index, y_truee_index_labels_index, start_time_test, end_time_test]


if __name__ == "__main__":
    for i in range(2, 3):  # runs from 1 to 10  end is not included

        print(f"test {i}")
        env = MySmartHome()
        state_size = env.state_size  # in case of 7 sensors, is 7 plus time if it is.
        action_size: int = env.action_space.__len__()  # is 8
        print("action_size: ", action_size)
        # todo
        agent = DQNAgent(state_size, action_size)

        # # watch untrained agent
        # mean_rewards_per_episode = test_agent(model=agent, env=env, n_episodes=20, test=True)
        # # test trained agent
        # print("Watch untrained agent  ")
        # print(f"Average Test  Reward:{mean_rewards_per_episode}")

        # capture duration of algorithm
        start_time_train = time.time()
        # # todo undo redo train agent
        # reward_per_episode = train_agent(env, agent)
        end_time_train = time.time()
        # # Showing Results -------------------------------------------------------
        # plt.plot(reward_per_episode)
        # plt.title(f"Rewards per Episode {name}")
        # plt.show()

        predicted_labels_index: List
        y_true_index_labels_index: List
        start_time_saved_agent = 0
        end_time_saved_agent = 0
        # todo
        print("Started Testing...")
        one_episode_reward, predicted_labels_index, y_true_index_labels_index, \
        start_time_saved_agent, end_time_saved_agent = get_test_results()

        # test Saved agent
        print("___________see Saved agent______________________ ")
        # print(f"Average Trained  Reward:{mean_rewards_per_episode}")
        # todo
        print(f"one_episode_reward Saved agent:{one_episode_reward}")
        print("________________________________________________ ")

        print("*****************************************")
        t1 = (end_time_train - start_time_train) * 1000
        print(f"                  train time:{t1}ms or{t1 / 1000}s or {t1 / 60000}minute ")

        t2 = (end_time_saved_agent - start_time_saved_agent) * 1000
        print(f"       saved agent test time:{t2}ms or{t2 / 1000}s or {t2 / 60000}minute   ")

        from sklearn.metrics import classification_report, confusion_matrix

        # print(confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred))
        # y_predicted_df = pd.DataFrame(predicted_labels_index, columns=["B5", "B6", "B7"])

        # print(confusion_matrix(env.all_labels_test, y_predicted_df))

        #  shape must be:(n_samples,)
        y_len = y_true_index_labels_index.__len__()
        yhat_len = predicted_labels_index.__len__()
        print("confusion_matrix:")
        print(confusion_matrix(
            np.array(y_true_index_labels_index).reshape(y_len, ),
            np.array(predicted_labels_index).reshape(yhat_len, ),
            # labels=[0, 1, 2, 3, 4, 5, 6]
            # labels=[0, 1, 2, 3, 4, 5]
            labels=[0, 1]
        )
        )
        print("classification_report:")
        print(classification_report(
            np.array(y_true_index_labels_index).reshape(y_len, ),
            np.array(predicted_labels_index).reshape(yhat_len, ),
        )
        )

        from sklearn.metrics import accuracy_score

        print("accuracy_score:")
        print(accuracy_score(np.array(y_true_index_labels_index).reshape(y_len, ),
                             np.array(predicted_labels_index).reshape(yhat_len, ), ))

        import seaborn as sns

        T5_lables = ['predicted:0', 'predicted:1']
        T5_lablesY = ['True:0', 'True:1']
        # T5_lables = [0, 1, 2, 3, 4, 5, 6]

        ax = plt.subplot()
        # labels, title and ticks
        ax.xaxis.set_ticklabels(T5_lables)
        ax.yaxis.set_ticklabels(T5_lablesY)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix {name} test{i}')
        cm = confusion_matrix(y_true_index_labels_index, predicted_labels_index)
        sns.heatmap(cm, annot=True, fmt='g', ax=ax, xticklabels=T5_lables,
                    yticklabels=T5_lablesY)  # annot=True to annotate cells, ftm='g' to disable scientific notation
        plt.show()
        # The confusion matrix takes a vector of labels (not the one-hot encoding). You should run
        #
        # confusion_matrix(y_test.values.argmax(axis=1), predictions.argmax(axis=1))
