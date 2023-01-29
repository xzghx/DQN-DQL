from typing import List, Any, Union

import pyodbc  # to read table from sql
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time as time
import sys
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.models import Sequential

# Guide Notes:
# This is a DQN(Deep Q Network) agent which consists of a neural net and a target net.
# each record of input data is like:
# if 7 sensors are input:
# normalizedTime | V21 | V22 | V23 | S137 | S138 | S139 | S140 | B5

# and if 86 sensors are input;
# normalizedTime |... 86 sensors ...| B5

# where normalizedTime is normal time of event and other columns are sensor values in that timestamp
# and last column (i.e B5 ) is label(i.e it's value must be predicted by our algorithm)

# model predicts the action index
from tensorflow import keras

EPISODES = 1000  # gradually change it to 1000. must be large enough to be trained

name = "dqn_in7_out1_81.h5"
# name = "dqn_in86_out1_81.h5"

# SaveLocation = "savedModel2/dqn_in86_out1_81.h5"
SaveLocation = "savedModel2/dqn_in7_out1_81.h5"
train_start_column = 1  # for train86 set 4 for train7 set 1
test_start_column = 3  # for train86 set    for test7 set 3
train_stop_reward = 15
np.random.seed(10)
# on_train = True
on_train = False


# DQN Agent
# has a 2 layer Neural Network to approximate Q function and replay memory & target Q network
class DQNAgent:
    def __init__(self, state_size, action_size: int, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size  # is 7 for 3 actuators
        self.memory = deque(maxlen=2000)  # replay memory
        self.gamma = 0.7  # discount rate 0.9 0.1 0.001
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01  # 0.01 - 0.00001
        self.batch_size = 64
        self.train_start = 1000
        # main and target model
        self.model = self._build_model()
        self.target_model = self._build_model()
        # initialize target model
        self.update_target_model()
        self.update_frequency = 50
        self.load_model = False
        if self.load_model:
            self.model.load_weights(SaveLocation)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(5, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        # model.summary()
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=[
                # tf.metrics.SparseCategoricalAccuracy()
                # tf.metrics.CategoricalAccuracy(),
            ],
        )
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # returns the action index.
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print(".")
            return np.random.randint(0, self.action_size)  # randrange is from [start - last)

        act_values = self.model.predict(state)

        # act_values is ndarray (1,7) like:  [[0,0,0,0,0,0,0,0]]
        # print("np.argmax(act_values[0]): ", np.argmax(act_values[0])) # argmax gives index of max value
        return np.argmax(act_values[0])  # returns action index

    # this is the train part which picks samples randomly from replay memory (with batch_size)
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]  # s
            action.append(mini_batch[i][1])  # a
            reward.append(mini_batch[i][2])  # r
            update_target[i] = mini_batch[i][3]  # s'
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)  # Q(s') from prediction net
        target_val = self.target_model.predict(update_target)  # Q(s') from target net

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (
                    np.amax(target_val[i]))

        # fit the model to update it's nn weights
        result = self.model.fit(update_input, target, batch_size=self.batch_size,
                                epochs=1, verbose=0)

    def load(self, path_weights):
        self.model.load_weights(path_weights)

    def save(self, path_weights):

        self.model.save_weights(path_weights)

    def setEpsilon(self, eps):
        self.epsilon = eps


class MySmartHome:

    def __init__(self):
        self.current_record_index = 0
        self.action_space = [[0], [1]]

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
        # ignore last 3 columns for state since they are labels.
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
        # get test dataset for testing algorithm in case of 7 inputs
        # query = "sp_getRandomTests86"
        query = "sp_getRandomTests7"  # this is used
        df = pd.read_sql(query, cnxn)
        # data initialization done
        # print("Test Data received.first record is:")
        # print(df.head(1))

        return df

    # open a connection to sql server. data had been pre proccessed in sql and we are getting data via stored procedures.
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
    reward_per_episode, episodes = [], []

    for e in range(1, EPISODES + 1):  # (i,j) start from i to j-1
        episode_reward = 0  # sum the score of episode
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
            agent.replay()
            state = next_state
            if done:
                # every episode update the target model to be same with model
                if e % agent.update_frequency == 0:
                    agent.update_target_model()
                print("episode: {}/{} , epsilon: {:.3} ,memory length:{},  rewards:{}"
                      .format(e, EPISODES, agent.epsilon, len(agent.memory), episode_reward))
                # every episode, plot the play time
                # episode_reward = score if score == 500 else score + 100
                reward_per_episode.append(episode_reward)
                episodes.append(e)
                pylab.plot(episodes, reward_per_episode, '-c')
                pylab.xlabel('Episode')
                pylab.ylabel('Reward')
                pylab.title('Rewards per episode')
                pylab.savefig("save_graph2/dqn2_graph.png")
                # if the mean of scores of last 10 episode is bigger than 20
                # stop training
                if np.mean(reward_per_episode[-min(10, len(reward_per_episode)):]) >= train_stop_reward:
                    print(f"episode {e}, agent saved.EarlyExit")
                    agent.save(SaveLocation)
                    sys.exit()

        if e % 2 == 0:
            print(f"episode {e}, agent saved.")
            agent.save(SaveLocation)

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
    i = 1
    # for i in range(2, 3):  # runs from 1 to 10  end is not included
    # print(f"test {i}")
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
    if on_train:
        reward_per_episode = train_agent(env, agent)
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
