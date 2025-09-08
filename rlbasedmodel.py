import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class MaskingDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size  
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=(self.state_size,))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.action_size, activation='sigmoid')(x)  
        model = Model(inputs, outputs)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 2, size=self.action_size)  
        probs = self.model.predict(state, verbose=0)[0]
        return (probs > 0.5).astype(int)  

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])

            target = self.model.predict(state, verbose=0)[0]

            if done:
                target = reward * np.ones(self.action_size)  
            else:
                next_q = self.model.predict(next_state, verbose=0)[0]
                target = reward + self.gamma * next_q  
            self.model.fit(state, np.reshape(target, [1, self.action_size]), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        self.model.save(path)
