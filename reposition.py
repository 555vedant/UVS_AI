import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import logging

logging.basicConfig(level=logging.DEBUG)

def restriction_rule(state, vehicle, stations):
    F = np.ones(len(stations) + 1)
    for i, s in enumerate(stations):
        n_vehicles = int(state[i])
        n_tasks = int(state[len(stations) + i])
        free_charging = s.charging_points - n_vehicles
        if n_tasks == 0 or vehicle.electricity < 20 or free_charging <= 0:
            F[i] = 0
    if vehicle.electricity < 20:
        F[len(stations)] = 0
    return F

class Environment:
    def __init__(self, stations, vehicles, tasks):
        self.stations = stations
        self.vehicles = vehicles
        self.tasks = tasks
        self.time_slot = 0
    
    def reset(self):
        self.time_slot = 0
        for t in self.tasks:
            t.assigned = False
        for v in self.vehicles:
            v.electricity = 100
        return [self.get_state(v) for v in self.vehicles]
    
    def get_state(self, vehicle):
        n_v = [sum(1 for v in self.vehicles if v.station == s.id) for s in self.stations]
        n_t = [sum(1 for t in self.tasks if t.dest == s.id and not t.assigned) for s in self.stations]
        demand = [sum(t.fee for t in self.tasks if t.dest == s.id and not t.assigned) for s in self.stations]
        time = [1 if i == self.time_slot % 24 else 0 for i in range(24)]
        vehicle_pos = [1 if s.id == vehicle.station else 0 for s in self.stations]
        state = np.concatenate([n_v, n_t, demand, time, vehicle_pos])
        logging.debug(f"State for {vehicle.id}: n_v={n_v}, n_t={n_t}, demand={demand}, vehicle_pos={vehicle_pos}")
        return state
    
    def step(self, vehicle, action):
        if action != len(self.stations):
            vehicle.station = self.stations[action].id
            vehicle.electricity = max(0, vehicle.electricity - 10)
        n_v = sum(1 for v in self.vehicles if v.station == vehicle.station) + 1
        n_t = sum(1 for t in self.tasks if t.dest == vehicle.station and not t.assigned)
        total_fee = sum(t.fee for t in self.tasks if t.dest == vehicle.station and not t.assigned)
        reward = total_fee / n_v if n_t > 0 else -5
        self.time_slot += 1
        return self.get_state(vehicle), reward

class RDR:
    def __init__(self, state_size, action_size):
        self.model = self.build_model(state_size, action_size)
        self.target_model = self.build_model(state_size, action_size)
        self.memory = []
        self.epsilon = 0.05
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.batch_size = 32
        if os.path.exists("rdr_model.weights.h5"):
            self.model.load_weights("rdr_model.weights.h5")
            self.target_model.load_weights("rdr_model.weights.h5")
            logging.debug("Loaded RDR model weights")
    
    def build_model(self, state_size, action_size):
        model = Sequential([
            Dense(64, input_dim=state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model
    
    def act(self, state, vehicle, stations):
        if random.random() < self.epsilon:
            valid_actions = [i for i in range(len(stations) + 1) if restriction_rule(state, vehicle, stations)[i] == 1]
            return random.choice(valid_actions) if valid_actions else len(stations)
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        F = restriction_rule(state, vehicle, stations)
        q_values = q_values * F
        logging.debug(f"RDR q_values: {q_values}")
        return np.argmax(q_values)
    
    def train(self, env, episodes=1000):
        for e in range(episodes):
            if e % 100 == 0:
                print(f"RDR Episode {e}/{episodes}")
            states = env.reset()
            for i, vehicle in enumerate(env.vehicles):
                state = states[i]
                action = self.act(state, vehicle, env.stations)
                next_state, reward = env.step(vehicle, action)
                self.memory.append((state, action, reward, next_state))
                if len(self.memory) > self.batch_size:
                    batch = random.sample(self.memory, self.batch_size)
                    states, actions, rewards, next_states = zip(*batch)
                    states = np.array(states)
                    next_states = np.array(next_states)
                    rewards = np.array(rewards)
                    targets = self.model.predict(states, verbose=0)
                    target_next = self.target_model.predict(next_states, verbose=0)
                    for i in range(self.batch_size):
                        targets[i][actions[i]] = rewards[i] + self.gamma * np.max(target_next[i])
                    self.model.fit(states, targets, epochs=1, verbose=0)
            if e % 10 == 0:
                self.target_model.set_weights(self.model.get_weights())
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class RAR:
    def __init__(self, state_size, action_size):
        self.actor = self.build_actor(state_size, action_size)
        self.critic = self.build_critic(state_size)
        self.critic_target = self.build_critic(state_size)
        self.memory = []
        self.gamma = 0.95
        self.batch_size = 32
        if os.path.exists("rar_actor.weights.h5"):
            self.actor.load_weights("rar_actor.weights.h5")
            logging.debug("Loaded RAR actor weights")
        if os.path.exists("rar_critic.weights.h5"):
            self.critic.load_weights("rar_critic.weights.h5")
            self.critic_target.load_weights("rar_critic.weights.h5")
            logging.debug("Loaded RAR critic weights")
    
    def build_actor(self, state_size, action_size):
        model = Sequential([
            Dense(64, input_dim=state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(action_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model
    
    def build_critic(self, state_size):
        model = Sequential([
            Dense(64, input_dim=state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model
    
    def act(self, state, vehicle, stations):
        probs = self.actor.predict(np.array([state]), verbose=0)[0]
        F = restriction_rule(state, vehicle, stations)
        probs = probs * F
        probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones(len(probs)) / len(probs)
        logging.debug(f"RAR probs: {probs}")
        return np.random.choice(len(probs), p=probs)
    
    def train(self, env, episodes=1000):
        for e in range(episodes):
            if e % 100 == 0:
                print(f"RAR Episode {e}/{episodes}")
            states = env.reset()
            for i, vehicle in enumerate(env.vehicles):
                state = states[i]
                action = self.act(state, vehicle, env.stations)
                next_state, reward = env.step(vehicle, action)
                self.memory.append((state, action, reward, next_state))
                if len(self.memory) > self.batch_size:
                    batch = random.sample(self.memory, self.batch_size)
                    states, actions, rewards, next_states = zip(*batch)
                    states = np.array(states)
                    next_states = np.array(next_states)
                    rewards = np.array(rewards)
                    targets = rewards + self.gamma * self.critic_target.predict(next_states, verbose=0).flatten()
                    advantages = targets - self.critic.predict(states, verbose=0).flatten()
                    action_probs = np.zeros((self.batch_size, len(self.stations) + 1))
                    for i, a in enumerate(actions):
                        action_probs[i][a] = 1
                    self.actor.fit(states, action_probs, sample_weight=advantages, epochs=1, verbose=0)
                    self.critic.fit(states, targets, epochs=1, verbose=0)
            if e % 10 == 0:
                self.critic_target.set_weights(self.critic.get_weights())
    
def simple_rule_reposition(stations, vehicles, tasks, assignments):
    task_threshold = 1
    reposition_results = []
    unassigned_tasks = {s.id: sum(1 for t in tasks if t.dest == s.id and not t.assigned) for s in stations}
    vehicle_counts = {s.id: sum(1 for v in vehicles if v.station == s.id) for s in stations}
    
    for vehicle in vehicles:
        if any(str(vehicle.id) in a for a in assignments):
            continue
        eligible_stations = []
        for s in stations:
            n_vehicles = sum(1 for v in self.vehicles if v.station == s.id)
            n_tasks = sum(1 for t in self.tasks if t.dest == s.id and not t.assigned)
            free_charging = s.charging_points - n_vehicles
            if n_tasks > task_threshold and vehicle.electricity >= 20 and free_charging > 0:
                eligible_stations.append(s)
        if eligible_stations:
            to_station = random.choice(eligible_stations)
            if to_station.id != vehicle.station:
                reposition_results.append(f"{vehicle.id} moves from {vehicle.station} to {to_station.id}")
                vehicle.station = to_station.id
                vehicle.electricity = max(0, vehicle.electricity - 10)
    
    verified = []
    for r in reposition_results:
        parts = r.split()
        to_id = parts[-1]
        if unassigned_tasks.get(to_id, 0) > 0 and vehicle_counts.get(to_id, 0) < unassigned_tasks.get(to_id, 0) + 2:
            verified.append(r)
    
    return verified

def reposition_vehicles(stations, vehicles, tasks, assignments, algo='rdr'):
    env = Environment(stations, vehicles, tasks)
    state_size = len(env.get_state(vehicles[0]))
    action_size = len(stations) + 1
    agent = RDR(state_size, action_size) if algo == 'rdr' else RAR(state_size, action_size)
    
    results = []
    for vehicle in vehicles:
        if any(str(vehicle.id) in a for a in assignments):
            continue
        state = env.get_state(vehicle)
        action = agent.act(state, vehicle, stations)
        if action != len(stations) and stations[action].id != vehicle.station:
            results.append(f"{vehicle.id} moves from {vehicle.station} to {stations[action].id}")
            vehicle.station = stations[action].id
            vehicle.electricity = max(0, vehicle.electricity - 10)
    
    verified = []
    unassigned_tasks = {s.id: sum(1 for t in tasks if t.dest == s.id and not t.assigned) for s in stations}
    vehicle_counts = {s.id: sum(1 for v in vehicles if v.station == s.id) for s in stations}
    for r in results:
        parts = r.split()
        to_id = parts[-1]
        if unassigned_tasks.get(to_id, 0) > 0 and vehicle_counts.get(to_id, 0) < unassigned_tasks.get(to_id, 0) + 2:
            verified.append(r)
    
    return verified

def pre_train_and_save(stations, vehicles, tasks, algo='rdr', episodes=1000):
    env = Environment(stations, vehicles, tasks)
    state_size = len(env.get_state(vehicles[0]))
    action_size = len(stations) + 1
    
    if algo == 'rdr':
        logging.debug("Training RDR...")
        agent = RDR(state_size, action_size)
        agent.train(env, episodes=episodes)
        agent.model.save_weights("rdr_model.weights.h5")
        logging.debug("RDR training complete.")
    else:
        logging.debug("Training RAR...")
        agent = RAR(state_size, action_size)
        agent.train(env, episodes=episodes)
        agent.actor.save_weights("rar_actor.weights.h5")
        agent.critic.save_weights("rar_critic.weights.h5")
        logging.debug("RAR training complete.")