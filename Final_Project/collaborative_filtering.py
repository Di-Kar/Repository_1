import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class ClusterCollaborativeFiltering:
        
    def __init__(self, train_data, k_clusters, k_neighbors, batch_size=100):
        
        #Устанавливает количество кластеров, которые будут использоваться для кластеризации пользователей.
        self.k_clusters = k_clusters
    
        #Устанавливает количество ближайших соседей, которые будут рассматриваться при поиске подходящих пользователей.
        self.k_neighbors = k_neighbors
    
        #Сохраняет разреженную матрицу взаимодействий между пользователями и объектами.
        self.train_data = train_data
    
        #Создает объект KMeans с заданным количеством кластеров и устанавливает начальное состояние генератора случайных чисел.
        self.kmeans = KMeans(n_clusters=self.k_clusters, random_state=0)
    
        #Производит кластеризацию пользователей на k_clusters кластеров и сохраняет присвоенные кластеры.
        self.cluster_assignments = self.kmeans.fit_predict(train_data)
    
        #Создает словарь, где каждому кластеру присваивается объект NearestNeighbors для поиска ближайших соседей.
        self.cluster_models = {cluster_id: NearestNeighbors(n_neighbors=self.k_neighbors+1, algorithm='brute', n_jobs=-1)
                                   for cluster_id in range(self.k_clusters)}
        #Для каждого кластера выполняется обучение модели NearestNeighbors на данных, принадлежащих кластеру.
        for cluster_id in range(self.k_clusters):
            cluster_data = train_data[self.cluster_assignments == cluster_id]
            self.cluster_models[cluster_id].fit(cluster_data)

   
    def find_max_liked_users(self, user_id):
        '''
        Функция для поиска пользователей с наивысшей вероятностью понравиться указанному пользователю в том же кластере.
        Parameters:
           - user_id: Идентификатор пользователя, для которого ищутся наиболее предпочтительные пользователи в том же кластере.
        Returns:
           - liked_users: Список идентификаторов пользователей с наивысшей вероятностью понравиться пользователю
        '''
        #Определение идентификатора кластера, к которому принадлежит указанный пользователь
        cluster_id = self.cluster_assignments[user_id]
    
        #Получение модели ближайших соседей для данного кластера.
        cluster_model = self.cluster_models[cluster_id]

        #Поиск ближайших соседей указанного пользователя с помощью модели.
        distances, indices = cluster_model.kneighbors(self.train_data[user_id].reshape(1, -1), n_neighbors=10)
    
        #Фильтрация найденных пользователей: сохранение только тех, кто не является текущим пользователем и имеет вероятность понравиться.
        liked_users = [user for user in indices.flatten()[1:] if user not in self.train_data[user_id].indices and user != 0]
    
        #Возвращение списка liked_users.
        return liked_users

    
    def find_potential_matches(self, user_id):
        '''
        Функция для поиска потенциальных совпадений для указанного пользователя на основе ближайших соседей в том же кластере.
        Parameters:
            - user_id: Идентификатор пользователя, для которого выполняется поиск потенциальных совпадений.
        Returns:
           - potential_matches: Список идентификаторов потенциальных совпадений для указанного пользователя.
        '''
        #Определение идентификатора кластера, к которому принадлежит указанный пользователь
        cluster_id = self.cluster_assignments[user_id]
    
        #Получение модели ближайших соседей для данного кластера
        cluster_model = self.cluster_models[cluster_id]
    
        #Поиск ближайших соседей указанного пользователя с помощью модели
        distances, indices = cluster_model.kneighbors(self.train_data[user_id].reshape(1, -1), n_neighbors=10)
    
        #Фильтрация найденных пользователей: сохранение только тех, для которых нет взаимодействия с указанным 
        # пользователем и они не представляют собой самого пользователя.
        potential_matches = [user for user in indices.flatten()[1:] if user not in self.train_data.indices and self.train_data[user_id, user] == 0 and user != 0]
            
        #Возвращение списка potential_matches
        return potential_matches



