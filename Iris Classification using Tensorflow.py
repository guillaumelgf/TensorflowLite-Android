
# coding: utf-8

# # Classification des Iris en utilisant tensorflow

# # I - Introduction
# 
# ---
# #### Objectif
# <div style="text-align:justify;">L'objectif est de suivre un projet de Machine du concept à son intégration. Nous allons donc partir d'une base de données simple existant déjà sur internet. Nous allons ensuite concevoir un classificateur multiclasse à l'aide de tensorflow et mettre ce modèle en place sur une application mobile.</div>
# 
# #### La base de données
# <div style="text-align:justify;">Nous allons utiliser la base de données de classification d'Iris du [site Kaggle](https://www.kaggle.com/uciml/iris). Dans cette base de données, il existe 3 labels: Iris-setosa, Iris-versicolor
# et Iris-virginica. Ces labels correspondent aux espèces d'Iris que nous souhaitons différencier. La base de données contient la largeur ainsi que la longueur des pétales et des sépales de 150 plantes.</div>

# # II - Génération du modèle
# 
# ---

# ## 1. Exploration de la base de données

# In[1]:


import pandas  as pd    # Data Structure
import seaborn as sns   # Data Vizualisation


# On commence par importer la base de données à l'aide de **pandas**.

# In[2]:


datas = pd.read_csv("datas/Iris.csv")


# In[3]:


display(datas.head())
print("Shape: {}".format(datas.shape))


# On utilise **seaborn** pour explorer graphiquement les données.

# In[4]:


g=sns.pairplot(datas, hue="Species", size=2.5)


# ## 2. Data Preprocessing

# ### 2.1 Drop Id

# L'id n'est d'aucune utilité, on s'en débarasse donc dès le début.

# In[5]:


datas.drop("Id", axis=1, inplace=True)


# ### 2.2 Séparation labels/features

# In[6]:


# On récupère les noms de toutes les colonnes 
cols=datas.columns

# On les sépare
features = cols[0:4]
labels = cols[4]

print("Liste des features:")
for k in features:
    print("- {}".format(k))
print("\nLabel: {}".format(labels))


# ### 2.3 Mélange des données

# In[7]:


import numpy as np   # Manipulation de listes


# **numpy** est utilisé ici pour mélanger la base de données.

# In[8]:


indices = datas.index.tolist()
indices = np.array(indices)
np.random.shuffle(indices)
X = datas.reindex(indices)[features]
y = datas.reindex(indices)[labels]


# ### 2.4 Categorical to numerical

# On convertit les valeurs des labels qui sont des catégories en valeurs numériques pour être intérprétées par notre algorithme.

# In[9]:


y.head()


# In[10]:


from pandas import get_dummies


# In[11]:


y=get_dummies(y)


# In[12]:


display(y.head())


# ### 2.5 Train/Test split
# 
# <div style="text-align:justify;"><br>Pour pouvoir évaluer la qualité de notre algorithme il faut séparer les données en deux. La base de données d'apprentissage est utilisée pour apprendre à l'algorithme comment classifier les données. Une fois que cela est fait, on est capable de prédire la classe avec une certaine précision. Pour vérifier si l'algorithme est capable de bien généraliser à des données qu'il n'a pas appris (éviter l'**overfitting**), on calcul la précision de l'algorithme pour prédire sur la base de données de test.</div>
# 
# - Train: 80%
# - Test : 20%

# In[13]:


from sklearn.cross_validation import train_test_split


# In[14]:


y=get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = np.array(X_train).astype(np.float32)
X_test  = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test  = np.array(y_test).astype(np.float32)


# In[15]:


print("Shapes:")
print("x_train: {}\ty_train: {}".format(X_train.shape, y_train.shape))
print("x_test:  {} \ty_test:  {}".format(X_test.shape, y_test.shape))


# ## 3. Tensorflow


import tensorflow as tf


# ### 3.1 Conception du modèle


training_size = X_train.shape[1]
test_size = X_test.shape[1]
num_features = 4
num_labels = 3


num_hidden = 10

tf.reset_default_graph()
tf.set_random_seed(1)
x = tf.placeholder("float", [None, num_features], name="input")
y = tf.placeholder("float", [None, num_labels])

weights_1 = tf.Variable(tf.random_normal([num_features, num_hidden]))
bias_1 = tf.Variable(tf.zeros([num_hidden]))
weights_2 = tf.Variable(tf.random_normal([num_hidden, num_labels]))
bias_2 = tf.Variable(tf.zeros([num_labels]))

logits_1 = tf.matmul(x , weights_1) + bias_1
rel_1 = tf.nn.relu(logits_1)
logits_2 = tf.matmul(rel_1, weights_2) + bias_2

prediction = tf.nn.softmax(logits_2, axis=1, name="prediction")
categorie = tf.argmax(prediction, axis=1, name="categorie")

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), axis=0))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


# ### 3.2 Entrainement | Evaluation | Exportation


epochs = 10000
display_step = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print("1. Entrainement ==============================")
    feed_dict = {
        x: X_train,
        y: y_train
    }
    for epoch in range(epochs):
        _, c = sess.run([optimizer, loss], feed_dict=feed_dict)
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(c))
    print("Optimization Finished!\n")
    
    
    print("2. Evaluation =================================")
    correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy: {}\n".format(accuracy.eval({x: X_test, y: y_test})))
    
    
    print("3. Exportation ================================")
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=["prediction"])
    # save frozen graph def to text file
    with open("estimator_frozen_graph.pbtxt", "w") as fp:
        fp.write(str(frozen_graph_def))

converter = tf.contrib.lite.TocoConverter.from_frozen_graph("estimator_frozen_graph.pbtxt", ["input"], ["prediction"])
tflite_model = converter.convert()
open("estimator_model.tflite", "wb").write(tflite_model)