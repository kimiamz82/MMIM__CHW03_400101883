#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Mathematical Methods In Engineering - 25872</h1>
# <h4 align="center">Dr. Amiri</h4>
# <h4 align="center">Sharif University of Technology, Fall 2023</h4>
# <h4 align="center">Python Assignment 3</h4>
# <h5 align="center"><font color="cyan"> Feel free to contact TA team for any possible questions about the assignment </font>
#  </h5>
# <h5 align="center"> <font color="cyan"> Questions 1,2: @mh_momeni  -  Question 2,3: @Mahdi_h721 </font> </h5>
# 

# You should write your code in the <font color='green'>Code Cell</font> and then run the <font color='green'>Evaluation Cell</font> to check the correctness of your code.<br>
# <font color='red'>**Please do not edit the codes in the Evaluation Cells.**</font>

# ##  Q1. Implementing QR Factorization <sub><sup>``(40 pt.)``</sup></sub>

# >In this question, we are going to use QR factorization in computing eigenvalues. It is an important building block in numerical linear algebra. Recall that for any matrix $A$
# , $A=QR$
#  where $Q$
#  is orthogonal and $R$
#  is upper-triangular.
# 
#  **Reminder**: The QR algorithm, uses the QR decomposition, but don't confuse the two.

# In[1]:


import numpy as np
np.set_printoptions(suppress=True, precision=4)


# In[2]:


n = 5
A = np.random.rand(n,n)
npQ, npR = np.linalg.qr(A)


# Check that Q is orthogonal:
# 

# In[3]:


np.allclose(np.eye(n), npQ @ npQ.T), np.allclose(np.eye(n), npQ.T @ npQ)


# Check that R is triangular

# In[4]:


npR


# ### Gram-Schmidt
# 
# #### Classical Gram-Schmidt (unstable)

# For each $j$
# , calculate a single projection
# 
# $$v_j=P_ja_j$$
# 
# where $P_j$
#  projects onto the space orthogonal to the span of $q_1,…,q_{j−1}$
# .

# In[5]:


def cgs(A):
    m, n = A.shape
    Q = np.zeros([m,n], dtype=np.float64)
    R = np.zeros([n,n], dtype=np.float64)
    ##Your Code start here
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    ##Your Code ends here
    return Q, R


# In[6]:


Q, R = cgs(A)
np.allclose(A, Q @ R)


# Check if Q is unitary:

# In[7]:


np.allclose(np.eye(len(Q)), Q.dot(Q.T))


# ### Modified Gram-Schmidt (optional)

# Classical (unstable) Gram-Schmidt: for each $j$
# , calculate a single projection
# $$v_j=P_ja_j$$
# where $P_j$
#  projects onto the space orthogonal to the span of $q_1,…,q_{j−1}$
# .
# 
# Modified Gram-Schmidt: for each $j$
# , calculate $j−1$
#  projections
# $$P_j=P_{⊥q_{j−1}⋯⊥q_2⊥q_1}$$

# In[8]:


n = 3
A = np.random.rand(n,n).astype(np.float64)


# In[9]:


def mgs(A):
    V = A.copy()
    m, n = A.shape
    Q = np.zeros([m,n], dtype=np.float64)
    R = np.zeros([n,n], dtype=np.float64)
    ##Your Code start here
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j]

        for i in range(j+1, n):
            R[j, i] = np.dot(Q[:, j], V[:, i])
            V[:, i] = V[:, i] - R[j, i] * Q[:, j]


    ##Your Code ends here
    return Q, R


# In[10]:


Q, R = mgs(A)
np.allclose(np.eye(len(Q)), Q.dot(Q.T.conj()))


# In[11]:


np.allclose(A, np.matmul(Q,R))


# ### Classical vs Modified Gram-Schmidt

# In this part, we want to construct a square matrix $A$ with random singular vectors and widely varying singular values spaced by factors of 2 between $2^{−1}$
#  and $2^{−(n+1)}$

# In[12]:


import matplotlib.pyplot as plt
from matplotlib import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


n = 100
U, X = np.linalg.qr(np.random.randn(n,n))   # set U to a random orthogonal matrix
V, X = np.linalg.qr(np.random.randn(n,n))   # set V to a random orthogonal matrix
S = np.diag(np.power(2,np.arange(-1,-(n+1),-1), dtype=float))  # Set S to a diagonal matrix w/ exp
                                                              # values between 2^-1 and 2^-(n+1)


# In[14]:


A = np.matmul(U,np.matmul(S,V))
QC, RC = cgs(A)
QM, RM = mgs(A) # if you don't complete the mgs function, comment this line


# In[15]:


plt.figure(figsize=(10,10))
plt.semilogy(np.diag(S), 'r.', label="True Singular Values")
plt.semilogy(np.diag(RM), 'go', label="Modified Gram-Shmidt")
plt.semilogy(np.diag(RC), 'bx', label="Classic Gram-Shmidt")
plt.legend()
rcParams.update({'font.size': 18})


# ### Eigenvalue Decomposition using QR Factorization

# Use the QR algorithm (or QR method) to get the eigenvalues of matrix $A$. Do 100 iterations, and print out the 1st, 5th, 10th, 20th and 100th iteration.
# 
#  **Reminder**: The QR algorithm (or QR method), uses the QR factorization, but don't confuse the two.

# In[16]:


##Your Code start here
def qr_algorithm(A, iterations=100):
    n = A.shape[0]
    eigenvalues = np.zeros(n, dtype=np.complex128)

    for i in range(iterations):
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)

        if i + 1 in [1, 5, 10, 20, 100]:
            eigenvalues = np.diag(A)
            print(f"Iteration {i+1}: Eigenvalues = {eigenvalues}")

    return eigenvalues

##Your Code ends here


# In[17]:


A = np.array([1.0, -1.0, 0.0, 0.0, -1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0]).reshape((4, 4))
print("The matrix: \n", A)

print("\n --- Built-in ---")
print(np.linalg.eig(A)[0])
##Your Code start here
def qr_algorithm(A, iterations=100):
    n = A.shape[0]
    eigenvalues = np.zeros(n, dtype=np.complex128)

    for i in range(iterations):
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)

    eigenvalues = np.diag(A)
    return eigenvalues

# Given matrix A
A = np.array([1.0, -1.0, 0.0, 0.0, -1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0]).reshape((4, 4))
print("The matrix: \n", A)

print("\n --- Built-in ---")
print(np.linalg.eig(A)[0])

# Using QR algorithm
eigenvalues_qr = qr_algorithm(A)
print("\n --- QR Algorithm ---")
print(eigenvalues_qr)

##Your Code ends here


# ##  Q2. SVD & Image processing <sub><sup>``(30 pt.)``</sup></sub>

# > In this question, we are going to investigate the use of SVD as a tool to obtain the basis of the matrix in digital image processing.
# The first issue we will examine is image compression. For this purpose, in general, we can consider the information inside the image as a data matrix, then find a suitable basis for this information matrix and by depicting the primary matrix on the more important basis vectors and keeping the necessary information to show the initial image with less information.
# This process results in image compression. It should be noted that for simplicity, the images we are examining in this exercise are in black and white.
# >With the help of SVD, we can find a basis for the column space of the image matrix that we want, so that the columns of the image matrix have the highest correlation with the basis vectors.
# >On the other hand, if these bases are considered in descending order of the size of the singular values, they will contain the most general information of the columns of the image, so by keeping only r initial columns of the found bases (U) along with the combined vectors (V) and the singular values (Σ) corresponding to them, we can retain the image's overall image with a good approximation.
# >From an expert point of view, the reconstruction is actually a lower-order reconstruction of the primary matrix.
# In this view, the value of r is the parameter to control the amount of compression we want.

# ###  part 1
# 
# In this part, the compression operation is performed on the input black and white image with the help of SVD analysis. One of the criteria for comparing the initial and final image is the PSNR criterion. Research about this criterion and draw the PSNR diagram in terms of different r. Then compare the results with the diagram by giving some test inputs and plotting the outputs.
# One of the inputs you give to the function should be the image pic.jpg.

# In[2]:


##Your Code start here
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.metrics import peak_signal_noise_ratio

# Load the image
img = io.imread('pic.jpg')
gray_img = color.rgb2gray(img)  # Convert to grayscale if needed

# Perform SVD
U, S, Vt = svd(gray_img, full_matrices=False)

# Iterate over different values of r (rank)
psnr_values = []
for r in range(1, min(gray_img.shape)):
    compressed_img = np.dot(U[:, :r], np.dot(np.diag(S[:r]), Vt[:r, :]))
    
    # Calculate PSNR
    psnr = peak_signal_noise_ratio(gray_img, compressed_img)
    psnr_values.append(psnr)

# Plot the PSNR diagram
plt.plot(range(1, min(gray_img.shape)), psnr_values)
plt.title('PSNR vs. Compression Rank (r)')
plt.xlabel('Compression Rank (r)')
plt.ylabel('PSNR')
plt.show()

##Your Code ends here


# ###  part 2
# 
# Another issue that we can explore with a similar idea of applying base transform is removing noise from images. In the condition that the noise in the image is uniform and in such a way that it does not distort the general information in the image. from the point of view of SVD, according to the examination of the general information, the direction of the image vector on bases with larger singular values has not changed much, and removing the information related to Examining general information to vectors with less importance can remove noise and keep the main information of the image.
# In this part, add two different noises salt and pepper and Gaussian noise with a desired and appropriate range to the image pic.jpg so that the PSNR of both images is in the same range, then perform the noise reduction process with the help of SVD analysis.
# For both noises, draw the PSNR diagram in terms of different r.
#   Then compare the results with the graph by plotting a number of outputs.
#   Which of the noises is more effective in this method?
# 

# In[4]:


##Your Code start here
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
# Load the original image
original_image = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE)

# Add salt and pepper noise
salt_and_pepper_noise = np.copy(original_image)
salt_and_pepper_noise[np.random.random(size=salt_and_pepper_noise.shape) < 0.02] = 0
salt_and_pepper_noise[np.random.random(size=salt_and_pepper_noise.shape) > 0.98] = 255

# Add Gaussian noise
gaussian_noise = original_image + np.random.normal(0, 25, original_image.shape).astype(np.uint8)
# Load the original image
original_image = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE)

# Add salt and pepper noise
salt_and_pepper_noise = np.copy(original_image)
salt_and_pepper_noise[np.random.random(size=salt_and_pepper_noise.shape) < 0.02] = 0
salt_and_pepper_noise[np.random.random(size=salt_and_pepper_noise.shape) > 0.98] = 255

# Add Gaussian noise
gaussian_noise = original_image + np.random.normal(0, 25, original_image.shape).astype(np.uint8)
def svd_noise_reduction(image, r):
    U, s, Vt = np.linalg.svd(image, full_matrices=False)
    s[r:] = 0
    reconstructed_image = np.dot(U, np.dot(np.diag(s), Vt))
    return reconstructed_image.astype(np.uint8)
r_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]

psnr_salt_and_pepper = []
psnr_gaussian = []

for r in r_values:
    # Noise reduction for salt and pepper noise
    denoised_salt_and_pepper = svd_noise_reduction(salt_and_pepper_noise, r)
    psnr_salt_and_pepper.append(peak_signal_noise_ratio(original_image, denoised_salt_and_pepper))

    # Noise reduction for Gaussian noise
    denoised_gaussian = svd_noise_reduction(gaussian_noise, r)
    psnr_gaussian.append(peak_signal_noise_ratio(original_image, denoised_gaussian))
plt.plot(r_values, psnr_salt_and_pepper, label='Salt and Pepper Noise')
plt.plot(r_values, psnr_gaussian, label='Gaussian Noise')
plt.xlabel('Number of Singular Values (r)')
plt.ylabel('PSNR')
plt.legend()
plt.show()

##Your Code ends here


# ##  Q3. PCA and Clustering <sub><sup>``(40 pt.)``</sup></sub>
# In this problem we want to cluster some data points.
# But first, you should reduce the number of features by the PCA algorithm then use kmeans clustering algorithm

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score as sil, calinski_harabasz_score as chs, silhouette_samples


# ### Import Dataset

# In[8]:


Datapoint = pd.read_csv('Datapoint.csv')


# ### Correlation Heatmap

# In this part plot the correlation Heatmap of features.
# what is your suggestion about the number of principal components that they have high variance? Explain why.

# In[ ]:


##Your Code start here

##Your Code ends here


# ###  Data Preprocessing

# In[ ]:


Datapoint.head()


# ### PCA (Principal Component Analysis)
# > As you know for applying PCA we should scale our data points.Here we use MinMax and Standard Scaler.
# (First, use Standard Scaler)

# Calculate principal components and plot Explained variance by each component.

# In[ ]:


##Your Code start here

##Your Code ends here


# Apply PCA Algorithm from scratch and reduce the number of features to the number you have guessed in previous parts.

# In[ ]:


##Your Code start here

##Your Code ends here


# ### PCA plot in 2D
# Plot data points using their two first features.What do you think about the number of clusters?

# In[ ]:


##Your Code start here

##Your Code ends here


# ### Do all previous steps for MinMax Scaling
# Tell your opinion about diffrences.
# 
# 
# 

# In[ ]:


##Your Code start here

##Your Code ends here


# ### KMeans Clustering
# #### Elbow Method for Determining Cluster Amount on  Dataset

# Using the elbow method to find the optimal number of clusters

# In[ ]:


##Your Code start here

##Your Code ends here


# What is inertia and silhouette metrics?
# Explain them briefly.
# 
# 
# 

# In[ ]:


#You should save your final datapoints in pca_std_datapoint variable
inertia = []
for i in tqdm(range(2,10)):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=15, max_iter= 500, random_state=17)
    kmeans.fit(pca_std_datapoint)
    inertia.append(kmeans.inertia_)


# In[ ]:


silhouette = {}
for i in tqdm(range(2,10)):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=15, max_iter=500, random_state=17)
    kmeans.fit(pca_std_datapoint)
    silhouette[i] = sil(pca_std_datapoint, kmeans.labels_, metric='euclidean')


# Plot the **inertia** and **silhouette ** values

# In[ ]:


##Your Code start here

##Your Code ends here


# Tune the hyper parameters based on your conclusions.

# In[ ]:


model_kmeans = KMeans(n_clusters=..., random_state=0, init='k-means++')
y_predict_kmeans1 = model_kmeans.fit_predict(pca_std_datapoints)


# #Now plot the datapoints usig two first features
# (Each cluster in a different colour)

# In[ ]:





# Do all previous steps for MinMax scaled data points.
# 
# 
# 
# 

# In[ ]:





# Compare the results of different scaling methods in a PCA problem
