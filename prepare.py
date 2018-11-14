#SVM related code
import numpy as np
'''
# dot product 
# x.y=||x||*||y||*cos(theta)
x=[3,4]
norm_x=np.linalg.norm(x) 
print norm_x #5

#direction of a vector
x/np.linalg.norm(x)

#dot product
def geopetric_dot_product(x,y,theta):
    x_norm=np.linalg.norm(x)
    y_norm=np.linalg.norm(y)
    return x_norm*y_norm*math.cos(math.radians(theta))
#numpy funcion
x=np.array([3,5])
y=np.array([8,2])
print(np.dot(x,y))
'''
# Compute the functional margin of an example (x,y)
# with respect to a hyperplane defined by w and b.
def example_functional_margin(w, b, x, y):
    result = y * (np.dot(w, x) + b)
    return result

# Compute the functional margin of a hyperplane
# for examples X with labels y.
def functional_margin(w, b, X, y):
    return np.min([example_functional_margin(w, b, x, y[i])
                  for i, x in enumerate(X)])
x = np.array([1, 1])
y = 1

b_1 = 5
w_1 = np.array([2, 1])

w_2 = w_1 * 10
b_2 = b_1 * 10

print(example_functional_margin(w_1, b_1, x, y))  # 8
print(example_functional_margin(w_2, b_2, x, y))  # 80

# Compute the geometric margin of an example (x,y)
# with respect to a hyperplane defined by w and b.
def example_geometric_margin(w, b, x, y):
    norm = np.linalg.norm(w)
    result = y * (np.dot(w/norm, x) + b/norm)
    return result

# Compute the geometric margin of a hyperplane
# for examples X with labels y.
def geometric_margin(w, b, X, y):
    return np.min([example_geometric_margin(w, b, x, y[i])
                  for i, x in enumerate(X)])

print(example_geometric_margin(w_1, b_1, x, y))  # 3.577708764
print(example_geometric_margin(w_2, b_2, x, y))  # 3.577708764


# Compare two hyperplanes using the geometrical margin.

positive_x = [[2,7],[8,3],[7,5],[4,4],[4,6],[1,3],[2,5]]
negative_x = [[8,7],[4,10],[9,7],[7,10],[9,6],[4,8],[10,10]]

X = np.vstack((positive_x, negative_x))
y = np.hstack((np.ones(len(positive_x)), -1*np.ones(len(negative_x))))
print(X,y)
w = np.array([-0.4, -1])
b = 8

# change the value of b
print(geometric_margin(w, b, X, y))          # 0.185695338177
print(geometric_margin(w, 8.5, X, y))        # 0.64993368362



# Use cvxopt package to do QP solver  about the dataset

from succinctly.datasets import get_dataset, linearly_separable as ls

import cvxopt.solvers


X, y = get_dataset(ls.get_training_examples)
m = X.shape[0]


# Gram matrix - The matrix of all possible inner products of X.
K = np.array([np.dot(X[i], X[j])
              for j in range(m)
              for i in range(m)]).reshape((m, m))

P = cvxopt.matrix(np.outer(y, y) * K)
q = cvxopt.matrix(-1 * np.ones(m))

# Equality constraints
A = cvxopt.matrix(y, (1, m))
b = cvxopt.matrix(0.0)

# Inequality constraints
G = cvxopt.matrix(np.diag(-1 * np.ones(m)))
h = cvxopt.matrix(np.zeros(m))

# Solve the problem
solution = cvxopt.solvers.qp(P, q, G, h, A, b)

# Lagrange multipliers
multipliers = np.ravel(solution['x'])

# Support vectors have positive multipliers.
has_positive_multiplier = multipliers > 1e-7
sv_multipliers = multipliers[has_positive_multiplier]

support_vectors = X[has_positive_multiplier]
support_vectors_y = y[has_positive_multiplier]

# 
def compute_w(multipliers, X, y):
    return np.sum(multipliers[i] * y[i] * X[i]
                  for i in range(len(y)))
# compute b using the average method
def compute_b(w, X, y):
    return np.sum([y[i] - np.dot(w, X[i]) 
                   for i in range(len(X))])/len(X)

#Because Lagrange multipliers for non-support vectors are almost zero,
#we can also compute using only support vectors data and their multipliers
w = compute_w(multipliers, X, y)
w_from_sv = compute_w(sv_multipliers, support_vectors, support_vectors_y)


print(w)          # [0.44444446 1.11111114]
print(w_from_sv)  # [0.44444453 1.11111128]
b = compute_b(w, support_vectors, support_vectors_y) # -9.666668268506335
#####above is hard svm#####

#kernel svm
def polynomial_kernel(a, b, degree, constant=0):
    result = sum([a[i] * b[i] for i in range(len(a))]) + constant
    return pow(result, degree)

#SMO
def kernel(x1, x2):
    return np.dot(x1, x2.T)

def objective_function_to_minimize(X, y, a, kernel):
    m, n = np.shape(X)
    return 1 / 2 * np.sum([a[i] * a[j] * y[i] * y[j]* kernel(X[i, :], X[j, :])
                           for j in range(m)
                           for i in range(m)])\
           - np.sum([a[i] for i in range(m)])
