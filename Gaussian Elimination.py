import numpy as np
import warnings

def swapRows(A, i, j):
    """
    interchange rows i and j of A
    operates on A in place
    """
    tmp = A[i].copy()
    A[i] = A[j]
    A[j] = tmp

def relError(a, b):
    """
    compute the relative error of a and b
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return np.abs(a-b)/np.max(np.abs(np.array([a, b])))
        except:
            return 0.0

def rowReduce(A, i, j, pivot):
    """
    subtract from row j a row i multiplied by factor
    factor = (element of row j) / (element of row i) at pivot column 
    operates on A in place
    """
    factor = A[j][pivot] / A[i][pivot]
    for k in range(len(A[j])):
        # we allow an accumulation of error 100 times larger than a single computation
        # this is crude but works for computations without a large dynamic range
        if relError(A[j][k], factor * A[i][k]) < 100 * np.finfo('float').resolution:
            A[j][k] = 0.0
        else: #subtract from row j a row i multiplied by factor, as a result A[j][pivot] = 0
            A[j][k] = A[j][k] - factor * A[i][k]

# stage 1 (forward elimination)
def forwardElimination(B):
    """
    Return the row echelon form of B
    """
    A = B.copy().astype(float)
    m, n = np.shape(A)
    for i in range(m-1):
        # Let lefmostNonZeroCol be the position of the leftmost nonzero value 
        # in row i or any row below it 
        leftmostNonZeroRow = m
        leftmostNonZeroCol = n
        ## for each row below row i (including row i)
        for h in range(i,m):
            ## search, starting from the left, for the first nonzero
            for k in range(i,n):
                if (A[h][k] != 0.0) and (k < leftmostNonZeroCol):
                    leftmostNonZeroRow = h
                    leftmostNonZeroCol = k
                    break
        # if there is no such position, stop
        if leftmostNonZeroRow == m:
            break
        # If the leftmostNonZeroCol in row i is zero, swap this row 
        # with a row below it
        # to make that position nonzero. This creates a pivot in that position.
        if (leftmostNonZeroRow > i):
            #swap rows leftmostNonZeroRow and i
            swapRows(A, i, leftmostNonZeroRow)
        
        # Use rowReduce function to create zeros below the pivot of row i
        # pivot is located at row i and collumn pivotPosition
        pivotPosition = leftmostNonZeroCol
        for h in range(i+1,m):
            rowReduce(A, i, h, pivotPosition)
    return A

#################### 

# If any operation creates a row that is all zeros except the last element,
# the system is inconsistent; stop.
def inconsistentSystem(A):
    """
    B is assumed to be in echelon form; return True if it represents
    an inconsistent system, and False otherwise
    """
    m, n = np.shape(A)
    for i in range(m):
        for j in range(n):
            if (A[i][j] != 0):
                if (j == n-1):
                    return True
                else:
                    break
    return False

def backsubstitution(B):
    """
    return the reduced row echelon form matrix of B
    """
    A = B.copy().astype(float)
    m, n = np.shape(A)
    for i in range(m):
        # If row i is all zeros, or if i exceeds the number of rows in A, stop.
        for j in range(n):
            if (A[i][j] != 0.0):
                break
        if (j == n-1):
            return A
        pivot = j
        A[i] = A[i] / A[i][pivot]
        # Divide row i by the value at its pivot collumn.
        # This creates 1 in the pivot position.
        
        for j in range(i):
            rowReduce(A, i, j, pivot)
    return A

#####################
#get a reduced echelon matrix from an augmented matrix of linear system
def GaussElimin(Aaug):
    #get AEchelon from Aaug, use already defined functions
    AEchelon = forwardElimination(Aaug)
    if (inconsistentSystem(Aaug)):
        print ('There is no solution')
    else:
        
        AReducedEchelon = backsubstitution(AEchelon)
        print ("Reduced Echelon Matrix:\n", AReducedEchelon)
    
#TEST THE PROGRAM

#test GaussElimin_incomplete_code for linear systems from HW1

Aaug = np.array([
[1,-0.2,-0.3, 320],
[-0.1, 1, -0.4, 90],
[-0.2,-0.5, 1, 150] 
])
    
print (GaussElimin(Aaug))

Aaug = np.array([
[1,-1, 0, 0 , 0, 0, -100],
[0, 1, -1, 0, 0, 0, 50],
[0, 0, 1, -1, 0, 0, 120],
[0, 0, 0, 1, -1, 0, 150],
[0, 0, 0, 0, 1, -1, -80],
[-1, 0, 0, 0, 0, 1, 100],
])
    
print (GaussElimin(Aaug))

Aaug = np.array([
[160, 110,-310],
[5, 2,-3.3],
[6, .1, -2.46],
[1, .4,-.64],
])
    
print (GaussElimin(Aaug))

Aaug = np.array([
[2, 0,-2, 0, 0],
[4, 0, 0, -2, 0],
[0, 2, 0, -1, 0],
])
    
print (GaussElimin(Aaug))

Aaug = np.array([
[1, 1, 1, 1, -1],
[1, 2, 4, 8, -5],
[0, 1, 2, 3, -2],
[0, 1, 4, 12, -9],
])
    
print (GaussElimin(Aaug))
