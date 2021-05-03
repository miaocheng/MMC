# #########################################################################################
# Name:        statspace.py
# Several classic line-space methods are implemented in functions.
# Author:      Miao Cheng
#
# Created Date: 2018-1-31
# E-mail: mewcheng@gmail.com
#         mew_cheng@outlook.com
# Copyright:   (c) Miao Cheng 2018
# 
# #########################################################################################

import numpy as np
from numpy import linalg as la

from dataCala import *



def PCA(X, rDim, effDim):
    xDim, xSam = np.shape(X)
    tmp, mx = meanX(X)
    cX = X - repVec(mx, xSam)  
    
    if (xDim < effDim):
        XX = np.dot(cX, np.transpose(cX))
        D, E = np.linalg.eig(XX)
        ind = np.argsort(D)
        E = E[:, ind]
        D = D[ind]
        V = E[:, ::-1]
        D = D[::-1]
        
        #U, S, V = la.svd(XX)
        #V = U
        #D = S
    else:
        U, S, V = la.svd(cX)
        S = S ** 2
        V = U
        D = S
        
    if (rDim < xDim):
        V = V[:, 0:rDim]
        D = D[0:rDim]
    
    return V, D


#def LDA(X, L, rDim, effDim):
def LDA(X, L, effDim):
    xDim, xSam = np.shape(X)
    tmp, mx = meanX(X)
    
    xL = np.unique(L)
    nL = len(xL)
    B = np.zeros((xDim, 1))
    W = np.zeros((xDim, 1))
    
    for i in range(nL):
        ind = seArr(xL[i], L)
        nC = len(ind)
        cX = X[:, ind]
        
        tmp, meanc = meanX(cX)
        B = np.column_stack((B, meanc))
        wX = cX - repVec(meanc, nC)
        W = np.column_stack((W, wX))
        
    B = B[:, 1:nL+1]
    W = W[:, 1:xSam+1]
    
    B = B - repVec(mx, nL)
    U, S, V = la.svd(B)
    S = S ** (-1)
    d = np.diag(S)
    U = U[:, 0:len(S)]
    bV = np.dot(U, d)
    W = np.dot(np.transpose(bV), W)
    
    U, S, V = la.svd(W)
    tmp, rW = np.shape(U)
    wV = U[:, ::-1]
    D = S[::-1]
    
    V = np.dot(bV, wV)
    
    if (rDim < rW):
        V = V[:, 0:rDim]
        D = D[0:rDim]
        
    Q, R = la.qr(V)
    
    return Q, D
    
    
    
def MPCA(X, rDim):
    xRow, xCol, xSam = np.shape(X)
    cX, mx = meanX2D(X)
    
    #St = np.zeros((xRow, xRow))
    St = np.zeros((xCol, xCol))
    for i in range(xSam):
        tmp = cX[:, :, i]
        St += np.dot(np.transpose(tmp), tmp)
        
    U, S, V = la.svd(St)
    
    if (rDim < xRow):
        U = U[:, 0:rDim]
        S = S[0:rDim]
        
    return U, S


def D2PCA(X, rDim, cDim):
    xRow, xCol, xSam = np.shape(X)
    cX, mx = meanX2D(X)
    
    Sl = np.zeros((xRow, xRow))
    Sr = np.zeros((xCol, xCol))
    for i in range(xSam):
        tmp = cX[:, :, i]
        Sl += np.dot(tmp, np.transpose(tmp))
        Sr += np.dot(np.transpose(tmp), tmp)
        
    # ------------------------------------------------------------  
    U, ls, tmp = la.svd(Sl)
    
    if (rDim < xRow):
        U = U[:, 0:cDim]
        ls = ls[0:cDim]
        
    V, rs, tmp = la.svd(Sr)
    
    if (cDim < xCol):
        V = V[:, 0:rDim]
        rs = rs[0:rDim]
        
        
    return U, V, ls, rs
    
    
def MLDA(X, xL, rDim):
    xRow, xCol, xSam = np.shape(X)
    X, mx = meanX2D(X)
    
    uL = np.unique(xL)
    nL = len(uL)
    B = np.zeros((xCol, xCol))
    #W = np.zeros((xCol, xCol))
    
    for i in range(nL):
        ind = seArr(uL[i], xL)
        nSam = len(ind)
        cX = X[:, :, ind]
        
        wX, meanc = meanX2D(cX)
        tmp = meanc - mx
        tmp = np.dot(np.transpose(tmp), tmp)
        B += tmp
        
    U, S, V = la.svd(B)
    S = S ** (-0.5)
    U = U[:, 0:len(S)]
    S = np.diag(S)
    bV = np.dot(U, S)
    #W = np.dot(np.transpose(bV), W)
    #W = np.dot(W, bV)
    
    tmp, bDim = np.shape(bV)
    W = np.zeros((bDim, bDim))
    for i in range(nL):
        ind = seArr(uL[i], xL)
        nSam = len(ind)
        cX = X[:, :, ind]
        
        wX, meanc = meanX2D(cX)
        
        for j in range(nSam):
            tmp = wX[:, :, j]
            tmp = np.dot(tmp, bV)
            tmp = np.dot(np.transpose(tmp), tmp)
            W += tmp
            
    U, S, V = la.svd(W)
    tmp, rW = np.shape(U)
    U = U[:, ::-1]
    S = S[::-1]
    
    V = np.dot(bV, U)
    
    if (rDim < rW):
        V = V[:, 0:rDim]
        S = S[0:rDim]
        
    Q, R = la.qr(V)
    
    return Q, S


def D2LDA(X, xL, rDim, cDim):
    xRow, xCol, xSam = np.shape(X)
    X, mx = meanX2D(X)
    
    uL = np.unique(xL)
    nL = len(uL)
    Bl = np.zeros((xRow, xRow))
    Br = np.zeros((xCol, xCol))
    
    for i in range(nL):
        ind = seArr(uL[i], xL)
        nSam = len(ind)
        cX = X[:, :, ind]
        
        wX, meanc = meanX2D(cX)
        tmp = meanc - mx
        tml = np.dot(tmp, np.transpose(tmp))
        Bl += tml
        tmr = np.dot(np.transpose(tmp), tmp)
        Br += tmr
        
    U, S, V = la.svd(Bl)
    S = S ** (-0.5)
    U = U[:, 0:len(S)]
    S = np.diag(S)
    bU = np.dot(U, S)
    U, S, V = la.svd(Br)
    S = S ** (-0.5)
    U = U[:, 0:len(S)]
    S = np.diag(S)
    bV = np.dot(U, S)
    
    tmp, bRow = np.shape(bU)
    tmp, bCol = np.shape(bV)
    Wl = np.zeros((bRow, bRow))
    Wr = np.zeros((bCol, bCol))
    for i in range(nL):
        ind = seArr(uL[i], xL)
        nSam = len(ind)
        cX = X[:, :, ind]
        
        wX, meanc = meanX2D(cX)
        
        for j in range(nSam):
            tmp = wX[:, :, j]
            tml = lprj2D(bU, tmp)
            tml = np.dot(tml, np.transpose(tml))
            Wl += tml
            tmr = rprj2D(bV, tmp)
            tmr = np.dot(np.transpose(tmr), tmr)
            Wr += tmr
            
    U, S, V = la.svd(Wl)
    wU = U[:, ::-1]
    ls = S[::-1]
    U, S, V = la.svd(Wr)
    rs = S[::-1]
    wV = U[:, ::-1]
    
    U = np.dot(bU, wU)
    V = np.dot(bV, wV)
    
    tmp, uDim = np.shape(U)
    tmp, vDim = np.shape(V)
    
    if (rDim < uDim):
        U = U[:, 0:rDim]
        ls = ls[0:rDim]
        
    if (cDim < vDim):
        V = V[:, 0:cDim]
        rs = rs[0:cDim]
        
    P, R = la.qr(U)
    Q, R = la.qr(V)
    
    
    return P, Q, ls, rs

    
    