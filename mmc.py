
# #########################################################################################
# Name:        mmc.py
# This python file implements the maximum margin criterion (MMC) method and several variants.
# Author:      Miao Cheng
#
# Created Date: 2018-2-2
# E-mail:      mewcheng@gmail.com
#              mew_cheng@outlook.com
# Copyright:   (c) Miao Cheng 2018
# 
# #########################################################################################

import numpy as np
from numpy import linalg as la
import random
from dataCala import *


def MMC(X, xL, effDim, rDim):
    
    xDim, xSam = np.shape(X)
    X, mx = meanX(X)
    
    uL = np.unique(xL)
    nL = len(uL)
    
    B = np.zeros((xDim, 1))
    W = np.zeros((xDim, 1))
    for i in range(nL):
        ind = seArr(uL[i], xL)
        nC = len(ind)
        cX = X[:, ind]
        
        mX, meanc = meanX(cX)
        tmp = meanc - mx
        tmp = np.sqrt(nC)*tmp
        B = np.column_stack((B, tmp))
        #B = np.column_stack((B, meanc))
        wX = cX - repVec(meanc, nC)
        #tmp = np.tile(meanc, (nC, 1))
        #wX = cX - np.transpose(tmp)
        W = np.column_stack((W, wX))
        
    B = B[:, 1:nL+1]
    W = W[:, 1:xSam+1]
    
    #B = B - repVec(mx, nL)
    #tmp = np.tile(mx, (nL, 1))
    #B = B - np.transpose(tmp)
    
    if (xDim > effDim):
        B = np.dot(np.transpose(X), B)
        W = np.dot(np.transpose(X), W)
        
    Sb = np.dot(B, np.transpose(B))
    Sw = np.dot(W, np.transpose(W))
    St = Sb - 10*Sw
    
    U, s, V = la.svd(St)
    D, m = getRank(s)
    U = U[:, 0:m]
    
    if (xDim > effDim):
        ss = D ** (-0.5)
        ss = np.diag(ss)
        tmp = np.dot(X, U)
        bV = np.dot(tmp, ss)
        
        rU, rS, rV = la.svd(bV)
        U = rU[:, 0:m]
        s = rS[0:m]
        s = rS ** (-2)
        
        D = s[::-1]
        V = U[:, ::-1]
        
    if (rDim < len(U[0,])):
        V = U[:, 0:rDim]
        D = s[0:rDim]
    
    return V, D


def RMMC(X, xL, rDim, effDim, nb):
    
    xDim, xSam = np.shape(X)
    X, mx = meanX(X)
    mSam = int(np.floor(nb*rDim))
    ind = list(range(xSam))
    random.shuffle(ind)
    ind = ind[0:mSam]
    sX = X[:, ind]
    
    uL = np.unique(xL)
    nL = len(uL)
    
    B = np.zeros((xDim, 1))
    W = np.zeros((xDim, 1))
    for i in range(nL):
        ind = seArr(uL[i], xL)
        nC = len(ind)
        cX = X[:, ind]
        
        mX, meanc = meanX(cX)
        tmp = meanc - mx
        tmp = np.sqrt(nC)*tmp
        B = np.column_stack((B, tmp))
        wX = cX - repVec(meanc, nC)
        #tmp = np.tile(meanc, (nC, 1))
        #wX = cX - np.transpose(tmp)
        W = np.column_stack((W, wX))
        
    B = B[:, 1:nL+1]
    W = W[:, 1:xSam+1]
    
    if (xDim > effDim):
        B = np.dot(np.transpose(sX), B)
        W = np.dot(np.transpose(sX), W)
        
    Sb = np.dot(B, np.transpose(B))
    Sw = np.dot(W, np.transpose(W))
    
    St = Sb - 10*Sw
    
    U, s, V = la.svd(St)
    
    D, m = getRank(s)
    U = U[:, 0:m]
    
    if (xDim > effDim):
        ss = D ** (-0.5)
        ss = np.diag(ss)
        tmp = np.dot(sX, U)
        bV = np.dot(tmp, ss)
        
        rU, rS, rV = la.svd(bV)
        U = rU[:, 0:m]
        s = rS[0:m]
        s = rS ** (-2)
        
        D = s[::-1]
        V = U[:, ::-1]
        
    if (rDim < len(U[0,])):
        V = U[:, 0:rDim]
        D = D[0:rDim]
        
    
    return V, D


def getVX(X, P, Q, layer, aFun):
    xDim, xSam = np.shape(X)
    tmp = P[0]
    PX = np.dot(tmp, X)
    
    if (aFun == 'Sigmoid'):
        PX = 1 + np.exp(PX)
        PX = PX ** (-1)
        
    elif (aFun == 'Sine'):
        PX = np.sin(PX)
        
    elif (aFun == 'Linear'):
        pass
    
    #if (np.ndim(R) > 2):
        #V = R[:, :, 0]
    #else:
        #V = R[:, :]
    
    V =  Q[0]
    VX = np.dot(np.transpose(V), PX)
    
    if (layer-1 > 0):
        for i in range(layer-1):
            tmp = P[i+1]
            PX = np.dot(tmp, VX)
            
            if (aFun == 'Sigmoid'):
                PX = 1 + np.exp(PX)
                PX = PX ** (-1)
                
            elif (aFun == 'Sine'):
                PX = np.sin(PX)
                
            elif (aFun == 'Linear'):
                pass
            
            V = Q[i+1]
            VX = np.dot(np.transpose(V), PX)
            
    return VX


def get2DVX(X, Pl, Pr, Ql, Qr, layer):
    xRow, xCol, xSam = np.shape(X)
    tmp = Pl[0]
    PX = lprj2D(np.transpose(tmp), X)
    tmp = Pr[0]
    PX = rprj2D(tmp, PX)
    
    V =  Ql[0]
    VX = lprj2D(V, PX)
    V = Qr[0]
    VX = rprj2D(V, VX)
    
    if (layer-1 > 0):
        for i in range(layer-1):
            tmp = Pl[i+1]
            PX = lprj2D(np.transpose(tmp), VX)
            tmp = Pr[i+1]
            PX = rprj2D(tmp, PX)
            
            V = Ql[i+1]
            VX = lprj2D(V, PX)
            V = Qr[i+1]
            VX = rprj2D(V, VX)
            
            
    return VX
    
    
def LMMC(X, xL, rDim, effDim, hDim, layer, aFun):
    xDim, xSam = np.shape(X)
    uL = np.unique(xL)
    nL = len(uL)
    
    P = []
    tmp = np.random.randn(hDim, xDim)
    PX = np.dot(tmp, X)
    P.append(tmp)
    
    if (aFun == 'Sigmoid'):
        PX = 1 + np.exp(PX)
        PX = PX ** (-1)
        
    elif (aFun == 'Sine'):
        PX = np.sin(PX)
        
    elif (aFun == 'Linear'):
        pass
    
    tmp = hDim - rDim
    tmp = tmp / layer
    elayer = int(np.floor(tmp))
    fDim = []
    for i in range(layer):
        tmp = hDim - (i+1)*elayer
        if (i < layer-1):
            fDim.append(tmp)
        else:
            fDim.append(rDim)
    
    
    V, D = RMMC(PX, xL, fDim[0], effDim, 2)
    VX = np.dot(np.transpose(V), PX)
    
    Q = []
    Q.append(V)
    
    if (layer-1 > 0):
        
        for i in range(layer-1):
            xDim, xSam = np.shape(VX)
            tmp = np.random.randn(hDim, xDim)
            
            PX = np.dot(tmp, VX)
            P.append(tmp)
            
            if (aFun == 'Sigmoid'):
                PX = 1 + np.exp(PX)
                PX = PX ** (-1)
                
            elif (aFun == 'Sine'):
                PX = np.sin(PX)
                
            elif (aFun == 'Linear'):
                pass
            
            V, D = RMMC(PX, xL, fDim[i+1], effDim, 2)
            
            VX = np.dot(np.transpose(V), PX)
            
            Q.append(V)
            
    return P, Q, VX


def TLMMC(X, xL, rDim, effDim, hDim, aFun):
    xDim, xSam = np.shape(X)
    uL = np.unique(xL)
    nL = len(uL)
    
    P = []
    tmp = np.random.randn(hDim, xDim)
    PX = np.dot(tmp, X)
    P.append(tmp)
    
    if (aFun == 'Sigmoid'):
        PX = 1 + np.exp(PX)
        PX = PX ** (-1)
        
    elif (aFun == 'Sine'):
        PX = np.sin(PX)
        
    elif (aFun == 'Linear'):
        pass
    
    V, D = MMC(PX, xL, effDim)
    
    if (rDim < len(V[0,])):
        V = V[:, 0:rDim]
        D = D[0:rDim]
        
    Q = []
    Q.append(V)
    
    
    return P, Q, D


# #########################################################################################
# Function: MMMC(X, xL, effDim, rDim)
# This function implements the standard MMC with 2D data, while dimensionality of column
# direction is reduced.
# Para: X - xRow * xCol * xSam
#       xL - class labels of 2D samples
#       effDim - Dimensional thresheld for efficient calculation
#       rDim - Reduced dimensionality of data
# #########################################################################################
def MMMC(X, xL, effDim, rDim):
    xRow, xCol, xSam = np.shape(X)
    X, mx = meanX2D(X)
    
    uL = np.unique(xL)
    nL = len(uL)
    
    ind = list(range(xSam))
    random.shuffle(ind)
    ind = ind[0]
    anyx = X[:, :, ind]
    
    B = np.zeros((xCol, xCol))
    W = np.zeros((xCol, xCol))
    
    for i in range(nL):
        ind = seArr(uL[i], xL)
        nSam = len(ind)
        cX = X[:, :, ind]
        
        wX, meanc = meanX2D(cX)
        tmp = meanc - mx
            
        tmp = np.dot(np.transpose(tmp), tmp)
        B += tmp
        
        for j in range(nSam):
            tmp = wX[:, :, j]
                
            tmp = np.dot(np.transpose(tmp), tmp)
            W += tmp
            
    if (xCol > effDim):
        B = np.dot(anyx, B)
        B = np.dot(B, np.transpose(anyx))
        W = np.dot(anyx, W)
        W = np.dot(W, np.transpose(anyx))
        
    B = (float(1) / nL)*B
    W = (float(1) / xSam)*W
    T = B - 10*W
    
    U, s, V = la.svd(T)
    s, m = getRank(s)
    U = U[:, 0:m]
    
    if (xCol > effDim):
        s = s ** (-0.5)
        s = np.diag(s)
        tmp = np.dot(np.transpose(anyx), U)
        bV = np.dot(tmp, s)
        
        rU, rS, rV = la.svd(bV)
        U = rU[:, 0:m]
        s = rS[0:m]
        s = s ** (-2)
        
        s = s[::-1]
        V = U[:, ::-1]
        
    if (rDim < len(U[0,])):
        V = U[:, 0:rDim]
        s = s[0:rDim]
        
        
    return V, s


# #########################################################################################
# Function: 2D^2MMC(X, xL, effDim, rDim)
# This function implements the standard MMC with 2D data, while dimensionality of column
# direction is reduced.
# Para: X - xRow * xCol * xSam
#       xL - class labels of 2D samples
#       effDim - Dimensional thresheld for efficient calculation
#       rDim - Reduced dimensionality of data
# #########################################################################################
def D2MMC(X, xL, effDim, rDim, cDim):
#def D2MMC(X, xL, effDim):
    xRow, xCol, xSam = np.shape(X)
    tmp, mx = meanX2D(X)
    
    uL = np.unique(xL)
    nL = len(uL)
    
    # ---------------------------------------------------------------------------------------------------
    # Select the random sample data for efficient calculation. For high dimensionality of both directions, 
    # further improvement can be obtained with certain rows or columns of selected sample data.
    # ---------------------------------------------------------------------------------------------------
    ind = list(range(xSam))
    random.shuffle(ind)
    ind = ind[0]
    anyx = X[:, :, ind]
    
    Bl = np.zeros((xRow, xRow))
    Br = np.zeros((xCol, xCol))
    Wl = np.zeros((xRow, xRow))
    Wr = np.zeros((xCol, xCol))
    
    for i in range(nL):
        ind = seArr(uL[i], xL)
        nSam = len(ind)
        cX = X[:, :, ind]
        
        wX, meanc = meanX2D(cX)
        tmp = meanc - mx
        
        tml = np.dot(tmp, np.transpose(tmp))
        Bl += nSam*tml
        tmr = np.dot(np.transpose(tmp), tmp)
        Br += nSam*tmr
        
        for j in range(nSam):
            tmp = wX[:, :, j]
            #tmp = tmp - meanc
            
            tml = np.dot(tmp, np.transpose(tmp))
            Wl += tml
            tmr = np.dot(np.transpose(tmp), tmp)
            Wr += tmr
            
    # Handling Row Direction
    if (xRow > effDim):
        Bl = np.dot(np.transpose(anyx), Bl)
        Bl = np.dot(Bl, anyx)
        Wl = np.dot(np.transpose(anyx), Wl)
        Wl = np.dot(Wl, anyx)
        
    Tl = Bl - 10*Wl
    
    U, lS, V = la.svd(Tl)
    ls, m = getRank(lS)
    U = U[:, 0:m]
    
    if (xRow > effDim):
        ls = ls ** (-0.5)
        ls = np.diag(ls)
        tmp = np.dot(anyx, U)
        bV = np.dot(tmp, ls)
        
        U, lS, V = la.svd(bV)
        U = U[:, 0:m]
        ls = lS[0:m]
        ls = ls ** (-2)
        
        ls = ls[::-1]
        U = U[:, ::-1]
        
    P = U
    
    # Handling Column Direction
    if (xCol > effDim):
        Br = np.dot(anyx, Br)
        Br = np.dot(Br, np.transpose(anyx))
        Wr = np.dot(anyx, Wr)
        Wr = np.dot(Wr, np.transpose(anyx))
        
    Tr = Br - 10*Wr
    
    U, rS, V = la.svd(Tr)
    rs, m = getRank(rS)
    U = U[:, 0:m]
    
    if (xCol > effDim):
        rs = rs ** (-0.5)
        rs = np.diag(rs)
        tmp = np.dot(np.transpose(anyx), U)
        bV = np.dot(tmp, rs)
        
        U, rS, V = la.svd(bV)
        U = U[:, 0:m]
        rs = rS[0:m]
        rs = rs ** (-2)
        
        rs = rs[::-1]
        U = U[:, ::-1]
        
    Q = U
    
    tmp, uDim = np.shape(P)
    tmp, vDim = np.shape(Q)
    
    if (rDim < uDim):
        P = P[:, 0:rDim]
        ls = ls[0:rDim]
        
    if (cDim < vDim):
        Q = Q[:, 0:cDim]
        rs = rs[0:cDim]
        
        
    return P, Q, ls, rs


# #########################################################################################
# Function: layered 2D^2MMC(X, xL, rDim, effDim, hDim, layer, aFun)
# This function implements the layered MMC with 2D data.
# Para: X - xRow * xCol * xSam
#       xL - class labels of 2D samples
#       rDim - Reduced dimensionality of data
#       effDim - Dimensional thresheld for efficient calculation
# #########################################################################################
def LD2MMC(X, xL, rDim, effDim, hDim, layer):
    xRow, xCol, xSam = np.shape(X)
    uL = np.unique(xL)
    nL = len(uL)
    
    Pl = []
    
    tmp = np.random.randn(hDim, xRow)
    PX = lprj2D(np.transpose(tmp), X)
    Pl.append(tmp)
    Pr = []
    tmp = np.random.randn(xCol, hDim)
    PX = rprj2D(tmp, PX)
    Pr.append(tmp)
    
    tmp = hDim - rDim
    tmp = tmp / layer
    elayer = int(np.floor(tmp))
    fDim = []
    for i in range(layer):
        tmp = hDim - (i+1)*elayer
        if (i < layer-1):
            fDim.append(tmp)
        else:
            fDim.append(rDim)
    
    
    U, V, ls, rs = D2MMC(PX, xL, effDim, fDim[0], fDim[0])
    VX = lprj2D(U, PX)
    VX = rprj2D(V, VX)
    
    Ql = []
    Ql.append(U)
    Qr = []
    Qr.append(V)
    
    
    if (layer-1 > 0):
        
        for i in range(layer-1):
            xRow, xCol, xSam = np.shape(VX)
            
            tmp = np.random.randn(hDim, xRow)
            PX = lprj2D(np.transpose(tmp), VX)
            Pl.append(tmp)
            tmp = np.random.randn(xCol, hDim)
            PX = rprj2D(tmp, PX)
            Pr.append(tmp)
            
            U, V, ls, rs = D2MMC(PX, xL, effDim, fDim[i+1], fDim[i+1])
            
            VX = lprj2D(U, PX)
            VX = rprj2D(V, VX)
            
            Ql.append(U)
            Qr.append(V)
            
            
    return Pl, Pr, Ql, Qr, VX



