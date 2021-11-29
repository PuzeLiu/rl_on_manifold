import numpy as np
import pinocchio as pino
from numpy.linalg import norm, solve

IT_MAX = 1000
eps = 1e-4
DT = 1e-1
damp = 1e-12


def clik(model, data, fDes, q, idx):
    q_cur = q.copy()
    i = 0
    while True:
        pino.forwardKinematics(model, data, q_cur)
        pino.updateFramePlacements(model, data)

        dMi = fDes.actInv(data.oMf[idx])
        lin_err = -dMi.translation
        ang_err = pino.log3(dMi.rotation)
        err = np.concatenate([lin_err, ang_err])
        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = pino.computeFrameJacobian(model, data, q_cur, idx, pino.LOCAL_WORLD_ALIGNED)
        v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q_cur = pino.integrate(model, q_cur, v*DT)
        i += 1
    idx = np.where(q_cur > model.upperPositionLimit)
    q_cur[idx] -= np.pi * 2
    idx = np.where(q_cur < model.lowerPositionLimit)
    q_cur[idx] += np.pi * 2
    if not (np.all(model.lowerPositionLimit < q_cur) and np.all(q_cur < model.upperPositionLimit)):
        return False, q_cur
    return success, q_cur


def fk(model, data, q, idx):
    pino.forwardKinematics(model, data, q)
    pino.updateFramePlacement(model, data, idx)
    return data.oMf[idx]
