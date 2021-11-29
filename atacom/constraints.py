import numpy as np


class ViabilityConstraint:
    """
    Class of viability constraint
    f(q) + K_f df(q, dq) = 0
    g(q) + K_g dg(q, dq) <= 0
    """

    def __init__(self, dim_q, dim_out, fun, J, b, K):
        """
        Constructor of the viability constraint

        Args
        dim_q (int): Dimension of the controllable variable
        dim_out (int): Dimension of the constraint
        fun (function): The constraint function f(q) or g(q)
        J (function): The Jacobian matrix of J_f(q) or J_g(q)
        b (function): The term: dJ(q, dq) dq
        K (scalar or array): The scale variable K_f or K_g
        """
        self.dim_q = dim_q
        self.dim_out = dim_out
        self.fun_origin = fun
        if np.isscalar(K):
            self.K = np.ones(dim_out) * K
        else:
            self.K = K
        self.J = J
        self.b_state = b

    def fun(self, q, dq, origin_constr=False):
        if origin_constr:
            return self.fun_origin(q)
        else:
            return self.fun_origin(q) + self.K * (self.J(q) @ dq)

    def K_J(self, q):
        return np.diag(self.K) @ self.J(q)

    def b(self, q, dq):
        return self.J(q) @ dq + self.K * self.b_state(q, dq)


class ConstraintsSet:
    """
    The class to gather multiple constraints
    """

    def __init__(self, dim_q):
        self.dim_q = dim_q
        self.constraints_list = list()
        self.dim_out = 0

    def add_constraint(self, c: ViabilityConstraint):
        self.dim_out += c.dim_out
        self.constraints_list.append(c)

    def fun(self, q, dq, origin_constr=False):
        ret = np.zeros(self.dim_out)
        i = 0
        for c in self.constraints_list:
            ret[i:i + c.dim_out] = c.fun(q, dq, origin_constr)
            i += c.dim_out
        return ret

    def K_J(self, q):
        ret = np.zeros((self.dim_out, self.dim_q))
        i = 0
        for c in self.constraints_list:
            ret[i:i + c.dim_out] = c.K_J(q)
            i += c.dim_out
        return ret

    def b(self, q, dq):
        ret = np.zeros(self.dim_out)
        i = 0
        for c in self.constraints_list:
            ret[i:i + c.dim_out] = c.b(q, dq)
            i += c.dim_out
        return ret
