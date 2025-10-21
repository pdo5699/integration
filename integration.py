from collections.abc import Iterable
from copy import copy
import numpy as np


class Rkf54():
    """
    A class for Runge-Kutta-Fehlberg 5(4) integration using Cass-Karp 
    coefficients.

    Steps are taken at a 5th order with the 4th order being used for truncation
    error estimation.

    Solves the function 

        dy
        -- = f(t, y)   y(t0) = y0 
        dt

    Parameters
    ----------
    f : function
        A function to be integrated. Must have the call f(t, y, *args).
    tspan : 2-element tuple, list
        The limits of integration.
    y0 : list
        The value of y at the lower limit of integration.

    Optional
    --------
    args : tuple
        Additional arguments which is passed to 'f'.
    tol : float
        The maximum allowable estimated truncation error on each integration
        step. Default value is 10% of 'tspan'.
    t_eval : sequence
        A monotonic sequence to time points at which the solution must be 
        solved. When used, output parameters 't' and 'y' will be at the points 
        in t_eval and not necessarily at all the successful steps. Default value
        is None.
    first_step : float
        The size of the first integration step. Default value is 10% of 'tspan'.
    dense_output : bool
        Indicates if additional parameters should be stored to later be able to
        interpolate solutions. Default value is False.
    max_rejected_steps : int
        The maximum number of integration steps before the integration is 
        prematurely halted. Default value is 20.
    verbose : int
        Indicates the level of information to return.
        0  : No extra information.
        1  : The final result, total estimated truncation error, and number of
             function calls will be printed to the standard output at the end of
             the integration.
        2  : Information provided at each successful integration step.

    
    Outputs
    -------
    success : bool
        Indicates if the integration was successful.
    exit_flag : int
        A integer indicating the type of completion of the integation.
    t : ndarray
        Vector of the successful time steps starting with the initial condition.
    y : ndarray
        Array of the y-values at each successful time step with the initial
        condition.
    total_truncation_error_estimate
        The sum of all the estimated truncation errors at each step.

    Usage
    -----
    Instantiate an object of the Rkf54 class, and provide at least the three 
    necessary parameters:
      
        int = Rkf54(f, tspan, y0)

    See the optional parameters above for more settings. The integration default 
    tolerance on the truncation error at each step is 10^-4.

    Integrate the function 'f' from the limits in 'tspan' with an initial value
    of 'y0' by calling the method 'integrate()'.
        
        int.integrate()

    The successful steps are stored in the object for each value of the time
    step (independent variable) 'int.t' and the correspnding function values
    'int.y'.

    If you want to interpolate values between the successful steps using the 5th
    order RKF approximation, then before calling the 'integrate()' method, set 
    the parameter 'dense_output=True'.

        int.dense_output = True
    
    or at instantiation,
    
        int = Rkf54(f, tspan, y0, dense_output=True)

    Then, an interpolated value of 'y' at a time 't' using the method 
    'from_dense()'.

        y(t) = int.from_dense(t)


    References
    ----------
    Numerical Methods for Engineers. 25.5.2 Runge-Kutta Fehlberg. 6th ed. 
    """

    # Coefficients for the integration approximation.
    a = [0.0, 1/5, 3/10, 3/5, 1.0, 7/8]
    b = [[1/5],
         [3/40, 9/40],
         [3/10, -9/10, 6/5],
         [-11/54, 5/2, -70/27, 35/27],
         [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]]
    c4 = [37/378, 250/621, 125/594, 512/1771]
    c5 = [2825/27648, 18575/48384, 13525/55296, 277/14336, 1/4]

    # Output parameter initialization.
    total_truncation_error_estimate = 0.0

    # Internal parameter initialization.
    _dense_output_terms = list()
    step_counter = 1
    rejected_step_counter = 0
    function_call_counter = 0

    # Boolean parameter initialization.
    step_converge = False
    initialization_error = False
    
    # Parameters for outputting information.
    verbose = 0
    verbose_line_counter = 0

    # Parameters for outputing solutions per input 't_eval'
    _set_evaluation_times_bool = False
    _evaluation_times_index = 0

    def __init__(self, f, tspan, y0,
                 args=tuple(),
                 tol=1e-4,
                 first_step=None,
                 t_eval=None,
                 dense_output=False,
                 verbose=0,
                 max_rejected_steps=20):

        # Validate the inputs.
        error_str = ""
        if not callable(f):
            self.initialization_error = True
            error_str += "- Input function 'f' is not callable.\n"
        if not isinstance(tspan, Iterable):
            self.initialization_error = True
            error_str += "- Input 'tspan' must be a 2-element tuple or list.\n"
        else:
            if len(tspan) != 2:
                self.initialization_error = True
                error_str += "- Input 'tspan' must be two elements in length.\n"
            elif tspan[0] == tspan[1]:
                self.initialization_error = True
                error_str += ("- Input 'tspan' elements must have different "
                              +"magnitudes.\n")
        if not isinstance(y0, Iterable):
            self.initialization_error = True
            error_str += "- Input 'y0' must be a iterable, such as a list.\n"
        if not isinstance(tol, float):
            self.initialization_error = True
            error_str += "- Option 'tol' must be a float.\n"
        if t_eval is not None:
            if not isinstance(t_eval, Iterable):
                self.initialization_error = True
                error_str += ("- Option 't_eval' must be a one-dimensional "
                              +"array.\n")
            if isinstance(t_eval, np.ndarray):
                if len(np.shape(t_eval)) > 1:
                    self.initialization_error = True
                    error_str += ("- Option 't_eval' must be a one-dimensional "
                                  +"array.\n")
            if not self._is_monotonic(np.asarray(t_eval), tspan):
                self.initialization_error = True
                error_str += "- Option 't_eval' must be monotonic."
        if type(dense_output) is not bool:
            self.initialization_error = True
            error_str += "- Option 'dense_output' must be a boolean.\n"
        if verbose not in [0,1,2]:
            self.initialization_error = True
            error_str += "- Option 'verbose' must be an integer in [0, 1].\n"
        if not isinstance(max_rejected_steps, int):
            self.initialization_error = True
            error_str += ("-Option 'max_rejected_steps' must be a positive "
                          "integer.\n")
        elif max_rejected_steps < 0:
            print("Input option 'max_rejected_steps' should be a positive "
                  +"integer. Assuming \nabs(max_rejected_steps) instead.")
        if self.initialization_error:
            print("Initialization errors occured!")
            print(error_str)
        if self.initialization_error:
            raise Exception("An error occured in the initialization.")

        # Initialize the parameters with there input values.
        self.f = f
        self.tspan = tspan
        self.tk = tspan[0]
        self.y0 = y0
        self.yk = y0
        self.args = args
        if first_step is None: 
            self.step_size = 0.1*abs(self.tspan[-1]-self.tspan[0])
        else:
            self.step_size = abs(first_step)
        self.tol = tol
        self.dense_output=dense_output
        self.verbose = verbose
        self.max_rejected_steps = abs(max_rejected_steps)
        if t_eval is not None:
            self.t_eval = t_eval
            self._set_evaluation_times_bool = True

        # Calculate some parameters about the problem.
        self.n = len(y0)
        self.direction = ((self.tspan[-1]-self.tspan[0])
                         /abs((self.tspan[-1]-self.tspan[0])))

        self.t = np.empty((0,))
        self.y = np.empty((self.n, 0))

    
    def _is_monotonic(self, a, tspan):
        if tspan[0] == tspan[-1]:
            return False

        direction = ((tspan[-1]-tspan[0])/abs((tspan[-1]-tspan[0])))
        if direction == 1:
            return np.all(a[:-1] < a[1:])
        elif direction == -1:
            return np.all(a[:-1] > a[1:])


    def _calculate_k(self):
        k0 = np.zeros(self.n)
        k1 = np.zeros(self.n)
        k2 = np.zeros(self.n)
        k3 = np.zeros(self.n)
        k4 = np.zeros(self.n)
        k5 = np.zeros(self.n)

        k0 = self.f(self.tk+self.a[0]*self.direction*self.step_size, 
                    self.yk,
                    *self.args)
        k1 = self.f(self.tk+self.a[1]*self.direction*self.step_size, 
                    self.yk
                    +self.direction*self.step_size
                    *(self.b[0][0]*k0),
                    *self.args)
        k2 = self.f(self.tk+self.a[2]*self.direction*self.step_size, 
                    self.yk
                    +self.direction*self.step_size
                    *(self.b[1][0]*k0+self.b[1][1]*k1),
                    *self.args)
        k3 = self.f(self.tk+self.a[3]*self.direction*self.step_size, 
                    self.yk
                    +self.direction*self.step_size
                    *(self.b[2][0]*k0+self.b[2][1]*k1
                    +self.b[2][2]*k2),
                    *self.args)
        k4 = self.f(self.tk+self.a[4]*self.direction*self.step_size,
                             self.yk
                             +self.direction*self.step_size
                             *(self.b[3][0]*k0+self.b[3][1]*k1
                              +self.b[3][2]*k2
                              +self.b[3][3]*k3),
                             *self.args)
        k5 = self.f(self.tk+self.a[5]*self.direction*self.step_size, 
                             self.yk
                             +self.direction*self.step_size
                             *(self.b[4][0]*k0+self.b[4][1]*k1
                              +self.b[4][2]*k2+self.b[4][3]*k3
                              +self.b[4][4]*k4),
                             *self.args)
        self.function_call_counter += 6
        self.k = [k0, k1, k2, k3, k4, k5]


    def _calculate_4th_order_estimate(self,):
        self.y4 = np.zeros(self.n)
        for i in range(self.n):
            self.y4[i] = (self.yk[i]+self.direction*self.step_size
                      *(self.c4[0]*self.k[0][i]
                      +self.c4[1]*self.k[2][i]
                      +self.c4[2]*self.k[3][i]
                      +self.c4[3]*self.k[5][i]))


    def _calculate_5th_order_estimate(self,):
        self.y5 = np.zeros(self.n)
        for i in range(self.n):
            self.y5[i] = (self.yk[i]+self.direction*self.step_size
                        *(self.c5[0]*self.k[0][i]
                        +self.c5[1]*self.k[2][i]
                        +self.c5[2]*self.k[3][i]
                        +self.c5[3]*self.k[4][i]
                        +self.c5[4]*self.k[5][i]))


    def compute_y5_from_k(self, y, k, d, h):
        y5 = np.zeros(self.n)
        for i in range(self.n):
            y5[i] = (y[i]+d*h
                        *(self.c5[0]*k[0][i]
                        +self.c5[1]*k[2][i]
                        +self.c5[2]*k[3][i]
                        +self.c5[3]*k[4][i]
                        +self.c5[4]*k[5][i]))
        return y5


    def _calculate_truncation_error(self,):
        max_error = -1.0

        for i in range(self.n):
            abs_error = abs(self.y5[i]-self.y4[i])
            #if self.y4[i] == 0:
            #    rel_error = abs(self.y5[i]-self.y4[i])
            #else:
            #    rel_error = abs_error
            max_error = max(max_error, abs_error)

        self.truncation_error_estimate = max_error
        

    def _take_step(self,):
        self._calculate_k()
        self._calculate_4th_order_estimate()
        self._calculate_5th_order_estimate()
        self._calculate_truncation_error()


    def _evaluate_step(self,):
        if self.truncation_error_estimate <= self.tol:
            self.step_converge = True


    def _accept_step(self,):
        # The step 't' and 'y' values should only be stored if the step was 
        # successfull, and if the 't' value is in 't_eval', if used.
        store_step = False
        if not self._set_evaluation_times_bool:
            store_step = True
        elif self.tk+self.direction*self.step_size in self.t_eval:
            store_step = True
            self._evaluation_times_index += 1

        if store_step:
            self.t = np.append(self.t, self.tk+self.direction*self.step_size)
            self.y = np.hstack((self.y, np.resize(self.y5, (self.n,1))))
        self.tk = self.tk+self.direction*self.step_size
        self.yk = copy(self.y5)
        self.total_truncation_error_estimate += self.truncation_error_estimate
        if self.dense_output:
            self._store_dense_solution()

        if self.verbose != 0:
            self._print_accept_step()

        self._step_resize()

        self.step_converge = False
        self.step_counter += 1
        self.rejected_step_counter = 0

    
    def _print_accept_step(self,):
        if self.verbose == 2:
            if self.verbose_line_counter%20 == 0:
                print(f"|----------------------------------------------------------------------")
                print(f"|      t      |     |y|     |      h      |     de      |     Tde     |")
                print(f"|----------------------------------------------------------------------")
            print(f"| {self.tk:+.4e} | {np.linalg.norm(self.yk):+.4e} | "
                  +f"{self.step_size:+.4e} | "
                  +f"{self.truncation_error_estimate:+.4e} | "
                  +f"{self.total_truncation_error_estimate:+.4e} |")
        self.verbose_line_counter += 1


    def _store_dense_solution(self,):
        step_terms = dict()
        step_terms['t_range'] = (self.t[-2], self.t[-1])
        step_terms['y_range'] = (self.y[-2,:], self.y[-1,:])
        step_terms['k_terms'] = self.k
        step_terms['d'] = self.direction
        self._dense_output_terms.append(step_terms)


    def from_dense(self, t):
        if not self.dense_output:
            raise Exception("A dense output is needed, but the object doesn't "
                            +"have a dense solution. "+"Recompute with the "
                            +"option 'dense_ouput=True'")

        for c in self._dense_output_terms:
            if abs(t) >= abs(c['t_range'][0]) and abs(t) <=abs(c['t_range'][1]):
                y5 = self.compute_y5_from_k(c['y_range'][0], c['k_terms'],
                                            c['d'], 
                                            abs(t)-abs(c['t_range'][0]))
        return y5


    def _step_resize(self,):
        self.step_size *= 0.9*(self.tol/self.truncation_error_estimate)**0.2


    def _reject_step(self,):
        self.step_converge = False
        self.rejected_step_counter += 1
        if self.rejected_step_counter >= self.max_rejected_steps:
            success = False
            raise Exception()
        self._step_resize()


    def _check_step_size(self,):
        # Limit the step to the next value in t_eval if necessary.
        if (self._set_evaluation_times_bool 
                and (abs(self.tk + self.direction*self.step_size) 
                > abs(self.t_eval[self._evaluation_times_index]))):
            self.step_size = abs(self.t_eval[self._evaluation_times_index]
                                 - self.tk)

        # Limit the final step to reach the upper limit of integration.
        if abs(self.tk + self.direction*self.step_size) > abs(self.tspan[-1]):
            self.step_size = abs(self.tspan[-1]-self.tk)


    def _record_initial_values(self,):
        store_initial_value = False
        if not self._set_evaluation_times_bool:
            store_initial_value = True
        elif self.t_eval[0] == self.tspan[0]:
            store_initial_value = True
            self._evaluation_times_index += 1
        if store_initial_value:
            self.t = np.append(self.t, self.tk)
            self.y = np.hstack((self.y, np.resize(self.y0, (self.n, 1))))


    def integrate(self,):
        self._record_initial_values()
        while abs(self.tk) < abs(self.tspan[-1]):
            self._check_step_size()
            self._take_step()
            self._evaluate_step()
            if self.step_converge:
                self._accept_step()
            else:
                self._reject_step()

        self.exit_flag = 1
        if self.exit_flag == 1:
            self.success = True
        else:
            self.success = False

        if self.verbose != 0:
            print(f"Found y(tf) = {self.y[:,-1]},\n"
                  +f"in {self.step_counter} steps, {self.function_call_counter}"
                  +" function calls.")
            print(f"Total error estimate = "
                  +f"{self.total_truncation_error_estimate:18g}")

