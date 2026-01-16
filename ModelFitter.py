"""
File: ModelFitter.py

Description: 
			ModelFitter is a class that can be used to fit a model of the function form f(x, p) to a set of measured data (x, y). 
			It uses a numerical method to estimate the partial derivatives of the model with respect to the parameters 
			to build the Jacobian matrix and obtains the normal equation about the delta of the parameters ((The transpose of J) x J). 
			It applies the Levenburgh-Marquardt method and impose a lambda value to the diagonal elements of the normal equation matrix.
			It uses Gaussian elemination method to solve the normal equations for the delta parameters.
			For the next iteration, it uses a line search method called Golden Section Search to determine the gradient descending step size
			and the parameters. It will repeat (iterate) the process until abs((p[k+1]-p[k])/p[k]) < epsilon (1e-6)

			To simplify the usage, a function called curve_fits can be imported and called in the client code:
			
Usage:		from ModelFitter import ModelFitter, curve_fits
			
			def model_antelope_population(x, *p):
				a = p[0]
				k = p[1]
				f = a * np.exp(k * x)
				return f

			X = np.array([1,2,4,5,8])
			Y = np.array([3,4,6,11,20])
			P = np.array([2, 1])
			model = model_antelope_population
			popt, pcov = curve_fits(model, X, Y, p0=P)
			print(popt)
			print(pcov)
				
Author:		Dashan Chang

Date:		01/16/2025
"""
#---------------------------------------------------------------------------------------------------
import numpy as np

class ModelFitter:
	
	def __init__(self, X, Y, ModelFunc, P, sigma=None, DerivativeFunc=None, MaxIterations=50, ftol=1e-6, xtol=1e-6):
		self.X = X
		self.Y = Y
		self.Sigma = [1] * len(X) if sigma is None else sigma										# assume sigma is 1 if not provided. inverse of weight, 1/sigma
		self.Model = ModelFunc																		# model function will have the format f(x, *p) where x is the independent variable and p is the parameter array
		self.P = P
		self.Derivative = self.DerivativeEstimate if DerivativeFunc is None else DerivativeFunc		# user does not need to provide it. We will use numberic method to get the partial derivative
		self.PP = None																				# Previous Parameter or parameters estimated at the current iteration
		self.DP = None																				# Delta Parameters
		self.epsilon = 0.000001 if xtol != 0 else xtol
		self.MaxIterations = 50 if MaxIterations == 0 else MaxIterations
		self.Iteration = 0
		self.x = [0] * len(X)
		self.y = [0] * len(Y)
		self.xError = [0] * len(X)
		self.yError = [0] * len(Y)
		self.xStandardDeviation = None
		self.yStandardDeviation = None
		self.ModelName = ModelFunc.__name__
		self.ModelFormula = ""
		self.ParameterNames = "" 
		self.lamda = 1
		self.pcov = []
		self.popt = []
	

	#-----------------------------------------------------------------------------------------
	def DerivativeEstimate(self, x, P):
		ph = P.copy()
		m = len(P)
		df = [0] * m
		for i in range(m):
			p = ph[i]
			h = 0.0001
			ph[i] = p + h
			f1 = self.Model(x, *ph)
			ph[i] = p - h
			f2 = self.Model(x, *ph)
			df[i] = (f1 - f2)/(2.0 * h)            # Three-point central difference formula
		return df

	#-----------------------------------------------------------------------------------------
	def DerivativeEstimate2(self, x, P):
		ph = P.copy()
		m = len(P)
		df = [0] * m

		for i in range(m):
			p = ph[i]
			h = 0.0001
			ph[i] = p + 2 * h
			f1 = - self.Model(x, *ph)
			ph[i] = p + h
			f2 = 8 * self.Model(x, *ph)
			ph[i] = p - h
			f3 = -8 * self.Model(x, *ph)
			ph[i] = p - 2 * h
			f4 = self.Model(x, *ph)
			df[i] = (f1 + f2 + f3 + f4) / (12 * h)

		return df

	#---------------------------------------------------------------------------------------------
	# Calculate the difference of measured and predicted Y values and standard deviation 
	def CalcStandarDeviation(self):	
		sgma = self.Sigma
		s = 0;
		ySD = 0;
		n = len(self.X)
		p = self.P
		for i in range(n):
			x = self.X[i]
			y = self.Model(x, *p)
			self.y[i] = y
			self.yError[i] = self.Y[i] - y
			s += np.pow(self.yError[i] / sgma[i], 2)
		
		ySD = np.sqrt(s/len(self.X))
		self.yStandardDeviation = ySD
		return ySD

	#----------------------------------------------------------------------------------------
	# this is the objective function to be minimized 
	def SqrtOfSquresSum(self):   #calculate the square root of the sum of the squares of the residual/difference/deviation
		SumOfSquares = 0
		X = self.X
		Y = self.Y
		sgma = self.Sigma
		P = self.P
		n = len(X)
		for i in range(n):
			x = X[i];
			y = self.Model(x, *P)
			s = (Y[i] - y) / sgma[i]
			SumOfSquares += (s ** 2)
		return np.sqrt(SumOfSquares / n)

	#-----------------------------------------------------------------------------------------
	# constructing Jacobian matrix with the partial derivatives of the sum of the residual squares 
	# with respect to each model parameter at each X point, which is a m * (m+1) matrix  
	# 
	def GetJacobianMatrix(self):
		m = len(self.P)											#number of parameters of the model
		n = len(self.X)											#number of measured x values
		sgma = self.Sigma

		J = [0] * n			
		for i in range(n):
			J[i] = [0] * m
			x = self.X[i]
			y = self.Y[i]
			p = self.P
			df = [0] * m
			df = self.Derivative(x, p)							# df - the partial derivative of the model with respect to each parameter.
			for j in range(m):
				J[i][j] = df[j] / sgma[i]
		return J

	#-----------------------------------------------------------------------------------------
	# Constructing the matrix for the normal equations for the Jacobian matrix with the partial derivatives of the sum of the residual squares 
	# with respect to each model parameter at each X point, which is a m * (m+1) matrix  
	# and Levenburgh-Marquardt method is applied to the diagonal elements of the matrix.
	def BuildNormalEquationMatrix(self):						
		m = len(self.P)
		n = len(self.X)
		sgma = self.Sigma

		r = [0] * n												# r is the residual vector of Y - f(x, p). aka, delta y
		J = [0] * n			
		for i in range(n):
			J[i] = [0] * m
			x = self.X[i]
			y = self.Y[i]
			p = self.P
			df = [0] * m
			r[i] = (y - self.Model(x, *p)) / sgma[i]			# residual, or, the difference of the meatured value and the model predicted values at the current parameters
			df = self.Derivative(x, p)							# df - the partial derivative of the model with respect to each parameter.
			for j in range(m):
				J[i][j] = df[j] / sgma[i]
	
		a = [0] * m
		for i in range(m):
			a[i] = [0] * (m+1)
			
			hf = 0
			for k in range(n):
				hf += J[k][i] * r[k]
			a[i][m] = hf

			for j in range(m):
				hf = 0;
				for k in range(n):
					hf += J[k][i] * J[k][j]
				a[i][j] = hf
				if i == j:
					a[i][i] += self.lamda						# Marquardt's method is applied to the diagonal elements: lambda*I 
					
		return a
	
	#-----------------------------------------------------------------------------------------
	# This is the GaussianEliminationMethod that solves simutaneous equation systems.
	# The returned vector is the solution of e.g., x, y, z etc, 
	"""
	a1 = [
		[2,5,-9,3,151],
		[5,6,-4,2,103],
		[3,-4,2,7,16],
		[11,7,4,-8,-32]
	]
	# solution: [3, 5, -11, 7]


	a2 = [
		[3, 2, 36],
		[5, 4, 64]
	]
	# solution: [8, 6]

	a3 = [
		[2,3,4,-5,-6],
		[6,7,-8,9,96],
		[10,11,12,13,312],
		[14,15,16,17,416]
	]
	# solution: [3, 5, 7, 11]
	"""
	
	def GaussianEliminationMethod(self,a):
		n = len(a)
		x = [0] * n
		k = -1
		i0 = 0

		while k < n - 1 and i0 > -1:
			k += 1
			i0 = k
			while np.abs(a[i0][k]) < 0.0001 and i0 < n - 1:
				i0 += 1

			if i0 == n - 1  and np.abs(a[i0][k]) < 0.000001:
				return None

			if i0 > k:
				for j in range(k, n):
					temp = a[k][j]
					a[k][j] = a[i0][j]
					a[i0][j] = temp

			t = 1.0 / a[k][k]
			for j in range(k+1, n + 1):
				a[k][j] = a[k][j] * t

			for i in range(k+1, n):
				for j in range(k + 1, n + 1):
					a[i][j] -= a[i][k] * a[k][j]

		x[n-1] = a[n-1][n]
		for i in range(n-1, -1, -1):
			x[i] = a[i][n]
			for j in range(i+1, n):
				x[i] -= x[j] * a[i][j]

		return x;

	#----------------------------------------------------------------------------------------
	# Golden-Section Search algorithm, one of the line search algorithms to determine 
	# the next set of parameters so that the objective function is descending the fastest.
	def GoldenSectionSearch(self):
		sqrtOfSqrSum = 0;
		b1 = 0;
		b2 = 1;
		c = 0.5 * (np.sqrt(5) - 1)         #0.618
		a2 = b1 + (b2 - b1) * c
		
		m = len(self.P)

		self.PP = self.P.copy()
		p0 = self.PP
		dp = self.DP
		p = [0] * m

		for i in range(m):
			p[i] = p0[i] - a2 * dp[i]

		self.P = p;
		f2 = self.SqrtOfSquresSum()

		a1 = b1 + b2 - a2
		for i in range(m):
			p[i] = p0[i] + a1 * dp[i]

		self.P = p;
		f1 = self.SqrtOfSquresSum()

		while (np.abs(f2 - f1) / (f2 + f1)) > 0.05:
			if f1 < f2:
				b2 = a2
				a2 = a1
				f2 = f1
				a1 = b1 + b2 - a2
				for i in range(m):
					p[i] = p0[i] + a1 * dp[i]

				self.P = p
				f1 = self.SqrtOfSquresSum()
				sqrtOfSqrSum = f1
			else:
				b1 = a1
				a1 = a2
				f1 = f2
				a2 = b1 + b2 - a1
				for i in range(m):
					p[i] = p0[i] + a2 * dp[i]

				self.P = p
				f2 = self.SqrtOfSquresSum()
				sqrtOfSqrSum = f2

		for i in range(m):
			p[i] = p0[i] + 0.5 * (b1 + b2) * dp[i]						#p[k+1] = p[k] + f * dp   (aka, delta Beta, which is solved by GaussianEleminationMethod), a method called Shift-butting.

		self.P = p
		sqrtOfSqrSum = self.SqrtOfSquresSum()

		return sqrtOfSqrSum
	
	#-----------------------------------------------------------------------------------------
	# Calculate parameter's variances (digonal elements) and covariances at the optimal parameters
	def CalcParamVariance(self):
		J = np.array(self.GetJacobianMatrix())
		JT = J.T
		JT_Dot_J = JT @ J
		inv_JT_J = np.linalg.inv(JT_Dot_J)	
		Q = (self.SqrtOfSquresSum() ** 2) * len(self.X)
		N = len(self.X)
		M = len(self.P)
		rv = Q/(N-M)
		pcov = inv_JT_J * rv
		self.pcov = pcov
		return pcov
	
	#-----------------------------------------------------------------------------------------
	# After the class is initialized with the model, X, Y, p0 parameter, call this method to start the iteration.
	def StartIteration(self):	

		# The square root of the sum of the square of Y - y(x, p).  
		residual = self.SqrtOfSquresSum()
		leastResidual = 0
		epsilon = self.epsilon

		k = 0
		lamda = 1.0
		Converged = False
		MaxIterations = self.MaxIterations

		np.set_printoptions(suppress=True, precision=8)

		while not Converged and k <= MaxIterations:
			leastResidual = residual
			k += 1
			
			self.Iteration = k
			a = self.BuildNormalEquationMatrix()
			dp = self.GaussianEliminationMethod(a)					#dp is Delta of the Parameter, the gradient decending step size for each paramter
			self.DP = dp
			
			p = self.P
			epsilon = self.epsilon
			
			Converged = False
			for i in range(len(p)):
				epsiln = np.abs(dp[i] / p[i])
				if epsiln < epsilon:
					Converged = True
				else:
					Converged = False

			if not Converged:
				residual = self.GoldenSectionSearch()
				self.lamda *= 0.8

		leastResidual = residual
		self.popt = self.P
		
		print("The Least Residual Error (square root): ", leastResidual)

		for i in range(len(p)):
			print(f"Parameter: {i} value: {p[i]}")

		self.CalcStandarDeviation()
		self.CalcParamVariance()
	
#---------------------------------------------------------------------------------------------------
# A simplified function to simulate to the curve_fit method of scipy.optimize
def curve_fits(Model, X, Y, p0=None, sigma=None, Derivative=None, MaxIterations=50, ftol=1e-6, xtol=1e-6, full_output=False):
	modelFitter = ModelFitter(X, Y, Model, p0, sigma, Derivative, MaxIterations, ftol, xtol)
	modelFitter.StartIteration()
	popt = modelFitter.popt
	pcov = modelFitter.pcov
	if full_output:
		return popt, pcov, modelFitter
	else:
		return popt, pcov
	
	
# example model function format:
"""	
def model_antelope_population(x, *p):
    a = p[0]
    k = p[1]
    f = a * np.exp(k * x)
    return f
"""	