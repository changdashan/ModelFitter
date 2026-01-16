# Model Fitter
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
