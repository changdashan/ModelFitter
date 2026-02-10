# Non-Linear Least Square Curve Fitting From Scratch
Description:

		ModelFitter is a Python class designed to fit a model function of the form f(x, p) to a set of measured data points (X, Y).

		The class is implemented from scratch and includes custom routines for:

			* Numerical estimation of partial derivatives of the model with respect to the parameters
			* Construction of the Jacobian matrix
			* Formation of the normal equations ((J^T * W * J))
			* Solution of the normal equations using Gaussian elimination

		ModelFitter employs the Levenberg-Marquardt algorithm to iteratively update the model parameters.
	    A damping factor (lambda) is applied to the diagonal elements of the normal equation matrix to control the gradient descent step size and improve convergence stability.

		To further refine each iteration step, the class uses a Golden Section Search line-search method to determine an optimal step length along the descent direction.

		The fitting process proceeds iteratively as follows:

			1. Numerically estimate partial derivatives and build the Jacobian matrix.
			2. Construct the normal equations (J^T J) with Levenberg-Marquardt damping.
			3. Solve the normal equations using Gaussian elimination to obtain parameter updates.
			4. Perform a Golden Section Search to determine the optimal step size.
			5. Update the parameters and repeat until convergence is achieved.

		The iteration stops when the relative parameter change satisfies: ((p[k+1] - p[k]) /  p[k]) < 1e-6 
		For ease of use, a convenience function called curve_fits is provided
		and can be imported directly into client code to perform model fitting without interacting with the class internals.
			
Usage:

			from ModelFitter import ModelFitter, curve_fits
			
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
