# Import Python packages
import numpy as np


##############################################
#          Interpolation routine             #
##############################################
def interpolation(x_arr, y_arr, xx, indx, interp_list, return_coefs=False):

    '''
    This function interpolates with the Steffen 1990 algorithm, adding
    linear extra points at both ends of the interval.

    Arguments
    =========

        x_arr: coordinate array
        y_arr: 1-D or 2-D numpy array for interpolation
        xx: value for which y_arr must be interpolated
        indx: interpolation index such that
            x_arr[indx] < xx < x_arr[indx + 1].
            The minimum value is 0 and the maximum is len(x_arr) - 1.
        interp_list: list holding the interpolation coefficients.
            it should have the same size and dimensions as y_arr and
            initialized to None.
        return_coefs: If True, return the calculated interp_list[indx]
            instead of returning the interpolated y_arr

    '''

    # Get the dimensions and catch non-numpy arrays
    try:
        dimensions = y_arr.ndim
    except AttributeError:
        raise Exception("The interpolation routine uses numpy arrays")
    except:
        raise

    # Get the dimensions and catch non-numpy arrays
    try:
        dimensions = len(y_arr.shape)
    except AttributeError:
        raise Exception("The interpolation routine uses numpy arrays")
    except:
        raise

    # Return if last extreme
    if indx == len(x_arr) - 1:
        return y_arr[indx]

    # Check that the indx is lower than the maximum
    if indx > len(x_arr) - 1:
        raise Exception("Interpolating outside of range!")

    # Return the calculation with coefficients if exists
    if dimensions == 1:
        coefCheck = interp_list[indx]
    elif dimensions == 2:
        coefCheck = interp_list[indx][0]
    else:
        raise Exception("Current support for up to 2-d in interpolation method")

    if coefCheck is not None:
        if return_coefs:
            return interp_list[indx]
        else:
            coefs = interp_list[indx]
            deltx = xx - x_arr[indx]
            return coefs[0]*deltx**3 + coefs[1]*deltx**2 + coefs[2]*deltx + y_arr[indx]

    # If not, we have to calculate the coefficients for this region
    x0 = x_arr[indx]; xp1 = x_arr[indx + 1]

    # Store yp1 and y0
    yp1 = y_arr[indx + 1]; y0 = y_arr[indx]

    hi0 = (xp1 - x0)
    si0 = (yp1 - y0)/hi0

    # Calculate sim1 and deriv0
    if indx > 0:
        # Store x, y
        ym1 = y_arr[indx - 1]
        xm1 = x_arr[indx - 1]
    else:
        # We are in the lowest extreme, create an extra point
        dx = x_arr[indx + 1] - x_arr[indx]
        dy = y_arr[indx + 1] - y_arr[indx]
        xm1 = x_arr[indx] - dx*1e-5
        ym1 = y_arr[indx + 1] - dy/dx*1e-5

    him1 = (x0 - xm1)
    sim1 = (y0 - ym1)/him1

    # Pi0 calculation
    pi0 = (sim1*hi0 + si0*him1)/(him1 + hi0)

    # Derivative
    deriv0 = np.sign(sim1) + np.sign(si0)
    if dimensions == 1:
        deriv0 = deriv0*min(abs(sim1), abs(si0), 0.5*abs(pi0))
    elif dimensions == 2:
        deriv0 = deriv0*np.minimum(abs(sim1),\
                np.minimum(abs(si0), 0.5*abs(pi0)))

    # Calculate sip1, pip1 and derivp1
    if indx < len(x_arr) - 2:
        yp2 = y_arr[indx + 2]
        xp2 = x_arr[indx + 2]
    else:
        # We are in the highest extreme, create an extra point
        dx = x_arr[indx + 1] - x_arr[indx]
        dy = y_arr[indx + 1] - y_arr[indx]
        xp2 = x_arr[indx + 1] + dx*1e-5
        yp2 = y_arr[indx + 1] + dy/dx*1e-5

    hip1 = (xp2 - xp1)
    sip1 = (yp2 - yp1)/hip1

    # Pip1 calculation
    pip1 = (si0*hip1 + sip1*hi0)/(hi0 + hip1)

    # Derivative
    derivp1 = np.sign(si0) + np.sign(sip1)
    if dimensions == 1:
        derivp1 = derivp1*min(abs(si0), abs(sip1), 0.5*abs(pip1))
    elif dimensions == 2:
        derivp1 = derivp1*np.minimum(abs(si0), \
                np.minimum(abs(sip1), 0.5*abs(pip1)))

    # Now calculate coefficients (ci = deriv0; di = y0)
    ai = (deriv0 + derivp1 - 2*si0)/(hi0*hi0)
    bi = (3*si0 - 2*deriv0 - derivp1)/hi0

    interp_list[indx] = (ai, bi, deriv0)
    return interpolation(x_arr, y_arr, xx, indx, interp_list, return_coefs=return_coefs)
