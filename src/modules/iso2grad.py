"""

ISO2GRAD model v1

Function to predict gradient retention times from an isocratic model

Reference: Bolanca, T. et al. Development of an ion chromatographic gradient retention model from isocratic elution
           experiments. J. Chromatogr. A 2006, 1121, 228-235. (doi:10.1016/j.chroma.2006.04.036)

"""

# Importing modules/classes from the numpy package
from numpy import zeros, size, arange, hstack

""" 
Notes on the inputs to the function model:
1) qsrr_model is an object with the fitted model to predict logk
2) iso_data is a numpy matrix comprising of model predictors:
   [Alcohol (1/0) Dragon descriptors]; rows are analytes
3) t_void is a vector of void times (gradient elution); rows are gradient profiles, columns are analytes
4) grad_data is a numpy matrix comprising of:
   [tG_1, c(KOH)_1, slope_1, tG_2, c(KOH)_2, slope_2, tG_3, c(KOH)_3, slope_3, ...]; rows are gradient profiles
"""


def ig_model(qsrr_model, iso_data, t_void, grad_data, sc):

    # Increase all the matrix elements by an arbitrarily small number to prevent zero values
    grad_data = grad_data + 0.00000001

    # Integration step
    del_t = 0.01

    # Pre-define the matrix of gradient retention times
    tg_total = zeros((size(t_void, axis=0), size(t_void, axis=1)))

    # iso2grad algorithm
    # Loop through the gradient profiles
    for i in range(int(size(grad_data, axis=0))):

        # Pre-define tg1, tg2, i_prev & k1_2
        tg1, tg2, i_prev, i_partial, k1_2 = zeros((5, 1))
        tr_g = zeros(int(size(t_void, axis=1)))

        # Loop through the analytes
        for b in range(int(size(t_void, axis=1))):

            # Initialize next integration step
            i_next = 0

            # Loop through the gradient segments
            for p in range(0, int(size(grad_data, axis=1)-2), 3):

                # First segment of the gradient
                ti1 = grad_data[i, p]
                conc1 = grad_data[i, p+1]
                slope = grad_data[i, p+2]

                # Second segment of the gradient
                ti2 = grad_data[i, p+3]
                conc2 = grad_data[i, p+4]

                # Loop through the retention times
                for tg in arange(ti1, ti2, del_t):

                    tg1 = tg
                    tg2 = tg1 + del_t

                    if tg2 < ti2:

                        conc_grad1 = slope * (tg1 - ti1) + conc1
                        conc_grad2 = slope * (tg2 - ti1) + conc1

                        # Update the integration step
                        del_t_updt = del_t

                    else:

                        conc_grad1 = slope * (tg1 - ti1) + conc1
                        conc_grad2 = conc2

                        # Update the integration step
                        del_t_updt = ti2 - tg1

                    # Re-define the next integration step as the previous one
                    i_prev = i_next

                    # Predict k from the machine learning model for the two gradient concentrations
                    k1 = 10 ** qsrr_model.model.predict(sc.transform(hstack((conc_grad1, iso_data[b])).reshape((1, -1))))
                    k2 = 10 ** qsrr_model.model.predict(sc.transform(hstack((conc_grad2, iso_data[b])).reshape((1, -1))))

                    # Average k between the two gradient concentrations
                    k1_2 = (k2 + k1) / 2

                    # Update integral
                    i_partial = del_t_updt / k1_2
                    i_next = i_prev + i_partial

                    if i_prev < t_void[i, b] < i_next:
                        break

                if i_prev < t_void[i, b] < i_next:
                    break

            # Calculate retention time for a specified gradient
            tr_g[b] = t_void[i, b] + tg1 + (t_void[i, b] - i_prev) * k1_2
            # print('# Gradient profile #{}: tG({}) = {} min'.format(i+1, b+1, tr_g[b]))

        tg_total[i] = tr_g

    # Final matrix [number of gradient profiles × number of analytes]
    return tg_total
