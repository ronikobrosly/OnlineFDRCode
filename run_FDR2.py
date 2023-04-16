"""
GAI++ with memory algorithm
"""

import json
from os.path import expanduser
import pdb

import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum
import pandas as pd
from prophet import Prophet
import scipy.stats



######################################
## GET TIME SERIES DATA, FORECAST, AND P-VALUES
######################################

# Read in dataset
df = pd.read_csv("~/Desktop/OnlineFDRCode/nab/data/realKnownCause/ambient_temperature_system_failure.csv")
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"])

#initial_plot_df = df.set_index("ds")
#initial_plot_df.plot(linestyle='-', lw=0.5, color = {"y": "steelblue"})
#plt.show()

# Fit the model
m = Prophet(interval_width = 0.95, mcmc_samples = 0, uncertainty_samples = 10000)
m.fit(df)

# Extend dataset by 5-minute units (the units of the original dataset)
future = m.make_future_dataframe(periods=0, freq = 'H')
forecast = m.predict(future)
forecast = forecast[["ds", 'yhat_lower', "yhat", 'yhat_upper']]

# Append real values
forecast["actual"] = df["y"]

# Estimate standard error of predictions based on confidence interval width
forecast["lower_width"] = forecast["yhat"] - forecast["yhat_lower"]
forecast["upper_width"] = forecast["yhat_upper"] - forecast["yhat"]
forecast["std"] = np.mean(forecast[["lower_width", "upper_width"]], axis = 1) / scipy.stats.norm.ppf(1-.05/2)
forecast["z_score"] = np.abs((forecast["actual"] - forecast["yhat"]) / forecast["std"])
forecast["p_value"] = scipy.stats.norm.sf(forecast["z_score"])*2




######################################
## GET TIME SERIES DATA, FORECAST, AND P-VALUES
######################################




# Desired false discovery rate
fdr = 0.5

# How many hypotheses will you test? (This should equal the length of the `pvec` array)
numhyp = len(forecast["p_value"])

# Related to penalty weight (default = 0.1)
startfac = 0.1

# Memory parameter
mempar = 1

# Prior weight vector (by default, set all values to 1)
prw_vec = np.repeat(1., numhyp)
   
# Penalty weight vector (by default, set all values to 1)
penw_vec = np.repeat(1., numhyp)

# Discount factor
tmp = range(1, 8000)
gamma_vec =  np.true_divide(log(np.maximum(tmp, ones(len(tmp))*2)), np.multiply(tmp, exp(sqrt(log(np.maximum(ones(len(tmp)), tmp))))))
gamma_vec = gamma_vec / float(sum(gamma_vec))  

# Vector of p-values (this would be streamed in real life, and not known ahead of time)
pvec = forecast["p_value"].to_list()


def thresh_func(x):
    """
    If you want you can add different thresholds here, 
    but for now not doing anything
    """
    return x


def run_fdr(pvec, numhyp, gamma_vec, fdr, startfac, mempar, prw_vec, penw_vec):
    """
    Function to run the actual FDR calculations and determine which hypotheses to reject 
    """

    pen_min = min(penw_vec)
    w0 = min(pen_min, startfac)*fdr
    wealth_vec = np.zeros(numhyp + 1)
    wealth_vec[0] = w0
    alpha = np.zeros(numhyp + 1)
    alpha[0:2] = [0, gamma_vec[0]*w0]  # vector of alpha_js
    last_rej = 0 # save last time of rejection

    numhyp = len(pvec)
    last_rej = 0
    first = 0 # Whether we are past the first rejection
    flag = 1
    rej = np.zeros(numhyp+1)
    phi = np.zeros(numhyp+1)
    psi = np.zeros(numhyp+1)

    # Have to change that since prior weights need to be adjusted 
    r = [np.true_divide(thresh_func(penw_vec[i]), prw_vec[i]) for i in range(numhyp)]

    #penw, prw, pvec without k+1 (just shifted for purpose of having a wealth[0] (rest starts at 1)
    # here t=k+1 (t in paper)
    last_rej = []
    psi_rej = []

    for k in range(0, numhyp):                

        if wealth_vec[k] > 0:
            this_alpha = alpha[k+1] # make sure first one doesn't do bullshit

            # Calc b, phi
            b_k = fdr - (1-first)*np.true_divide(w0, penw_vec[k])
            phi[k + 1] = min(this_alpha, mempar*wealth_vec[k] + (1-mempar)*flag*w0)
            # Adjust prior weight depending on penw, phi, b - calc prw, r
            max_weight = (phi[k+1]*thresh_func(penw_vec[k]))/((1-b_k)*this_alpha)
            prw_vec[k] = min(prw_vec[k], max_weight)
            r[k] = np.true_divide(thresh_func(penw_vec[k]), prw_vec[k])

            # Calc psi
            # The max is to get rid of numerical issues when computing max_weight
            psi[k + 1] = max(min(phi[k + 1] + penw_vec[k]*b_k, np.true_divide(phi[k + 1], this_alpha)*r[k] - penw_vec[k] + penw_vec[k]*b_k),0)

        
            # Rejection decision
            rej[k + 1] = (pvec[k] < np.true_divide(this_alpha,r[k]))

            if (rej[k + 1] == 1):
                if (first == 0):
                    first = 1
                last_rej = np.append(last_rej, k + 1).astype(int)
                psi_rej = np.append(psi_rej, psi[k + 1]).astype(float)
            
            # Update wealth
            wealth = mempar*wealth_vec[k] + (1-mempar)*flag*w0 - phi[k + 1] + rej[k + 1]*psi[k + 1]

            # Calc new alpha
            
            if len(last_rej) > 0:
                # first_gam = mempar**(k+1-last_rej[0])*gamma_vec[k + 1 - last_rej[0]]
                #t_taoj = ((k+1)*np.ones(len(last_rej[0:-1]),dtype=int) - last_rej[0:-1])
                # sum_gam = sum(np.multiply(mempar**t_taoj, gamma_vec[t_taoj])) - first_gam + gamma_vec[k+1 - last_rej[-1]]
                # next_alpha = gamma_vec[k+1]*wealth_vec[0] + (fdr - w0/float(penw_vec[last_rej[0]]))*first_gam + fdr*sum_gam

                #gam_vec = np.append(np.multiply(mempar**t_taoj, gamma_vec[t_taoj]), gamma_vec[k + 1 - last_rej[-1]])

                t_taoj = ((k+1)*np.ones(len(last_rej),dtype=int) - last_rej)
                gam_vec = np.multiply(mempar**t_taoj, gamma_vec[t_taoj])

                next_alpha = gamma_vec[k + 1]*wealth_vec[0] + np.dot(psi_rej, gam_vec)

            else:
                sum_gam = 0
                next_alpha = gamma_vec[k+1]*wealth_vec[0]

            # next_alpha = mempar**(k+1-last_rej)*gamma_vec[k+1 - last_rej]*wealth_vec[last_rej]
            #next_alpha = gamma_vec[k+1 - last_rej]*wealth_vec[last_rej]
        else:
            break

        wealth_vec[k + 1] = wealth
        if k < numhyp-1:
            alpha[k+2] = next_alpha
        
        # After past the first reject, set flag to 0
        if (first == 1):
            flag = 0

    # Cut off the first zero
    rej = rej[1:]
    alpha = alpha[1:]

    return rej
    


rej = run_fdr(pvec, numhyp, gamma_vec, fdr, startfac, mempar, prw_vec, penw_vec)


results_df = pd.DataFrame(
    {
        'time_step': [i for i in np.arange(numhyp)],
        'pvalue': pvec,
        '-log10_pvalue': [-np.log10(i) for i in pvec],
        'reject': rej
    }
)

df_reject = results_df[results_df["reject"] == 1]

plt.figure(figsize=(14,8))
fig, axs = plt.subplots(2)
fig.suptitle(
    f"""Anomaly Detection Results\n(FDR = {fdr})""",
    fontsize=10
)


axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
initial_plot_df = df.set_index("ds")
axs[0].set_xlabel("Datetime", fontsize=6)
axs[0].set_ylabel("Value", fontsize=6)
axs[0].set_title("Time Series Forecast", fontsize=8)
axs[0].tick_params(axis='both', which='major', labelsize=5)
axs[0].scatter(df.index, df["y"], c = "green", alpha = 0.6, s = 0.1, label = "actual values")
axs[0].plot(forecast.index, forecast["yhat"], linestyle='-', lw=0.4, color = "black", label = "forecast values")
axs[0].fill_between(forecast.index, forecast["yhat_lower"], forecast["yhat_upper"], color = 'steelblue', edgecolor = None, alpha = 0.3, label = "confidence interval")
axs[0].legend(loc="upper left", fontsize=6)

axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].set_xlabel("Time step", fontsize=6)
axs[1].set_ylabel("-log10(p-value)", fontsize=6)
axs[1].set_title("Anomaly p-values", fontsize=8)
axs[1].tick_params(axis='both', which='major', labelsize=5)
axs[1].plot(results_df["time_step"], results_df["-log10_pvalue"], '-', linewidth=0.5, markersize=1.5)
axs[1].plot(df_reject["time_step"], df_reject["-log10_pvalue"], 'o', color = 'red', markersize=3)
axs[1].axhline(y=1.301, color='r', linewidth=0.8, linestyle='--', label = "p = 0.05")
axs[1].legend(loc="upper left", fontsize=6)
fig.tight_layout()
fig.savefig(expanduser("~/Desktop/ts_results.png"), dpi = 500)
