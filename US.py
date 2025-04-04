import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import lognorm
from plotly.subplots import make_subplots



# Call and Put functions
def call(S, K, T, r, sigma, q=0):
    ''' Calcola il prezzo di un call con il metodo di Black-Scholes '''

    # calcolo dei parametri d1 e d2
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # calcolo del prezzo di un call con il metodo di Black-Scholes
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def put(S, K, T, r, sigma, q=0):
    ''' Calcola il prezzo di una put con il metodo di Black-Scholes '''

    # calcolo dei parametri d1 e d2
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # calcolo del prezzo di una put con il metodo di Black-Scholes
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)



# Title and introduction
col1, col2 = st.columns([3,1])
with col1: st.header('Montecarlo option pricing model')
with col2: st.markdown("""Created by 
    <a href="https://www.linkedin.com/in/davide-saccone/" target="_blank">
        <button style="background-color: #262730; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">
            Davide Saccone
        </button>
    </a>
    """, unsafe_allow_html=True)
st.write("To simulate how stock prices might change in the future, the model uses a method called geometric Brownian motion. This approach starts with the current price of the stock and projects future prices by combining two key elements: a deterministic trend, known as the drift, which represents the expected direction of the stock's movement over time, and random fluctuations, or volatility, which account for the uncertainty and variability in the market. \n \n At each step in the simulation, the model adjusts the stock price by applying both the drift (risk-free rate), and random changes that introduce unpredictability. These random changes are modeled using a log-normal distribution, which is appropriate because stock prices tend to grow exponentially over time and cannot go below zero. \n \n By repeating this process many times, the model generates a large number of possible future price paths. Each of these paths represents a different scenario of how the stock price could evolve, taking into account both the general market trend and the random nature of price movements. This collection of simulated paths provides a comprehensive view of the range of outcomes that the stock price might experience in the future.")



# sidebar parameters
st.sidebar.header('Input option parameters')
S = st.sidebar.number_input('Stock price (S)', min_value=0.00, value=100.00)
K = st.sidebar.number_input('Strike price (K)', min_value=0.00, value=100.00)
date = st.sidebar.date_input('Expiry Date', value=dt.datetime(2027,9,19))
date = dt.datetime.combine(date, dt.datetime.min.time())
T = (date - dt.datetime.today()).days / 365
sigma = st.sidebar.number_input('Volatility (in decimal)', min_value=0.00, value=0.20)
r = st.sidebar.number_input('Risk Free Rate (in decimal)', value=0.02)


# Black-Scholes price
BScall = call(S, K, T, r, sigma)
BSput = put(S, K, T, r, sigma)


# montecarlo simulation
np.random.seed(15)
n_simulations = 30000   
nSteps = int(T * 365)
dt = T/nSteps

S_paths = np.zeros((n_simulations, nSteps + 1))
S_paths[:,0] = S
for t in range(1, nSteps+1):
    Z = np.random.standard_normal(n_simulations)
    S_paths[:,t] = S_paths[:,t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)


payoff_call = np.maximum(S_paths[:,-1] - K, 0)
payoff_put = np.maximum(K - S_paths[:,-1], 0)

# Montecarlo price
MCcall = np.exp(-r*T) * np.mean(payoff_call)
MCput = np.exp(-r*T) * np.mean(payoff_put)


# Montecarlo graph
fig = go.Figure()
for i in range(15):
    fig.add_trace(go.Scatter(x=np.linspace(0, T, nSteps + 1), y=S_paths[i], mode='lines', showlegend=False))
fig.update_layout(title='Monte Carlo Simulation of Stock Price Paths', xaxis_title='Time (Years)', yaxis_title='Stock Price')
st.plotly_chart(fig)


 
# Plotting distribution
cola, colb = st.columns(2)
fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=S_paths[:, -1], nbinsx=50, histnorm='probability density', name='Simulated Data', marker_color="lightblue"))

shape, loc, scale = lognorm.fit(S_paths[:, -1], floc=0)
x = np.linspace(min(S_paths[:, -1]), max(S_paths[:, -1]), 1000)
pdf = lognorm.pdf(x, shape, loc, scale)
fig2.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='Lognormal Distribution', line=dict(color='#440154')))

fig2.update_layout(title="Distribution of final Stock Prices",xaxis_title='Stock Price', yaxis_title='Density', showlegend=False)
with cola:
    with st.expander("Distribution of final Stock Prices"):
        st.plotly_chart(fig2)



# Evolution of prices distribution
fig3 = go.Figure()
num_day = 8
nSteps = S_paths.shape[1]
random_steps = np.linspace(30, nSteps-2, num_day, dtype=int)
colors = plt.cm.viridis(np.linspace(0, 1, num_day))


for i, step in enumerate(random_steps):
    shape, loc, scale = lognorm.fit(S_paths[:, step], floc=0)
    x = np.linspace(min(S_paths[:, step]), max(S_paths[:, step]), 1000)
    pdf = lognorm.pdf(x, shape, loc, scale)
    
    color = f'rgba({colors[i][0]*255}, {colors[i][1]*255}, {colors[i][2]*255}, 1)'
    fig3.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name=f'{step}', line=dict(color=color)))


fig3.update_layout(title='Evolution of Prices Lognormal Distributions', xaxis_title='Stock Price', yaxis_title='Density', legend=dict(title='Day', traceorder='normal', yanchor="top", y=1, xanchor="left", x=0.9))
with colb:
    with st.expander("Evolution of Prices Lognormal Distributions"):
        st.plotly_chart(fig3)




st.write("")
st.header("European Options")
st.write("Monte Carlo simulations provide a way to model the potential future price paths of a stock by generating numerous hypothetical scenarios based on the stock's volatility. For each simulated path, the option's potential payoff is calculated, allowing us to capture a wide range of possible outcomes. By averaging these payoffs and discounting them to their present value, we obtain an estimate of the option's price, reflecting the comprehensive range of scenarios considered in the simulation.")
col1, col2 = st.columns(2)
with col1: st.metric(label='Montecarlo call option price', value=f"${MCcall:.2f}")
with col2: st.metric(label='Montecarlo put option price', value=f"${MCput:.2f}")

with col1:
    with st.expander("Black-Scholes call price"):
        st.write(f"**Call:** ${BScall:.2f}")

with col2:
    with st.expander("Black-Scholes put price"):
        st.write(f"**Put:** ${BSput:.2f}")



# Payoff plot
bins = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, max(max(payoff_call), max(payoff_put), 110)]
payoff_call_bins = pd.cut(payoff_call, bins=bins, right=False)
payoff_put_bins = pd.cut(payoff_put, bins=bins, right=False)

call_counts = payoff_call_bins.value_counts().sort_index()
put_counts = payoff_put_bins.value_counts().sort_index()

call_freq = call_counts / len(payoff_call)
put_freq = put_counts / len(payoff_put)

fig4 = go.Figure()
fig4.add_trace(go.Bar(x=[str(interval) for interval in call_counts.index], y=call_freq, name='Call Options Payoff', marker_color="lightblue"))
fig4.add_trace(go.Bar(x=[str(interval) for interval in put_counts.index], y=put_freq, name='Put Options Payoff', marker_color="#440154"))

fig4.update_layout(title="Options Payoff Frequency", xaxis_title="Payoff", yaxis_title="Frequency", showlegend=True, legend=dict(yanchor="top", y=1, xanchor="left", x=0.85))
st.plotly_chart(fig4)




st.write("")
st.write("")
st.header("American Options")
st.write("To price American options, which offer the flexibility to be exercised at any point before expiration, we utilize the Least Squares Monte Carlo (LSMC) method. This approach begins by simulating numerous potential future price paths for the underlying asset. At each point along these paths, the method evaluates whether it is more advantageous to exercise the option immediately or to continue holding it. This decision is made by comparing the immediate payoff from exercising the option with the estimated value of holding it, which is determined through a regression analysis of the future payoffs. \n The method systematically determines the optimal action—whether to hold or exercise the option—at each step of the simulation. The cash flows associated with these optimal actions are then discounted back to the present using the risk-free interest rate. Finally, by averaging the discounted values across all simulated paths, we arrive at an estimate of the option's price.")
st.write("NOTE: The Least Squares Monte Carlo (LSMC) method is generally effective in pricing American options, providing the added flexibility of early exercise. However, for very short times to expiry, it is possible that the American option price might be slightly lower than the European option price due to the approximation nature of the method. To ensure accuracy and fairness in pricing, the model compares both the American and European option prices and reports the higher of the two. This safeguard ensures that the option is valued correctly, reflecting the maximum potential benefit to the holder.")



# American call option pricing with montecarlo
def US_call(S, K, T, r, sigma):
    
    # Montecarlo simulation
    np.random.seed(15)
    n_simulations = 10000
    nSteps = int(T * 365)
    dt = T/nSteps

    path = np.zeros((n_simulations, nSteps + 1))
    path[:,0] = S

    for t in range(1, nSteps+1):
        Z = np.random.standard_normal(n_simulations)
        path[:,t] = path[:,t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    

    # Last day payoff
    cash_flows = np.maximum(path[:,-1] - K, 0)

    for t in range(nSteps-1, 0, -1):
        itm = path[:, t] > K
        X = path[itm, t]
        Y = cash_flows[itm] * np.exp(-r * dt)

        if len(X) > 0:
            regression = np.polyfit(X, Y, 2)

            continuation_values = np.polyval(regression, X)
            exercise_values = np.maximum(X - K, 0)
            cash_flows[itm] = np.where(exercise_values > continuation_values, exercise_values, cash_flows[itm])
    
    return np.mean(cash_flows)


# American put option pricing with montecarlo
def US_put(S, K, T, r, sigma):
    
    # Montecarlo simulation
    np.random.seed(15)
    n_simulations = 10000
    nSteps = int(T * 365)
    dt = T/nSteps

    path = np.zeros((n_simulations, nSteps + 1))
    path[:,0] = S

    for t in range(1, nSteps+1):
        Z = np.random.standard_normal(n_simulations)
        path[:,t] = path[:,t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    

    # Last day payoff
    cash_flows = np.maximum(K - path[:,-1], 0)

    # Calculating continuation values
    for t in range(nSteps-1, 0, -1):
        itm = path[:, t] < K
        X = path[itm, t]
        Y = cash_flows[itm] * np.exp(-r * dt)

        if len(X) > 0:
            regression = np.polyfit(X, Y, 2)
        
            continuation_values = np.polyval(regression, X)
            exercise_values = np.maximum(K - X, 0)
            cash_flows[itm] = np.where(exercise_values > continuation_values, exercise_values, cash_flows[itm])
    
    return np.mean(cash_flows)




american_call_price = US_call(S, K, T, r, sigma)
american_put_price = US_put(S, K, T, r, sigma)
UScall = max(american_call_price, BScall)
USput = max(american_put_price, BSput)

col3, col4 = st.columns(2)
with col3: st.metric(label='Montecarlo call option price', value=f"${UScall:.2f}")
with col4: st.metric(label='Montecarlo put option price', value=f"${USput:.2f}")




# Call prices comparison
min_price = min(BScall, MCcall, american_call_price) - 1
max_price = max(BScall, MCcall, american_call_price) + 1.5

fig5 = go.Figure()
fig5.add_trace(go.Bar(y=[BScall], x=["BS Call"], name='Black-Scholes European Call', marker_color="#483D8B"))
fig5.add_trace(go.Bar(y=[MCcall], x=["MC European Call"], name='Monte Carlo European Call Estimate', marker_color="#3CB371"))
fig5.add_trace(go.Bar(y=[american_call_price], x=["MC American Call"], name='Monte Carlo American Call Estimate', marker_color="#DAA520"))
fig5.update_layout(title="Comparison of Call Option Pricing Models", yaxis_title="Price", yaxis=dict(range=[min_price, max_price]), showlegend=True, legend=dict(yanchor="top", y=1, xanchor="left", x=0))



# Put prices comparison
min_price = min(BSput, MCput, american_put_price) - 1
max_price = max(BSput, MCput, american_put_price) + 1.5

fig6 = go.Figure()
fig6.add_trace(go.Bar(y=[BSput], x=["BS Put"], name='Black-Scholes European Put', marker_color="#483D8B"))
fig6.add_trace(go.Bar(y=[MCput], x=["MC European Put"], name='Monte Carlo European Put Estimate', marker_color="#3CB371"))
fig6.add_trace(go.Bar(y=[american_put_price], x=["MC American Put"], name='Monte Carlo American Put Estimate', marker_color="#DAA520"))
fig6.update_layout(title="Comparison of Put Option Pricing Models", yaxis_title="Price", yaxis=dict(range=[min_price, max_price]), showlegend=True, legend=dict(yanchor="top", y=1, xanchor="left", x=0))


with col3: st.plotly_chart(fig5)
with col4: st.plotly_chart(fig6)



