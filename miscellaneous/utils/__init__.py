import matplotlib.pyplot as plt
import os
import sys

# set stylesheet
# prj_path = "/Users/darien/Library/Mobile Documents/com~apple~CloudDocs/Code/QuantTrading/TradingProject/TradingLibADS"
# plt.style.use(os.path.join(prj_path, ".templates", "qb_dark.mplstyle"))


# plt.style.use(os.path.join(os.getcwd(), ".templates", "qb_dark.mplstyle"))

# load mpl custom stylesheet
try:
    plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".templates", "qb_dark.mplstyle"))
except:
    plt.style.use(os.path.join(os.getcwd(), ".templates", "qb_dark.mplstyle"))
finally:
    pass