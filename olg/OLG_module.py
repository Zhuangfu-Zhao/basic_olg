import numpy as np 
import matplotlib.pyplot as plt
from typing import Optional

class OLG_benchmark:

    _TAX_NAMES = {
        1: "Consumption tax endogenous",
        2: "Wage & interest tax endogenous",
        3: "Labor tax endogenous",
        4: "Interest tax endogenous",
    }

    def __init__(self, TT, n, gamma, beta, alpha, delta, tax_0, tax_1, damp = 0.25, tol = 0.00001, itermax = 1000): 

        # transitional periods [0, TT]
        self.TT = TT

        # parameters of households 
        self.n = n # cohort (not population) growth rate
        self.gamma = gamma # elasticity of intertemporal substitution 
        self.beta = beta # utility discount factor
        
        # parameters of firms 
        self.alpha = alpha # fraction of returns to capital
        self.delta = delta # depreciation rate

        # parameters of government 
        self.g = np.array([0.12, 0.12, 0])
        self.b_y = np.zeros(self.TT + 1) 
        self.kappa = np.zeros(self.TT + 1) 
        self.tau_c = np.zeros(self.TT + 1)
        self.tau_w = np.zeros(self.TT + 1)
        self.tau_r = np.zeros(self.TT + 1)
        self.tau_p = np.ones(self.TT + 1) * (self.kappa/((2+self.n)*(1+self.n)))  # initialize by SS value 

        # tax system reform
        self.tax_0 = tax_0 # priori-tax system
        self.tax_1 = tax_1 # reform-tax system
        self.tax = np.ones(self.TT + 1) * self.tax_0
        self.tax[1:] = self.tax_1

        # parameters of computation 
        self.damp = damp
        self.tol = tol
        self.itermax = itermax

        # initialize prices 
        self.r = np.zeros(self.TT + 1)
        self.w = np.zeros(self.TT + 1)
        self.wn = np.zeros(self.TT + 1)
        self.Rn = np.zeros(self.TT + 1)
        self.p = np.zeros(self.TT + 1)
        self.pen = np.zeros(self.TT + 1)

        # initialize consumptions and assets
        self.c = np.zeros((self.TT + 1, 3))
        self.a = np.zeros((self.TT + 1, 3)) # NOTE: index for decision variables statrs from 0 (but represents 1)

        # initialize aggregated variables
        self.LL = np.ones(self.TT + 1) * (1 + 1/(1+self.n)) # 不随时变 
        self.CC = np.zeros(self.TT + 1) 
        self.AA = np.zeros(self.TT + 1)  
        self.GG = np.zeros(self.TT + 1) 
        self.KK = np.ones(self.TT + 1) * 1
        self.YY = np.zeros(self.TT + 1) 
        self.BB = np.zeros(self.TT + 1) 
        self.II = np.zeros(self.TT + 1)

    def _prices(self, it): 

        # calculate prices given state variables

        self.r[it] = self.alpha*(self.KK[it]/self.LL[it])**(self.alpha-1)-self.delta 
        self.w[it] = (1-self.alpha)*(self.KK[it]/self.LL[it])**self.alpha 
        self.wn[it] = self.w[it]*(1-self.tau_w[it]-self.tau_p[it]) 
        self.Rn[it] = 1 + self.r[it]*(1-self.tau_r[it]) 
        self.p[it] = 1 + self.tau_c[it] 
        self.pen[it] = self.kappa[it]*self.w[max(it-1, 0)] # for steady state at 0 

    def _decisions(self, it): 

        # calculate consumptions and assets given prices 

        # recall the optimality principle. At period t: 
        # 1. The youngest cohort need to make decisions based on prices at t, t+1, t+2; 
        # 2. The middle-aged cohort need to make decisions based on prices at t, t+1 and assets from t-1; 
        # 3. The oldest-aged cohort need to make decisions based on prices at t and assets from t-1. 
        # prices are imported from _prices(). 
        # consumptions are computed according to Euler equation. 
        # assets from the last period will be computed in this method recursively. 

        # Time for steady states and transitional dynamics 
        if (it == 0 or it == self.TT): # steady states
            it1 = it2 = itm = it 
        else: # transitional dynamics
            it1 = min(it+1, self.TT)
            it2 = min(it+2, self.TT)
            itm = max(it-1, 0)

        # ==========================================================
        #      Outline of Consumption Calculation (Transition)
        # ----------------------------------------------------------
        # 1. Calculate all c[it,0] (c_1t in the model)
        # 2. Calculate c[1,1] and c[1,2] (c_21, c_31 in the model)
        # 3. Iterate according to Euler equation
        # ==========================================================

        # transitional dynamics and steady states
        PVI = self.wn[it] + self.wn[it1]/self.Rn[it1] + self.pen[it2]/(self.Rn[it1]*self.Rn[it2]) 
        coeff_c0 = 1/(self.p[it]*(1 + self.beta**self.gamma*(self.p[it1]/self.p[it]/self.Rn[it1])**(1-self.gamma) + 
                                 self.beta**(2*self.gamma)*(self.p[it2]/self.p[it]/self.Rn[it1]/self.Rn[it2])**(1-self.gamma)))
        self.c[it, 0] = coeff_c0 * PVI 

        if(it == 1): # transitional dynamics
            # c[1,1]: consumption of the middle-aged cohort at period 1 (alive at 0, 1, 2)
            self.a[it, 1] = self.wn[itm] - self.p[itm]*self.c[itm, 0]
            PVI = self.Rn[it]*self.a[it, 1] + self.wn[it] + self.pen[it1]/self.Rn[it1]
            coeff_c0 = 1/(self.p[it]*(1 + self.beta**self.gamma*(self.p[it1]/self.p[it]/self.Rn[it1])**(1-self.gamma)))
            self.c[it, 1] = coeff_c0 * PVI 
            # c[1,2]: consumption of the oldest cohort at period 1 (alive at -1, 0, 1)
            self.a[it, 2] = self.wn[itm] + self.Rn[itm]*self.a[itm, 1] - self.p[itm]*self.c[itm, 1]
            self.c[it, 2] = (self.pen[it] + self.Rn[it]*self.a[it, 2])/self.p[it] # just spend all assets left

        else: # transitional dynamics and steady states
            self.c[it, 1] = (self.beta*self.Rn[it]*self.p[itm]/self.p[it])**self.gamma*self.c[itm, 0]
            self.c[it, 2] = (self.beta*self.Rn[it]*self.p[itm]/self.p[it])**self.gamma*self.c[itm, 1]
            self.a[it, 1] = self.wn[itm] - self.p[itm]*self.c[itm, 0] 
            self.a[it, 2] = self.wn[itm] + self.Rn[itm]*self.a[itm, 1] - self.p[itm]*self.c[itm, 1]

    def _aggregations(self, it): 

        # aggregate varibales and set the procedure of iteration of KK[it]

        if (it == 0 or it == self.TT): 
            it1 = itm = it 
        else: 
            it1 = min(it+1, self.TT)
            itm = max(it-1, 0)

        self.CC[it] = self.c[it, 0] + self.c[it, 1]/(1+self.n) + self.c[it, 2]/((1+self.n)*(1+self.n))
        self.AA[it] = self.a[it, 1]/(1+self.n) + self.a[it, 2]/((1+self.n)*(1+self.n))
        self.GG[it] = self.g[0] + self.g[1]/(1+self.n) + self.g[2]/((1+self.n)*(1+self.n))

        # one round of iteration
        self.YY[it] = self.KK[it]**self.alpha * self.LL[it]**(1-self.alpha) # calculate YY[it] based on KK[it] from last iteration
        self.BB[it] = self.b_y[itm]*self.YY[it]
        self.KK[it] = self.damp*(self.AA[it]-self.BB[it]) + (1-self.damp)*self.KK[it] # calculate KK[it] for next round of iteration

        # calculate II[it] based on the updated KK[it]
        self.II[it] = (1+self.n)*self.KK[it1] - (1-self.delta)*self.KK[it] 

    def _government(self, it): 

        # calculate endogenous tax_rates of each tax system to balance the government budget 

        if (it == 0 or it == self.TT): 
            it1 = it 
        else: 
            it1 = min(it+1, self.TT) 

        if(self.tax[it] == 1): 
            self.tau_c[it] = ((1+self.r[it])*self.BB[it] + self.GG[it] - (self.tau_w[it]*self.w[it]*self.LL[it] + self.tau_r[it]*self.r[it]*self.AA[it] + (1+self.n)*self.BB[it1])) / self.CC[it]
        
        elif(self.tax[it] == 2):
            self.tau_w[it] = ((1+self.r[it])*self.BB[it] + self.GG[it] - (self.tau_c[it]*self.CC[it] + (1+self.n)*self.BB[it1]))/(self.w[it]*self.LL[it] + self.r[it]*self.AA[it])
            self.tau_r[it] = self.tau_w[it]

        elif(self.tax[it] == 3):
            self.tau_w[it] = ((1+self.r[it])*self.BB[it] + self.GG[it] - (self.tau_c[it]*self.CC[it] + self.tau_r[it]*self.r[it]*self.AA[it] + (1+self.n)*self.BB[it1]))/(self.w[it]*self.LL[it])
        
        else:
            self.tau_r[it] = ((1+self.r[it])*self.BB[it] + self.GG[it] - (self.tau_c[it]*self.CC[it] + self.tau_w[it]*self.w[it]*self.LL[it] + (1+self.n)*self.BB[it1])) / (self.r[it]*self.AA[it])

        self.tau_p[it] = (self.pen[it]/((2+self.n)*(1+self.n))) / self.w[it] 

    def get_SteadyState(self): 

        for _ in range(self.itermax):
            self._prices(0)
            self._decisions(0)
            self._aggregations(0) 
            self._government(0)
            if(abs(self.YY[0] - self.CC[0] - self.II[0] - self.GG[0])/self.YY[0] < self.tol):
                break 
    
    def get_Transition(self): 

        for _ in range(self.itermax): # Gauss-Seidel iteration (since the model is forward-looking, we need to iterate KK[:] as a whole)

            for it in range(1, self.TT+1): 
                self._prices(it) 
            for it in range(1, self.TT+1): 
                self._decisions(it) 
            for it in range(1, self.TT+1): 
                self._aggregations(it) 
            for it in range(1, self.TT+1): 
                self._government(it) 

            n_market = 0 
            for it in range(1, self.TT+1): 
                if(abs(self.YY[it] - self.CC[it] - self.II[it] - self.GG[it])/self.YY[it] < self.tol): 
                    n_market = n_market + 1 
            
            if(n_market == self.TT): 
                break   
    
    def plot_cohort_consumption(self, figsize: tuple = (10, 4), save_path: Optional[str] = None):

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
        titles = ["Young  c(t, 0)", "Middle  c(t, 1)", "Old  c(t, 2)"]
        colors = ["steelblue", "darkorange", "forestgreen"]
        t = np.arange(0, self.TT + 1)

        for i, (ax, title, col) in enumerate(zip(axes, titles, colors)):
            ax.plot(t, self.c[:, i], color=col, linewidth=2)
            ax.axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5, label="Reform")
            ax.axhline(self.c[0, i], color="grey", linewidth=0.9, linestyle="--", alpha=0.7, label="Initial SS")
            ax.axhline(self.c[self.TT, i], color="black", linewidth=0.9, linestyle=":", alpha=0.7, label="Final SS")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Period")
            ax.set_ylabel("Consumption level")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, linestyle=":")

        fig.suptitle(
            f"Cohort Consumption Paths\n Tax reform: {self._TAX_NAMES[self.tax_0]}  →  {self._TAX_NAMES[self.tax_1]}",
            fontweight="bold", fontsize=11,
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Figure saved → {save_path}")
        else:
            plt.show()

    def plot_aggregate_consumption(self, figsize: tuple = (8, 4), save_path: Optional[str] = None):

        fig, ax = plt.subplots(figsize=figsize)
        t = np.arange(0, self.TT + 1)

        ax.plot(t, self.CC, color="steelblue", linewidth=2)
        ax.axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5, label="Reform")
        ax.axhline(self.CC[0], color="grey", linewidth=0.9, linestyle="--", alpha=0.7, label="Initial SS")
        ax.axhline(self.CC[self.TT], color="black", linewidth=0.9, linestyle=":", alpha=0.7, label="Final SS")
        ax.set_title("Aggregate Consumption Path", fontweight="bold")
        ax.set_xlabel("Period")
        ax.set_ylabel("Consumption level")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=":")

        fig.suptitle(
            f"Cohort Consumption Paths\n Tax reform: {self._TAX_NAMES[self.tax_0]}  →  {self._TAX_NAMES[self.tax_1]}",
            fontweight="bold", fontsize=11,
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Figure saved → {save_path}")
        else:
            plt.show()

    def plot_cohort_assets(self, figsize: tuple = (10, 4), save_path: Optional[str] = None):

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
        titles = ["Young  a(t, 0)", "Middle  a(t, 1)", "Old  a(t, 2)"]
        colors = ["steelblue", "darkorange", "forestgreen"]
        t = np.arange(0, self.TT + 1)

        for i, (ax, title, col) in enumerate(zip(axes, titles, colors)):
            ax.plot(t, self.a[:, i], color=col, linewidth=2)
            ax.axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5, label="Reform")
            ax.axhline(self.a[0, i], color="grey", linewidth=0.9, linestyle="--", alpha=0.7, label="Initial SS")
            ax.axhline(self.a[self.TT, i], color="black", linewidth=0.9, linestyle=":", alpha=0.7, label="Final SS")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Period")
            ax.set_ylabel("Asset level")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, linestyle=":")

        fig.suptitle(
            f"Cohort Asset Paths\n Tax reform: {self._TAX_NAMES[self.tax_0]}  →  {self._TAX_NAMES[self.tax_1]}",
            fontweight="bold", fontsize=11,
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Figure saved → {save_path}")
        else:
            plt.show()

    def plot_aggregate_assets(self, figsize: tuple = (8, 4), save_path: Optional[str] = None):

        fig, ax = plt.subplots(figsize=figsize)
        t = np.arange(0, self.TT + 1)

        ax.plot(t, self.KK, color="darkorange", linewidth=2)
        ax.axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5, label="Reform")
        ax.axhline(self.KK[0], color="grey", linewidth=0.9, linestyle="--", alpha=0.7, label="Initial SS")
        ax.axhline(self.KK[self.TT], color="black", linewidth=0.9, linestyle=":", alpha=0.7, label="Final SS")
        ax.set_title("Capital Stock Path", fontweight="bold")
        ax.set_xlabel("Period")
        ax.set_ylabel("Capital level")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=":")
        
        fig.suptitle(
            f"Cohort Asset Paths\n Tax reform: {self._TAX_NAMES[self.tax_0]}  →  {self._TAX_NAMES[self.tax_1]}",
            fontweight="bold", fontsize=11,
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Figure saved → {save_path}")
        else:
            plt.show()

    def plot_tax_rates(self, figsize: tuple = (10, 4), save_path: Optional[str] = None):

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
        titles = [r"$\tau_c$ (Consumption tax)", r"$\tau_w$ (Wage tax)", r"$\tau_r$ (Interest tax)"]
        colors = ["steelblue", "darkorange", "forestgreen"]
        data = [self.tau_c, self.tau_w, self.tau_r]
        t = np.arange(0, self.TT + 1)

        for i, (ax, title, col, dat) in enumerate(zip(axes, titles, colors, data)):
            ax.plot(t, dat, color=col, linewidth=2)
            ax.axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5, label="Reform")
            ax.axhline(dat[0], color="grey", linewidth=0.9, linestyle="--", alpha=0.7, label="Initial SS")
            ax.axhline(dat[self.TT], color="black", linewidth=0.9, linestyle=":", alpha=0.7, label="Final SS")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Period")
            ax.set_ylabel("Tax rate")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, linestyle=":")

        fig.suptitle(
            f"Tax rates Paths\n Tax reform: {self._TAX_NAMES[self.tax_0]}  →  {self._TAX_NAMES[self.tax_1]}",
            fontweight="bold", fontsize=11,
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Figure saved → {save_path}")
        else:
            plt.show()

    def plot_output_components(self, figsize: tuple = (10, 4), save_path: Optional[str] = None):

        fig, ax = plt.subplots(figsize=figsize)
        t = np.arange(0, self.TT + 1)

        ax.plot(t, self.YY, color="steelblue", linewidth=2, label="Output Y")
        ax.plot(t, self.CC, color="darkorange", linewidth=2, label="Consumption C")
        ax.plot(t, self.II, color="forestgreen", linewidth=2, label="Investment I")
        ax.axhline(self.GG[0], color="red", linewidth=0.8, linestyle="--", alpha=0.7, label=f"Government G")
        ax.axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5, label="Reform")
        ax.set_title("Output and Components", fontweight="bold")
        ax.set_xlabel("Period")
        ax.set_ylabel("Level")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=":")
        
        fig.suptitle(
            f"Cohort Consumption Paths\n Tax reform: {self._TAX_NAMES[self.tax_0]}  →  {self._TAX_NAMES[self.tax_1]}",
            fontweight="bold", fontsize=11,
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Figure saved → {save_path}")
        else:
            plt.show()

    def plot_all_variables(self, figsize: tuple = (12, 10), save_path: Optional[str] = None):

        fig, axes = plt.subplots(3, 3, figsize=figsize)
        t = np.arange(0, self.TT + 1)

        axes[0, 0].plot(t, self.CC, color="steelblue", linewidth=2)
        axes[0, 0].axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5)
        axes[0, 0].set_title("Aggregate Consumption", fontweight="bold", fontsize=9)
        axes[0, 0].set_xlabel("Period")
        axes[0, 0].grid(True, alpha=0.3, linestyle=":")

        axes[0, 1].plot(t, self.KK, color="darkorange", linewidth=2)
        axes[0, 1].axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5)
        axes[0, 1].set_title("Capital Stock", fontweight="bold", fontsize=9)
        axes[0, 1].set_xlabel("Period")
        axes[0, 1].grid(True, alpha=0.3, linestyle=":")

        axes[0, 2].plot(t, self.YY, color="forestgreen", linewidth=2, label="Y")
        axes[0, 2].plot(t, self.II, color="red", linewidth=2, linestyle="--", label="I")
        axes[0, 2].axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5)
        axes[0, 2].set_title("Output & Investment", fontweight="bold", fontsize=9)
        axes[0, 2].set_xlabel("Period")
        axes[0, 2].legend(fontsize=7)
        axes[0, 2].grid(True, alpha=0.3, linestyle=":")

        axes[1, 0].plot(t, self.tau_c, color="steelblue", linewidth=2)
        axes[1, 0].axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5)
        axes[1, 0].set_title(r"$\tau_c$ (Consumption tax)", fontweight="bold", fontsize=9)
        axes[1, 0].set_xlabel("Period")
        axes[1, 0].grid(True, alpha=0.3, linestyle=":")

        axes[1, 1].plot(t, self.tau_w, color="darkorange", linewidth=2)
        axes[1, 1].axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5)
        axes[1, 1].set_title(r"$\tau_w$ (Wage tax)", fontweight="bold", fontsize=9)
        axes[1, 1].set_xlabel("Period")
        axes[1, 1].grid(True, alpha=0.3, linestyle=":")

        axes[1, 2].plot(t, self.tau_r, color="forestgreen", linewidth=2)
        axes[1, 2].axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5)
        axes[1, 2].set_title(r"$\tau_r$ (Interest tax)", fontweight="bold", fontsize=9)
        axes[1, 2].set_xlabel("Period")
        axes[1, 2].grid(True, alpha=0.3, linestyle=":")

        axes[2, 0].plot(t, self.c[:, 0], color="steelblue", linewidth=2)
        axes[2, 0].axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5)
        axes[2, 0].set_title("Young Consumption", fontweight="bold", fontsize=9)
        axes[2, 0].set_xlabel("Period")
        axes[2, 0].grid(True, alpha=0.3, linestyle=":")

        axes[2, 1].plot(t, self.c[:, 1], color="darkorange", linewidth=2)
        axes[2, 1].axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5)
        axes[2, 1].set_title("Middle Consumption", fontweight="bold", fontsize=9)
        axes[2, 1].set_xlabel("Period")
        axes[2, 1].grid(True, alpha=0.3, linestyle=":")

        axes[2, 2].plot(t, self.c[:, 2], color="forestgreen", linewidth=2)
        axes[2, 2].axvline(1, color="red", linewidth=0.8, linestyle="-.", alpha=0.5)
        axes[2, 2].set_title("Old Consumption", fontweight="bold", fontsize=9)
        axes[2, 2].set_xlabel("Period")
        axes[2, 2].grid(True, alpha=0.3, linestyle=":")

        fig.suptitle(
            f"Transitional Dynamics Overview\n Tax reform: {self._TAX_NAMES[self.tax_0]}  →  {self._TAX_NAMES[self.tax_1]}",
            fontweight="bold", fontsize=12,
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Figure saved → {save_path}")
        else:
            plt.show()

    def info(self):
        tax0 = self._TAX_NAMES.get(self.tax_0, f"Unknown ({self.tax_0})")
        tax1 = self._TAX_NAMES.get(self.tax_1, f"Unknown ({self.tax_1})")

        lines = [
            "",
            "=" * 40,
            "  OLG Model - Economic Setup",
            "=" * 40,
            "",
            "Household Parameters",
            f"  Periods (TT)          : {self.TT}",
            f"  Cohort growth (n)     : {self.n:.3f}",
            f"  EIS (gamma)           : {self.gamma:.3f}",
            f"  Discount (beta)       : {self.beta:.3f}",
            "",
            "Firm Parameters",
            f"  Capital share (alpha) : {self.alpha:.3f}",
            f"  Depreciation (delta)  : {self.delta:.3f}",
            "",
            "Government",
            f"  Public spending (g)   : [{self.g[0]}, {self.g[1]}, {self.g[2]}]",
            f"  Initial tax system    : {tax0} ({self.tax_0})",
            f"  Reform tax system     : {tax1} ({self.tax_1})",
            "",
            "Solver Settings",
            f"  Damping               : {self.damp:.3f}",
            f"  Tolerance             : {self.tol:.2e}",
            f"  Max iterations        : {self.itermax}",
            "",
            "=" * 40,
            "",
        ]
        print("\n".join(lines))