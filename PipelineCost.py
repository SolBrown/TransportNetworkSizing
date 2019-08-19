import pyomo
import pandas
import pyomo.opt
import pyomo.environ as pe
import numpy as np

class MinCostFlow:
    """This class implements a standard min-cost-flow model.  
    
    It takes as input two csv files, providing data for the nodes and the arcs of the network.  The nodes file should have columns:
    
    Node, Imbalance

    that specify the node name and the flow imbalance at the node.  The arcs file should have columns:

    Start, End, Length, UpperBound, LowerBound

    that specify an arc start node, an arc end node, a cost for the arc, and upper and lower bounds for the flow."""
    def __init__(self, nodesfile, arcsfile):
        """Read in the csv data."""
        # Read in the nodes file
        self.node_data = pandas.read_csv('nodes.csv')
        self.node_data.set_index(['Node'], inplace=True)
        self.node_data.sort_index(inplace=True)
        # Read in the arcs file
        self.arc_data = pandas.read_csv('arcs_second.csv')
        self.arc_data.set_index(['Start','End'], inplace=True)
        self.arc_data.sort_index(inplace=True)

        self.node_set = self.node_data.index.unique()
        self.arc_set = self.arc_data.index.unique()

        self.createModel()

    def createModel(self):
        """Create the pyomo model given the csv data."""
        self.m = pe.ConcreteModel()

        # Create sets
        self.m.node_set = pe.Set( initialize=self.node_set )
        self.m.arc_set = pe.Set( initialize=self.arc_set , dimen=2)


        self.m.T = pe.RangeSet(2)
        #Create parameters
        self.m.Qmax = 100.
        self.m.TargetReduction = {1: 20., 2: 30} 

        def Pipeline_routing_block(b, t):
            # Create variables
            b.Q = pe.Var(self.m.arc_set, domain=pe.NonNegativeReals)
            b.Cost = pe.Var(self.m.arc_set, domain=pe.NonNegativeReals)        
            b.nx = pe.Var(self.m.arc_set, domain=pe.Binary,initialize=1)
            b.nx0 = pe.Var(self.m.arc_set, domain=pe.Binary)
            b.xt = pe.Var(self.m.arc_set, domain=pe.Binary)
            b.CaptureRate = pe.Var(self.m.node_set, domain=pe.NonNegativeReals)
            b.StorageRate = pe.Var(self.m.node_set, domain=pe.NonNegativeReals)
            #b.PipelineDiameter  = pe.Var(self.m.arc_set, domain=pe.NonNegativeReals)
            #Diameter calculations
            #def Diameter_rul(m, n1, n2):
            #    e = (n1,n2)
            #    f_f = 0.25*(np.log(eps/(3.7*d_i) + 5.74 / Re**0.9)**-2
            #    d_i = (2. * f_f * m.Y[e]**2 )/(np.pi**2 * self.m.rho * dPL)
            #    return m.D_o == d_i + 2.* t
            #self.m.Diam = pe.Constraint(self.m.node_set, rule=upper_bounds_rule)

                           
            # Flow Ballance rule
            def flow_bal_rule(b, n):
                arcs = self.arc_data.reset_index()
                preds = arcs[ arcs.End == n ]['Start']
                succs = arcs[ arcs.Start == n ]['End']
                return sum(b.Q[(p,n)] for p in preds) - sum(b.Q[(n,s)] for s in succs) + b.CaptureRate[n] - b.StorageRate[n] == 0
            b.FlowBal = pe.Constraint(self.m.node_set, rule=flow_bal_rule)

            # Build constraint
            def build_rule(b, n1, n2):
                e = (n1,n2)
                return b.Q[e] <= self.m.Qmax * b.nx[e]
            b.Build = pe.Constraint(self.m.arc_set, rule=build_rule)
        
            # Capture Bounds rule
            def capture_rate_rule(b, n):
                return b.CaptureRate[n] <= self.node_data.ix[n, 'Capture']
            b.CaptureBound = pe.Constraint(self.m.node_set, rule=capture_rate_rule)

            # Capture target rule
            def capture_target_rule(b):
                return sum(b.CaptureRate[n] for n in self.m.node_set) >= self.m.TargetReduction[t]
            b.CaptureRateBound = pe.Constraint( rule=capture_target_rule)
            
            # Storage Bounds rule
            def storage_rate_rule(b, n):
                return b.StorageRate[n] <= self.node_data.ix[n, 'Storage']
            b.StorageBound = pe.Constraint(self.m.node_set, rule=storage_rate_rule)

            # Storage target rule
            def storage_target_rule(b):
                return sum(b.CaptureRate[n] for n in self.m.node_set) == sum(b.StorageRate[n] for n in self.m.node_set)
            b.StorageRateBound = pe.Constraint( rule=storage_target_rule)
        
            #Simplified Cost model
            def Pipeline_Cost_rule(b, n1, n2):
                e = (n1,n2)
                return (  1.13 * b.Q[e] ) == b.Cost[e]
            b.Cost_Pipeline = pe.Constraint(self.m.arc_set, rule=Pipeline_Cost_rule)

            def build_previous_rule(b, n1, n2):
                e = (n1,n2)
                return b.nx[e] == b.nx0[e] + b.xt[e]
            b.BuildPrevious = pe.Constraint(self.m.arc_set, rule=build_previous_rule)

        # create block for a single time period
        self.m.pr = pe.Block(self.m.T, rule = Pipeline_routing_block)
        
        # Create objective
        def obj_rule(m):
            return sum( sum( m.pr[t].nx[e] * m.pr[t].Cost[e] * self.arc_data.ix[e,'Length'] for e in self.arc_set) + sum( m.pr[t].CaptureRate[n] * 100. for n in self.node_set) for t in m.T)
        self.m.OBJ = pe.Objective(rule=obj_rule, sense=pe.minimize)

        

        
    def solve(self):
        """Solve the model."""

        #results = pe.SolverFactory('mindtpy').solve(self.m, mip_solver='glpk', nlp_solver='ipopt')
        solver = pyomo.opt.SolverFactory('gurobi')
        results = solver.solve(self.m, tee=True, keepfiles=False, options_string="mip_tolerances_integrality=1e-9 mip_tolerances_mipgap=0")

        if (results.solver.status != pyomo.
            opt.SolverStatus.ok):
            logging.warning('Check solver not ok?')
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            logging.warning('Check solver optimality?') 


if __name__ == '__main__':
    sp = MinCostFlow('nodes.csv', 'arcs.csv') 
    sp.solve()
    print('\n\n---------------------------')
    print('Length: ', sp.m.OBJ())
    for e in sp.arc_set:
        print(pe.value(sp.m.pr[2].Cost[e]),pe.value(sp.m.pr[2].Q[e]),pe.value(sp.m.pr[2].nx[e]),e)

    for e in sp.node_set:
        print(pe.value(sp.m.pr[2].CaptureRate[e]),pe.value(sp.m.pr[2].StorageRate[e]),e)
