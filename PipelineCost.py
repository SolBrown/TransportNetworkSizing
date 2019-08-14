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

        # Create variables
        self.m.Q = pe.Var(self.m.arc_set, domain=pe.NonNegativeReals)
        self.m.Cost = pe.Var(self.m.arc_set, domain=pe.NonNegativeReals)        
        self.m.nx = pe.Var(self.m.arc_set, domain=pe.Binary,initialize=1)
        self.m.CaptureRate = pe.Var(self.m.node_set, domain=pe.NonNegativeReals)
        self.m.StorageRate = pe.Var(self.m.node_set, domain=pe.NonNegativeReals)
        self.m.PipelineDiameter  = pe.Var(self.m.arc_set, domain=pe.NonNegativeReals)
        
        #Create parameters
        self.m.Qmax = 100.
        self.m.TargetReduction = 100.
        # Create objective
        def obj_rule(m):
            return sum( m.nx[e] * m.Cost[e] * self.arc_data.ix[e,'Length'] for e in self.arc_set) + sum( m.CaptureRate[n] * 100. for n in self.node_set)
        self.m.OBJ = pe.Objective(rule=obj_rule, sense=pe.minimize)

        #Diameter calculations
        #def Diameter_rul(m, n1, n2):
        #    e = (n1,n2)
        #    f_f = 0.25*(np.log(eps/(3.7*d_i) + 5.74 / Re**0.9)**-2
        #    d_i = (2. * f_f * m.Y[e]**2 )/(np.pi**2 * self.m.rho * dPL)
        #    return m.D_o == d_i + 2.* t
        #self.m.Diam = pe.Constraint(self.m.node_set, rule=upper_bounds_rule)

                           
        # Flow Ballance rule
        def flow_bal_rule(m, n):
            arcs = self.arc_data.reset_index()
            preds = arcs[ arcs.End == n ]['Start']
            succs = arcs[ arcs.Start == n ]['End']
            return sum(m.Q[(p,n)] for p in preds) - sum(m.Q[(n,s)] for s in succs) + m.CaptureRate[n] - m.StorageRate[n] == 0
        self.m.FlowBal = pe.Constraint(self.m.node_set, rule=flow_bal_rule)

        # Build constraint
        def build_rule(m, n1, n2):
            e = (n1,n2)
            return m.Q[e] <= self.m.Qmax * m.nx[e]
        self.m.Build = pe.Constraint(self.m.arc_set, rule=build_rule)
        
        # Capture Bounds rule
        def capture_rate_rule(m, n):
            return m.CaptureRate[n] <= self.node_data.ix[n, 'Capture']
        self.m.CaptureBound = pe.Constraint(self.m.node_set, rule=capture_rate_rule)

        # Capture target rule
        def capture_target_rule(m):
            return sum(m.CaptureRate[n] for n in m.node_set) >= 15.
        self.m.CaptureRateBound = pe.Constraint(rule=capture_target_rule)

        # Storage Bounds rule
        def storage_rate_rule(m, n):
            return m.StorageRate[n] <= self.node_data.ix[n, 'Storage']
        self.m.StorageBound = pe.Constraint(self.m.node_set, rule=storage_rate_rule)

        # Storage target rule
        def storage_target_rule(m):
            return sum(m.CaptureRate[n] for n in m.node_set) == sum(m.StorageRate[n] for n in m.node_set)
        self.m.StorageRateBound = pe.Constraint( rule=storage_target_rule)
        
        #Simplified Cost model
        def Pipeline_Cost_rule(m, n1, n2):
            e = (n1,n2)
            return (  1.13 * m.Q[e] ) == m.Cost[e]
        self.m.Cost_Pipeline = pe.Constraint(self.m.arc_set, rule=Pipeline_Cost_rule)
        
    def solve(self):
        """Solve the model."""

        #results = pe.SolverFactory('mindtpy').solve(self.m, mip_solver='glpk', nlp_solver='ipopt')
        solver = pyomo.opt.SolverFactory('gurobi')
        results = solver.solve(self.m, tee=True, keepfiles=False, options_string="mip_tolerances_integrality=1e-9 mip_tolerances_mipgap=0")

        if (results.solver.status != pyomo.opt.SolverStatus.ok):
            logging.warning('Check solver not ok?')
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            logging.warning('Check solver optimality?') 


if __name__ == '__main__':
    sp = MinCostFlow('nodes.csv', 'arcs.csv') 
    sp.solve()
    print('\n\n---------------------------')
    print('Length: ', sp.m.OBJ())
    for e in sp.arc_set:
        print(pe.value(sp.m.Cost[e]),pe.value(sp.m.Q[e]),pe.value(sp.m.nx[e]),e)

    for e in sp.node_set:
        print(pe.value(sp.m.CaptureRate[e]),pe.value(sp.m.StorageRate[e]),e)
