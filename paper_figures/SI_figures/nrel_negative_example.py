import pybamm
import matplotlib.pyplot as plt


def update_phase_fraction(params, am_pf):
    params["Positive electrode active material volume fraction"] = am_pf
    params["Positive electrode porosity"] = 1 - am_pf
    params["Positive electrode Bruggeman coefficient (electrode)"] = 1.5
    return params

def run_simulation(params):
    model = pybamm.lithium_ion.DFN()
    sim = pybamm.Simulation(model, parameter_values=params)
    sim.solve([0, 20000])
    return sim

def plot_results(sim, phase_fraction):
    discharge_capacity = sim.solution["Discharge capacity [A.h]"]
    voltage = sim.solution["Terminal voltage [V]"]
    print(f"Last Ah = {discharge_capacity.entries[-1]}")
    plt.plot(discharge_capacity.entries, voltage.entries, label=f"phase fraction = {phase_fraction}")
    plt.title("DFN model discharge curves")
    plt.xlabel("Discharge Capacity / Ah")
    plt.ylabel("Voltage / V")

def run_and_plot(param_values, phase_fraction):
    param_values = update_phase_fraction(param_values, phase_fraction)
    # Run simulation
    simulation = run_simulation(param_values)
    # Plot
    plot_results(simulation, phase_fraction)
    

if __name__ == "__main__":
    # Load parameter values
    param_values = pybamm.ParameterValues("Chen2020")
    param_values["Current function [A]"] = 5
    # Update parameter values
    phase_fraction1 = 0.484
    phase_fraction2 = 0.504
    run_and_plot(param_values, phase_fraction1)
    run_and_plot(param_values, phase_fraction2)
    plt.legend()

    # save the figure
    plt.savefig('paper_figures/output/Discharge_curve_example.pdf', format='pdf', dpi=300)
    
