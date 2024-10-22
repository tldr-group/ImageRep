import pybamm
import matplotlib.pyplot as plt


def update_phase_fraction(params, porosity):
    # cbd = 0.14
    # am_pf = am_pf + cbd
    am_pf = 1 - porosity
    original_am_pf = params["Positive electrode active material volume fraction"]
    original_am_pf += 0.07
    params["Positive electrode active material volume fraction"] = am_pf
    # params["Positive electrode thickness [m]"] *= (am_pf/original_am_pf)
    # print(f"electrode thickness = {params['Positive electrode thickness [m]']}")
    # params["Positive electrode density [kg.m-3]"] *= am_pf/original_am_pf
    params["Maximum concentration in positive electrode [mol.m-3]"] *= am_pf/original_am_pf
    
    params["Positive electrode porosity"] = porosity
    params["Positive electrode Bruggeman coefficient (electrode)"] = 1.5
    return params

def run_simulation(params):
    model = pybamm.lithium_ion.DFN()
    sim = pybamm.Simulation(model, parameter_values=params)
    current = params["Current function [A]"]
    if current < 1:
        sim.solve([0, 18500/current])
    else:
        sim.solve([0, 593])
    return sim

def plot_results(ax, sim, phase_fraction, param_values):
    discharge_capacity = sim.solution["Discharge capacity [A.h]"]
    voltage = sim.solution["Terminal voltage [V]"]
    print(f"Last Ah = {discharge_capacity.entries[-1]}")
    current = param_values["Current function [A]"]
    ax.plot(discharge_capacity.entries, voltage.entries, label=f"Porosity = {phase_fraction} (current = {current} A)")
    ax.set_xlabel("Discharge Capacity / Ah")
    ax.set_ylabel("Voltage / V")

def run_and_plot(ax, phase_fraction1, phase_fraction2, amps):
    for phase_fraction in [phase_fraction1, phase_fraction2]:
        # Reset param values
        param_values = pybamm.ParameterValues("Chen2020")
        param_values["Current function [A]"] = amps
        param_values = update_phase_fraction(param_values, phase_fraction)
        simulation = run_simulation(param_values)
        plot_results(ax, simulation, phase_fraction, param_values)
    ax.legend()
    

if __name__ == "__main__":
    # Load parameter values
    
    # Update parameter values
    phase_fraction1 = 0.351
    phase_fraction2 = 0.380
    
    # Create a 2x2 figure, showing on the right side the discharge curves and on the left
    # side the two screenshots taken from www.imagerep.io for the nrel dataset:

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # First image
    ax_screen1 = axs[0, 0]
    ax_screen1.imshow(plt.imread("paper_figures/figure_data/ImageRep_estimation_webapp.tiff"))
    ax_screen1.axis('off')
    ax_screen1.set_title("(a)")

    # Second image
    ax_screen2 = axs[1, 0]
    ax_screen2.imshow(plt.imread("paper_figures/figure_data/required_length.tiff"))
    ax_screen2.axis('off')
    ax_screen2.set_title("(c)")

    # Discharge curve for phase fraction 1
    ax_discharge1 = axs[0, 1]
    amps = 5.09/100
    c_rate = "C/100"
    run_and_plot(ax_discharge1, phase_fraction1, phase_fraction2, amps)
    ax_discharge1.set_title("(b)")

    

    # Discharge curve for phase fraction 2
    ax_discharge2 = axs[1, 1]
    amps = 5.09*3
    c_rate = "3C"
    run_and_plot(ax_discharge2, phase_fraction1, phase_fraction2, amps)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(hspace=0.5)
    ax_discharge2.set_title("(d)")

    plt.tight_layout()
    # save the figure
    plt.savefig('paper_figures/output/SI_discharge_curve_example.pdf', format='pdf', dpi=300)
    
