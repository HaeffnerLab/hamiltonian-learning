using IonSim
using QuantumOptics: timeevolution, stochastic
using StochasticDiffEq
using ScikitLearn
using Random
using Optim
using Distributions

# hard-coded physical parameters of the experiment
magnetic_field_gauss = 4.0
real_sym_ac_stark_shift_khz = 1.0
ν_ax_mhz = 1.0
ν_rad_mhz = 3.0

struct HamiltonianParameters
    detuning_khz::Float64
    pi_time_blue_μs::Float64
    pi_time_red_μs::Float64
    sym_ac_stark_shift_khz::Float64
    HamiltonianParameters(params::Vector) = new(params...)
end

function as_vector(θ::HamiltonianParameters)
    return [getproperty(θ, Symbol(name)) for name in fieldnames(HamiltonianParameters)]
end

function configure_trap(ions, δB=0)
    for ion in ions
        ion.stark_shift["S-1/2"] = real_sym_ac_stark_shift_khz * 1e3
    end
    chain = LinearChain(
        ions=ions,
        com_frequencies=(x=ν_rad_mhz*1e6, y=ν_rad_mhz*1e6, z=ν_ax_mhz*1e6), 
        vibrational_modes=(;z=[1]),
    )
    lasers = [
        Laser(k = (x̂ + ẑ)/√2, ϵ = (x̂ - ẑ)/√2),
        Laser(k = (x̂ + ẑ)/√2, ϵ = (x̂ - ẑ)/√2)
    ]
    trap = Trap(configuration=chain, B=magnetic_field_gauss*1e-4, Bhat=ẑ, δB=δB, lasers=lasers)

    mode = trap.configuration.vibrational_modes.z[1]
    mode.N = 10
    η = abs(get_η(mode, lasers[1], ions[1]))

    return trap, η
end

function get_experimental_probabilities(tspan, θ::HamiltonianParameters)
    ca_ions = [Ca40(), Ca40()]
    trap, _ = configure_trap(ca_ions)

    f = transition_frequency(trap, 1, ("S-1/2", "D-1/2"))
    Δ = θ.detuning_khz * 1e3    
    δ_blue = θ.sym_ac_stark_shift_khz * 1e3
    δ_red = θ.sym_ac_stark_shift_khz * 1e3
    
    mode = trap.configuration.vibrational_modes.z[1]
    detuned_lasers = [copy(trap.lasers[1]), copy(trap.lasers[2])]
    detuned_lasers[1].Δ = f + mode.ν + Δ + δ_blue
    detuned_lasers[2].Δ = f - mode.ν - Δ + δ_red
    
    trap.lasers = detuned_lasers
    
    Efield_from_pi_time!(θ.pi_time_blue_μs * 1e-6, trap, 1, 1, ("S-1/2", "D-1/2"));
    Efield_from_pi_time!(θ.pi_time_red_μs * 1e-6, trap, 2, 1, ("S-1/2", "D-1/2"));

    h = hamiltonian(trap, timescale=1e-6, rwa_cutoff=1e5)
    tout, sol = timeevolution.schroedinger_dynamic(
        tspan, ca_ions[1]["S-1/2"] ⊗ ca_ions[2]["S-1/2"] ⊗ mode[0],
        h)
    
    p_SS = real.(expect(ionprojector(trap, "S-1/2", "S-1/2"), sol))
    p_DD = real.(expect(ionprojector(trap, "D-1/2", "D-1/2"), sol))
    p_SD = real.(expect(ionprojector(trap, "S-1/2", "D-1/2"), sol))
    p_DS = real.(expect(ionprojector(trap, "D-1/2", "S-1/2"), sol))
    
    bell_state = dm((ca_ions[1]["S-1/2"] ⊗ ca_ions[2]["S-1/2"] + 1im * ca_ions[1]["D-1/2"] ⊗ ca_ions[2]["D-1/2"])/√2) ⊗ one(mode)
    p_bell = real.(expect(bell_state, sol))
    
    return p_SS, p_DD, p_SD, p_DS, p_bell, tout
end

function log_likelihood(samples, num_experiments, times, θ::HamiltonianParameters)
    out = 0    
    p_SS, p_DD, p_SD, p_DS, _ = get_experimental_probabilities(times, θ)
    for i in eachindex(times)
        probabilities = [p_SS[i], p_DD[i], p_SD[i], p_DS[i]] ./ (p_SS[i] + p_DD[i] + p_SD[i] + p_DS[i])
        dist = Multinomial(num_experiments, probabilities)
        term = logpdf(dist, [samples[i]["SS"], samples[i]["DD"], samples[i]["SD"], samples[i]["DS"]])
        if term != NaN
            out += term
        end
    end    
    return out
end

function get_samples_from_probabilities(n_shots, SS, DD, SD, DS)
    samples = []
    for i in eachindex(SS)
        sample = Dict("SS" => 0, "DD" => 0, "SD" => 0, "DS" => 0)
        for _ in 1:n_shots
            number = rand(Float64)
            if number <= SS[i]
                sample["SS"] += 1
            elseif number <= SS[i] + DD[i]
                sample["DD"] += 1
            elseif number <= SS[i] + DD[i] + SD[i]
                sample["SD"] += 1
            else
                sample["DS"] += 1
            end
        end
        push!(samples, sample)
    end
    return samples
end

function ms_calibration(
    θ_actual::HamiltonianParameters,
    θ_initial_guess::HamiltonianParameters,
    N::Int,
    tspan_ideal,
    tspan_experiment
)
    ideal_SS, ideal_DD, ideal_SD, ideal_DS, _ = get_experimental_probabilities(tspan_ideal, θ_actual);

    # run the experiment, learn the model, and reconstruct the curve from the learned model
    exp_SS, exp_DD, exp_SD, exp_DS, _, tout = get_experimental_probabilities(tspan_experiment, θ_actual)
    samples = get_samples_from_probabilities(N, exp_SS, exp_DD, exp_SD, exp_DS)

    function objective(θ::Vector)
        return -log_likelihood(samples, N, tout, HamiltonianParameters(θ))
    end
    
    res = optimize(objective, as_vector(θ_initial_guess))
    θ_learned = Optim.minimizer(res)
    θ_learned = HamiltonianParameters(θ_learned)

    pred_SS, pred_DD, pred_SD, pred_DS, _ = get_experimental_probabilities(tspan_ideal, θ_learned)
    
    return Dict(
        "ideal_curve" => [
            Dict(
                "SS" => ideal_SS[i],
                "DD" => ideal_DD[i],
                "SD" => ideal_SD[i],
                "DS" => ideal_DS[i],
                )
            for i in eachindex(ideal_SS)],
        "ideal_fit_params" => θ_actual,
        "experimental_data" => [
            Dict(
                "SS" => sample["SS"] / N,
                "DD" => sample["DD"] / N,
                "SD" => sample["SD"] / N,
                "DS" => sample["DS"] / N
                )
            for sample in samples],
        "learned_curve" => [
            Dict(
                "SS" => pred_SS[i],
                "DD" => pred_DD[i],
                "SD" => pred_SD[i],
                "DS" => pred_DS[i],
                )
            for i in eachindex(pred_SS)],
        "learned_fit_params" => θ_learned,
    )
end

function ms_fidelity(θ_actual, θ_learned)
    """
    This function assumes we experimentally don't touch the detuning,
    adjust the laser power to attempt to reach pi_time_μs = η / Δ,
    adjust the frequency to match the learned AC stark shift,
    and then run the MS gate using a gate time of 1 / Δ.
    """
    # the physical laser frequency will be the actual detuning
    physical_detuning_khz = θ_actual.detuning_khz

    # calculate the desired experimental pi time using the learned detuning,
    # and the resulting physical pi time will then be this desired pi time
    # adjusted by the ratio of actual/learned pi time
    _, η = configure_trap([Ca40(), Ca40()])
    desired_pi_time_μs = 1e6 * η / (θ_learned.detuning_khz * 1e3)
    physical_pi_time_blue_μs = desired_pi_time_μs * (θ_actual.pi_time_blue_μs/θ_learned.pi_time_blue_μs)    
    physical_pi_time_red_μs = desired_pi_time_μs * (θ_actual.pi_time_red_μs/θ_learned.pi_time_red_μs)

    # use the learned AC Stark shift value
    sym_ac_stark_shift_khz = θ_learned.sym_ac_stark_shift_khz

    θ_calibrated = HamiltonianParameters([
        physical_detuning_khz,
        physical_pi_time_blue_μs,
        physical_pi_time_red_μs,
        sym_ac_stark_shift_khz,
    ])

    # evolve for the learned gate time
    learned_gate_time_μs = 1e6 / (θ_learned.detuning_khz * 1e3)
    tspan = LinRange(0, learned_gate_time_μs, 50)
    _, _, _, _, p_bell, _ = get_experimental_probabilities(tspan, θ_calibrated)

    # MS fidelity is overlap with desired Bell state at final time
    return p_bell[end]
end