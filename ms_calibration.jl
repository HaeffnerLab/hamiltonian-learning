using IonSim
using QuantumOptics: timeevolution, stochastic
using StochasticDiffEq
using ScikitLearn
using Random
using Optim
using Distributions

function ms_calibration(detuning, pi_time, initial_guess, N, tspan_ideal, tspan_experiment, timescale=1e-6)

    function simulate_trap(tspan, detuning, pi_time)
        # defining trap parameters
        ca_ions = [Ca40(["S-1/2", "D-1/2"]), Ca40(["S-1/2", "D-1/2"])]
        chain = LinearChain(
            ions=ca_ions,
            com_frequencies=(x=3e6,y=3e6,z=1e6), 
            vibrational_modes=(;z=[1]),
        )
        lasers = [
            Laser(k = (x̂ + ẑ)/√2, ϵ = (x̂ - ẑ)/√2),
            Laser(k = (x̂ + ẑ)/√2, ϵ = (x̂ - ẑ)/√2)
        ]
        trap = Trap(configuration=chain, B=4e-4, Bhat=ẑ, δB=0, lasers=lasers)

        mode = trap.configuration.vibrational_modes.z[1]
        mode.N = 5

        Efield_from_pi_time!(pi_time, trap, 1, 1, ("S-1/2", "D-1/2"));
        Efield_from_pi_time!(pi_time, trap, 2, 1, ("S-1/2", "D-1/2"));
        
        Δ = detuning
        d = 100  # AC Stark shift compensation
        f = transition_frequency(trap, 1, ("S-1/2", "D-1/2"))
        
        detuned_lasers = [copy(lasers[1]), copy(lasers[2])]
        detuned_lasers[1].Δ = f + mode.ν + Δ - d
        detuned_lasers[2].Δ = f - mode.ν - Δ + d
        
        trap.lasers = detuned_lasers    

        h = hamiltonian(trap, timescale=1e-6)
        @time tout, sol = timeevolution.schroedinger_dynamic(
            tspan, ca_ions[1]["S-1/2"] ⊗ ca_ions[2]["S-1/2"] ⊗ mode[0],
            h)
        
        SS = real.(expect(ionprojector(trap, "S-1/2", "S-1/2"), sol))
        DD = real.(expect(ionprojector(trap, "D-1/2", "D-1/2"), sol))
        SD = real.(expect(ionprojector(trap, "S-1/2", "D-1/2"), sol))
        DS = real.(expect(ionprojector(trap, "D-1/2", "S-1/2"), sol))
        
        return tout, SS, DD, SD, DS
    end;

    tout, ideal_SS, ideal_DD, ideal_SD, ideal_DS = simulate_trap(tspan_ideal, detuning, pi_time);

    # function to simulate taking real data
    function simulate_experiment(n_shots, SS, DD, SD, DS)
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

    function log_likelihood(samples, num_experiments, times, θ₁, θ₂)
        """
        θ₁ = detuning
        θ₂ = pi_time
        """
        out = 0
        
        tout, p_SS, p_DD, p_SD, p_DS = simulate_trap(times, θ₁, θ₂)
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

    # run the experiment, learn the model, and reconstruct the curve from the learned model
    tout, exp_SS, exp_DD, exp_SD, exp_DS = simulate_trap(tspan_experiment, detuning, pi_time)
    samples = simulate_experiment(N, exp_SS, exp_DD, exp_SD, exp_DS)
    
    function objective(θ::Vector)
        return -log_likelihood(samples, N, tout, θ[1], θ[2])
    end
    res = optimize(objective, initial_guess)
    θ₁, θ₂ = Optim.minimizer(res)

    tout, pred_SS, pred_DD, pred_SD, pred_DS = simulate_trap(tspan_ideal, θ₁, θ₂)
    
    return Dict(
        "ideal_curve" => [
            Dict(
                "SS" => ideal_SS[i],
                "DD" => ideal_DD[i],
                "SD" => ideal_SD[i],
                "DS" => ideal_DS[i],
                )
            for i in eachindex(ideal_SS)],
        "ideal_fit_params" => [detuning, pi_time],
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
        "learned_fit_params" => [θ₁, θ₂],
    )
end

function ms_fidelity(actual_detuning, actual_pi_time, learned_detuning, learned_pi_time, timescale=1e-6)
    """
    This function assumes we experimentally don't touch the detuning and
    adjust only the laser power to attempt to reach pi_time = η / Δ,
    and then run the MS gate using a gate time of 1 / Δ.
    """
    ca_ions = [Ca40(["S-1/2", "D-1/2"]), Ca40(["S-1/2", "D-1/2"])]
    chain = LinearChain(
        ions=ca_ions,
        com_frequencies=(x=3e6,y=3e6,z=1e6), 
        vibrational_modes=(;z=[1]),
    )
    lasers = [
        Laser(k = (x̂ + ẑ)/√2, ϵ = (x̂ - ẑ)/√2),
        Laser(k = (x̂ + ẑ)/√2, ϵ = (x̂ - ẑ)/√2)
    ]
    trap = Trap(configuration=chain, B=4e-4, Bhat=ẑ, δB=0, lasers=lasers)

    mode = trap.configuration.vibrational_modes.z[1]
    mode.N = 5
    η = abs(get_η(mode, lasers[1], ca_ions[1]))
    
    # calculate the desired experimental pi time using the learned detuning
    desired_pi_time = η / learned_detuning
    
    # the resulting physical pi time will then be this desired pi time
    # adjusted by the ratio of actual/learned pi time
    physical_pi_time = desired_pi_time * (actual_pi_time/learned_pi_time)
    Efield_from_pi_time!(physical_pi_time, trap, 1, 1, ("S-1/2", "D-1/2"));
    Efield_from_pi_time!(physical_pi_time, trap, 2, 1, ("S-1/2", "D-1/2"));

    # the physical laser frequency will be the actual detuning
    Δ = actual_detuning

    d = 100  # AC Stark shift compensation
    f = transition_frequency(trap, 1, ("S-1/2", "D-1/2"))
    
    detuned_lasers = [copy(lasers[1]), copy(lasers[2])]
    detuned_lasers[1].Δ = f + mode.ν + Δ - d
    detuned_lasers[2].Δ = f - mode.ν - Δ + d
    
    trap.lasers = detuned_lasers
    
    # evolve for the learned gate time
    learned_gate_time = (1. / learned_detuning) / timescale
    tspan = [0, learned_gate_time]
    h = hamiltonian(trap, timescale=1e-6)
    @time tout, sol = timeevolution.schroedinger_dynamic(
        tspan, ca_ions[1]["S-1/2"] ⊗ ca_ions[2]["S-1/2"] ⊗ mode[0],
        h)

    # MS fidelity is overlap with desired Bell state
    bell_state = dm((ca_ions[1]["S-1/2"] ⊗ ca_ions[2]["S-1/2"] + 1im * ca_ions[1]["D-1/2"] ⊗ ca_ions[2]["D-1/2"])/√2) ⊗ one(mode)
    prob_bell = real.(expect(bell_state, sol))
    return prob_bell[end]
end